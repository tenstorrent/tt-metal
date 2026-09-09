#include "propagate.h"
#include "assign.h"
#include "macros.h"
#include "message.h"
#include "ruler.h"
#include "utilities.h"

struct watch *ring_propagate (struct ring *ring, bool stop_at_conflict,
                              struct clause *ignore) {
  assert (!ring->inconsistent);
  assert (!ignore || !is_binary_pointer (ignore));
  struct ring_trail *trail = &ring->trail;
  struct watch *conflict = 0;
#ifdef METRICS
  uint64_t *visits = ring->statistics.contexts[ring->context].visits;
#endif
  signed char *values = ring->values;
  uint64_t ticks = 0, propagations = 0;
  while (trail->propagate != trail->end) {
    if (stop_at_conflict && conflict)
      break;
    unsigned lit = *trail->propagate++;
    LOG ("propagating %s", LOGLIT (lit));
    propagations++;
    unsigned not_lit = NOT (lit);
    struct references *watches = &REFERENCES (not_lit);

    // First traverse all irredundant (globally shared) binary clauses
    // with this literal (negation of propagated one) if there are any.
    // These binary clauses are encoded by a flat array of the 'other'
    // literals in binary clauses for each literal (including this one)
    // and only need to be allocated once and can thus be shared among all
    // threads, since these irredundant binary clauses do not change
    // during search (and are collected during cloning of rings).

    unsigned *binaries = watches->binaries;
    if (binaries) {
      unsigned other, *p;
      for (p = binaries; (other = *p) != INVALID; p++) {
        struct watch *watch = tag_binary (false, other, not_lit);
        signed char other_value = values[other];
        if (other_value < 0) {
          conflict = watch;
          if (stop_at_conflict)
            break;
        } else if (!other_value) {
          struct watch *reason = tag_binary (false, other, not_lit);
          assign_with_reason (ring, other, reason);
          ticks++;
        }
      }

      ticks += cache_lines (p, binaries);
      if (stop_at_conflict && conflict)
        break;
    }

    // Then traverse (and update) the watch list of the literal.

    struct watch **begin = watches->begin, **q = begin;
    struct watch **end = watches->end, **p = begin;

    ticks++;

    while (p != end) {
      assert (!stop_at_conflict || !conflict);
      struct watch *watch = *q++ = *p++;

      // This tagged 'watch' pointer is either a binary watch or an
      // index to the corresponding watcher in the (ring/thread local)
      // watcher stack.  Tagging uses bit-stuffing to distinguish these
      // two cases, through the least significand bit actually.

      // If the clause is binary (least significand bit set) we find the
      // other literal of the binary clause in the upper half of the
      // pointer (together with another bit which encodes redundancy).
      // The lower half of the pointer encodes the negation of the
      // propagated literal.

      // For larger (non-binary) clauses we have a similar situation and
      // store in the upper half of the watch pointer word the blocking
      // literal (conceptually an abitrary literal of the clause but
      // supposed to be different from the negated propagaged literal).
      // The other literal of binary clauses plays the same role.

      // Now we check first, which often happens, whether this blocking
      // literal of both binary and large clauses is actually already
      // satisfied, in which case we just continue (and keep the watch).

      unsigned blocking = other_pointer (watch);
      assert (lit != blocking);
      assert (not_lit != blocking);
      signed char blocking_value = values[blocking];
      if (blocking_value > 0)
        continue;

      if (is_binary_pointer (watch)) {
        assert (lit_pointer (watch) == not_lit);
        if (blocking_value < 0) {
          conflict = watch;
          if (stop_at_conflict)
            break;
        } else {
          // Only learned and thus redundant clauses are kept as
          // virtual binary clauses, where virtual means that
          // they only exist in the watch list of this ring.  They
          // are thus really copied (if shared among rings).

          assert (redundant_pointer (watch));

          // The 'assign' function expects the literals in the
          // opposite order as 'watch' and we have in essence to
          // swap upper and lower half of the watch pointer word.

          struct watch *reason = tag_binary (true, blocking, not_lit);

          assert (reason != watch);
          assign_with_reason (ring, blocking, reason);
          ticks++;
        }
      } else {
        // We now have to access the actual watcher data ...

        unsigned idx = index_pointer (watch);
        struct watcher *watcher = index_to_watcher (ring, idx);

        ticks++; // ... and pay the prize.

        // Satisfied (and vivified) but not removed clauses (actually
        // watchers to the clause) might still be watched and should
        // be ignored during propagation.

        if (watcher->garbage) // This induces the 'tick' above.
          continue;

        // Ignore the vivified clause during vivification.

        struct clause *clause = watcher->clause;
        if (ignore && clause == ignore)
          continue;

        unsigned watcher_glue = watcher->glue;
        unsigned clause_glue = clause->glue;
        assert (clause_glue <= watcher_glue);
        if (clause_glue < watcher_glue) {
          watcher->glue = clause_glue;
          LOGWATCH (watch, "updated from glue %u to", watcher_glue);
        }

        // The watchers need to precisely know the two watched
        // literals, which might be different from the blocking
        // literal.  Otherwise unit propagation is not efficient
        // (other invariants might also break).

        // However as watchers are only accessed while traversing a
        // watch list we always know during such a traversal already
        // one of the two literals.  Therefore we can simply use the
        // XOR trick and only store the bit-wise difference (the
        // 'XOR') between the two watched literals in the watcher
        // instead of both literals and get the other watched literal
        // during traversal by adding (with 'XOR') to that difference.

        unsigned other = watcher->sum ^ not_lit;

        signed char other_value;
        if (other == blocking)
          other_value = blocking_value;
        else {
          other_value = values[other];
          if (other_value > 0) {
            bool redundant = redundant_pointer (watch);
            watch = tag_index (redundant, idx, other);
            q[-1] = watch;
            continue;
          }
        }

        // Now the irredundant and virtual redundant binary clauses
        // are handled and neither the blocking literal nor the other
        // watched literal (if different) are assigned to true, and
        // it is time to either find a non-false replacement watched
        // literal, or determine that the clause is unit or
        // conflicting (all replacement candidates are false).

        unsigned replacement = INVALID;
        signed char replacement_value = -1;

        // The watchers can store literals of short clauses (currently
        // three or four literals long) directly in the watcher data
        // structure in order to avoid a second pointer dereference
        // (not needed for sequential solvers) to the actual clause
        // data (the latter being shared among threads).  While
        // initializing the watcher the size field is set to the
        // actual size of the clause if it is short enough and to zero
        // if it is too long (has more than four literals).

        unsigned watcher_size = watcher->size;
        if (watcher_size) {
          unsigned *literals = watcher->aux;
          unsigned *end_literals = literals + watcher_size;
          for (unsigned *r = literals; r != end_literals; r++) {
            replacement = *r;
            if (replacement != not_lit && replacement != other) {
              replacement_value = values[replacement];
              if (replacement_value >= 0)
                break;
            }
          }
        } else {
          // Now we pay the prize of accessing the actual clause too
          // (one of the following 'clause->size' accesses).

          // During propagation the 'tick' above for accessing
          // watchers and this one form the hot-spots of the solver,
          // due to irregular memory access (cache read misses).
          // All this special treatment of binary clauses, the
          // blocking literal and keeping short clause literals
          // directly in the watcher data-structure are all only
          // used to reduce the time spent in these two hot-spots.

          // The following code matches the same standard
          // propagation code in for instance CaDiCaL and Kissat.

          ticks++;
#ifdef METRICS
          assert (clause->size > 2);
          if (clause->size >= SIZE_VISITS)
            visits[0]++;
          else
            visits[clause->size]++;
#endif
          unsigned *literals = clause->literals;
          unsigned *end_literals = literals + clause->size;
          assert (watcher->aux[0] <= clause->size);
          unsigned *middle_literals = literals + watcher->aux[0];
          unsigned *r = middle_literals;
          while (r != end_literals) {
            replacement = *r;
            if (replacement != not_lit && replacement != other) {
              replacement_value = values[replacement];
              if (replacement_value >= 0)
                break;
            }
            r++;
          }
          if (replacement_value < 0) {
            r = literals;
            while (r != middle_literals) {
              replacement = *r;
              if (replacement != not_lit && replacement != other) {
                replacement_value = values[replacement];
                if (replacement_value >= 0)
                  break;
              }
              r++;
            }
          }
          watcher->aux[0] = r - literals;
        }

        if (replacement_value >= 0) {
          watcher->sum = other ^ replacement;
          LOGCLAUSE (clause, "unwatching %s in", LOGLIT (not_lit));
          watch_literal (ring, replacement, other, watcher);
          ticks++;
          q--;
        } else if (other_value) {
          assert (other_value < 0);
          conflict = watch;
          if (stop_at_conflict)
            break;
        } else {
          assign_with_reason (ring, other, watch);
          ticks++;
        }
      }
    }
    while (p != end)
      *q++ = *p++;
    ticks += cache_lines (p, begin);
    watches->end = q;
    if (q == watches->begin)
      RELEASE (*watches);
  }

  struct ring_statistics *statistics = &ring->statistics;
  struct context *context = statistics->contexts + ring->context;

  if (conflict) {
    LOGWATCH (conflict, "conflicting");
    context->conflicts++;
    ring->import_after_propagation_and_conflict = true;

    if (ring->context == SEARCH_CONTEXT && ring->randec) {
      if (!--ring->randec)
        very_verbose (ring, "last random decision conflict");
      else if (ring->randec == 1)
        very_verbose (ring, "one more random decision conflict to go");
      else
        very_verbose (ring, "%u more random decision conflicts to go",
                      ring->randec);
    }
  }

  context->propagations += propagations;
  context->ticks += ticks;

  return conflict;
}

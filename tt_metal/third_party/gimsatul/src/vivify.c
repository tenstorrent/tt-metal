#include "vivify.h"
#include "allocate.h"
#include "analyze.h"
#include "assign.h"
#include "backtrack.h"
#include "export.h"
#include "import.h"
#include "message.h"
#include "promote.h"
#include "propagate.h"
#include "reduce.h"
#include "report.h"
#include "ring.h"
#include "search.h"
#include "sort.h"
#include "utilities.h"

#include "cover.h"

#include <inttypes.h>
#include <string.h>

struct vivifier {
  struct ring *ring;
  struct unsigneds candidates;
  struct unsigneds decisions;
  struct unsigneds sorted;
  unsigned *counts;
};

static void init_vivifier (struct vivifier *vivifier, struct ring *ring) {
  vivifier->ring = ring;
  INIT (vivifier->candidates);
  INIT (vivifier->decisions);
  INIT (vivifier->sorted);
  vivifier->counts =
      allocate_and_clear_array (2 * ring->size, sizeof *vivifier->counts);
}

static void release_vivifier (struct vivifier *vivifier) {
  RELEASE (vivifier->candidates);
  RELEASE (vivifier->decisions);
  RELEASE (vivifier->sorted);
  free (vivifier->counts);
}

static inline bool watched_vivification_candidate (struct ring *ring,
                                                   struct watcher *watcher,
                                                   unsigned tier) {
  assert (tier == 1 || tier == 2);
  if (watcher->garbage)
    return false;
  if (!watcher->redundant)
    return false;
  unsigned watcher_glue = watcher_glue;
  unsigned tier1 = ring->tier1_glue_limit[0]; // NO TYPO!
  unsigned tier2 = ring->tier2_glue_limit[0]; // NO TYPO!
  if (tier == 1) {
    if (watcher_glue > tier1) {
      struct clause *clause = watcher->clause;
      unsigned clause_glue = clause->glue;
      if (clause_glue < watcher_glue) {
        watcher->glue = clause_glue;
        if (clause_glue > tier1)
          return false;
      }
      return false;
    }
  } else if (tier == 2) {
    if (watcher->glue <= tier1)
      return false;
    if (watcher->glue > tier2) {
      struct clause *clause = watcher->clause;
      unsigned clause_glue = clause->glue;
      if (clause_glue < watcher_glue) {
        watcher->glue = clause_glue;
        if (clause_glue > tier2)
          return false;
      }
      return false;
    }
  }

  // The following does not hold anymore with 'promote_watcher' as
  // the clause glue might have been decreased by another ring.
  // Therefore we should eventually
#if 0

   
  // As long we increase the glue of imported clauses in 'watches.c' to
  // 'MAX_GLUE', which is larger than 'TIER2_GLUE_LIMIT', the following
  // condition can not be true. Therefore the side-effect of that option
  // enabled is to never vivify imported clauses.  If we increase the
  // glue of imported clauses just by one we can get in this situation.
  //
  assert (!ring->options.increase_imported_glue ||
          watcher->glue == watcher->clause->glue + 1 ||
          watcher->clause->origin == ring->id);
#endif

  if (watcher->clause->vivified) {
    LOGCLAUSE (watcher->clause, "already vivified");
    mark_garbage_watcher (ring, watcher);
    return false;
  }
  return true;
}

static void schedule_vivification_candidate (struct ring *ring,
                                             unsigned *counts,
                                             struct unsigneds *candidates,
                                             struct watcher *candidate) {
  assert (!ring->level);
  signed char *values = ring->values;
  for (all_watcher_literals (lit, candidate))
    if (values[lit] > 0) {
      LOGCLAUSE (candidate->clause, "root-level satisfied");
      mark_garbage_watcher (ring, candidate);
      return;
    }
  for (all_watcher_literals (lit, candidate))
    if (!values[lit])
      counts[lit]++;
  PUSH (*candidates, watcher_to_index (ring, candidate));
}

static bool better_vivification_candidate (unsigned *counts,
                                           struct watcher *c,
                                           struct watcher *d) {
  unsigned csize = c->size ? c->size : SIZE_WATCHER_LITERALS;
  unsigned dsize = d->size ? d->size : SIZE_WATCHER_LITERALS;
  struct clause *cclause = c->size ? 0 : c->clause;
  struct clause *dclause = d->size ? 0 : d->clause;
  unsigned *clits = c->aux, *dlits = d->aux;
  unsigned *end_clits = clits + csize;
  unsigned *end_dlits = dlits + dsize;
  unsigned *p = clits, *q = dlits;

  while (p != end_clits && q != end_dlits) {
    unsigned a = *p++;
    unsigned b = *q++;
    unsigned u = counts[a];
    unsigned v = counts[b];
    if (u < v)
      return false;
    if (u > v)
      return true;
    if (a < b)
      return true;
    if (a > b)
      return false;
  }

  if (p == end_clits && q != end_dlits)
    return true;

  if (p != end_clits && q == end_dlits)
    return false;

  if (cclause)
    csize = cclause->size;
  if (dclause)
    dsize = dclause->size;

  if (csize < dsize)
    return true;
  if (csize > dsize)
    return false;

  return c > d;
}

#define LESS_VIVIFICATION_CANDIDATE(A, B) \
  better_vivification_candidate (counts, index_to_watcher (ring, (A)), \
                                 index_to_watcher (ring, (B)))

static bool better_vivification_literal (unsigned *counts, unsigned a,
                                         unsigned b) {
  unsigned u = counts[a];
  unsigned v = counts[b];
  if (u < v)
    return false;
  if (u > v)
    return true;
  return a < b;
}

static void sort_vivivification_candidates (struct ring *ring,
                                            unsigned *counts, size_t size,
                                            unsigned *candidates) {
  unsigned *end = candidates + size;
  for (unsigned *c = candidates; c != end; c++) {
    unsigned idx = *c;
    struct watcher *watcher = index_to_watcher (ring, idx);
    if (watcher->size) {
      unsigned *lits = watcher->aux;
      for (unsigned i = 0; i + 1 != watcher->size; i++)
        for (unsigned j = i + 1; j != watcher->size; j++)
          if (better_vivification_literal (counts, lits[j], lits[i]))
            SWAP (unsigned, lits[i], lits[j]);
    } else {
      struct clause *clause = watcher->clause;
      assert (clause->size > SIZE_WATCHER_LITERALS);
      unsigned *src = clause->literals;
      unsigned *end_src = src + clause->size;
      unsigned *dst = watcher->aux;
      unsigned *end_dst = dst + SIZE_WATCHER_LITERALS;
      unsigned last = INVALID;
      for (unsigned *q = dst; q != end_dst; q++) {
        unsigned next = INVALID;
        for (unsigned *p = src; p != end_src; p++) {
          unsigned other = *p;
          if (last == INVALID ||
              better_vivification_literal (counts, last, other))
            if (next == INVALID ||
                better_vivification_literal (counts, other, next))
              next = other;
        }
        assert (next != INVALID);
        last = *q = next;
      }
    }
  }

  SORT (unsigned, size, candidates, LESS_VIVIFICATION_CANDIDATE);

  for (unsigned *c = candidates; c != end; c++) {
    unsigned idx = *c;
    struct watcher *watcher = index_to_watcher (ring, idx);
    if (watcher->size) {
#ifdef LOGGING
      do {
        unsigned size = watcher->size;
        LOGPREFIX ("sorted glue %u size %u watcher[%u] "
                   "vivification candidate",
                   watcher->glue, size, idx);
        unsigned *lits = watcher->aux;
        unsigned *end_lits = lits + size;
        for (unsigned *p = lits; p != end_lits; p++) {
          unsigned lit = *p;
          printf (" %s#%u", LOGLIT (lit), counts[lit]);
        }
        LOGSUFFIX ();
      } while (0);
#endif
    } else {
#ifdef LOGGING
      do {
        struct clause *clause = watcher->clause;
        LOGPREFIX ("sorted glue %u size %u watcher[%u] "
                   "vivification candidate",
                   watcher->glue, clause->size, idx);
        unsigned *lits = watcher->aux;
        unsigned *end_lits = lits + SIZE_WATCHER_LITERALS;
        for (unsigned *p = lits; p != end_lits; p++) {
          unsigned lit = *p;
          printf (" %s#%u", LOGLIT (lit), counts[lit]);
        }
        printf (" ... clause[%" PRIu64 "]", clause->id);
        LOGSUFFIX ();
      } while (0);
#endif
      watcher->aux[0] = 0;
    }
  }
}

static size_t reschedule_vivification_candidates (struct vivifier *vivifier,
                                                  unsigned tier) {
  struct unsigneds *candidates = &vivifier->candidates;
  unsigned *counts = vivifier->counts;
  struct ring *ring = vivifier->ring;
  assert (EMPTY (*candidates));
  for (all_redundant_watchers (watcher))
    if (watcher->vivify &&
        watched_vivification_candidate (ring, watcher, tier))
      schedule_vivification_candidate (ring, counts, candidates, watcher);
  size_t size = SIZE (*candidates);
  sort_vivivification_candidates (ring, counts, size, candidates->begin);
  return size;
}

static size_t schedule_vivification_candidates (struct vivifier *vivifier,
                                                unsigned tier) {
  struct unsigneds *candidates = &vivifier->candidates;
  unsigned *counts = vivifier->counts;
  struct ring *ring = vivifier->ring;
  memset (counts, 0, sizeof (unsigned) * 2 * ring->size);
  size_t before = SIZE (*candidates);
  for (all_redundant_watchers (watcher))
    if (!watcher->vivify &&
        watched_vivification_candidate (ring, watcher, tier))
      schedule_vivification_candidate (ring, counts, candidates, watcher);
  size_t after = SIZE (*candidates);
  size_t delta = after - before;
  sort_vivivification_candidates (ring, counts, delta,
                                  candidates->begin + before);
  return after;
}

#define ANALYZE(OTHER) \
  do { \
    assert (ring->values[OTHER]); \
    unsigned idx = IDX (OTHER); \
    struct variable *v = variables + idx; \
    unsigned level = v->level; \
    if (!level) \
      break; \
    if (subsuming && marked_literal (ring_marks, OTHER) <= 0) \
      subsuming = false; \
    if (v->seen) \
      break; \
    v->seen = true; \
    PUSH (*analyzed, idx); \
    if (level != ring->level && !used[level]) { \
      used[level] = 1; \
      PUSH (*levels, level); \
    } \
    if (!v->reason) \
      PUSH (*ring_clause, OTHER); \
  } while (0)

static struct watch *vivify_deduce (struct vivifier *vivifier,
                                    struct watch *candidate,
                                    struct watch *conflict,
                                    unsigned implied) {
  struct ring *ring = vivifier->ring;
  struct unsigneds *analyzed = &ring->analyzed;
  struct variable *variables = ring->variables;

  struct unsigneds *ring_clause = &ring->clause;
  struct unsigneds *levels = &ring->levels;

  unsigned char *used = ring->used;

  assert (EMPTY (*analyzed));
  assert (EMPTY (*ring_clause));

  if (implied != INVALID) {
    unsigned idx = IDX (implied);
    struct variable *v = variables + idx;
    unsigned level = v->level;
    assert (!v->seen);
    assert (level);
    v->seen = true;
    PUSH (*analyzed, idx);
    if (v->level != ring->level) {
      used[level] = 1;
      PUSH (*levels, level);
    }
    PUSH (*ring_clause, implied);
    conflict = v->reason;
  }

  struct watch *reason = conflict ? conflict : candidate;

  struct ring_trail *trail = &ring->trail;
  unsigned *begin = trail->begin;
  unsigned *t = trail->end;

  signed char *ring_marks = ring->marks;

  while (t != begin) {
    assert (reason);
    LOGWATCH (reason, "vivify analyzing");
    bool subsuming = (reason != candidate);
    if (is_binary_pointer (reason)) {
      unsigned lit = lit_pointer (reason);
      ANALYZE (lit);
      unsigned other = other_pointer (reason);
      ANALYZE (other);
    } else {
      struct watcher *watcher = get_watcher (ring, reason);
      for (all_watcher_literals (other, watcher))
        ANALYZE (other);
    }
    if (subsuming) {
      assert (candidate != reason);
      LOGWATCH (reason, "vivify subsuming");
      return reason;
    }
    while (t != begin) {
      unsigned lit = *--t;
      unsigned idx = IDX (lit);
      struct variable *v = variables + idx;
      if (!v->level) {
        t = begin;
        break;
      }
      if (!v->seen)
        continue;
      reason = v->reason;
      if (reason == conflict)
        continue;
      if (reason)
        break;
    }
  }
  LOGTMP ("vivification deduced");

  return 0;
}

static bool vivify_shrink (struct ring *ring, struct watch *conflict,
                           struct watcher *candidate,
                           unsigned *implied_ptr) {
  assert (!is_binary_pointer (candidate));
  struct variable *variables = ring->variables;
  signed char *values = ring->values;
  for (all_watcher_literals (lit, candidate)) {
    unsigned idx = IDX (lit);
    signed char value = values[lit];
    if (!value) {
      LOG ("vivification removes at least unassigned %s", LOGLIT (lit));
      return true;
    } else if (value > 0) {
      if (conflict)
        return true;
      if (*implied_ptr == INVALID)
        *implied_ptr = lit;
    } else {
      assert (value < 0);
      struct variable *v = variables + idx;
      if (!v->level)
        continue;
      if (!v->seen) {
        LOG ("vivification removes at least unseen %s", LOGLIT (lit));
        return true;
      }
      if (v->reason) {
        LOG ("vivification removes at least falsified %s", LOGLIT (lit));
        return true;
      }
    }
  }
  return false;
}

static void vivify_learn (struct vivifier *vivifier,
                          struct watch *candidate) {
  struct ring *ring = vivifier->ring;
  struct unsigneds *decisions = &vivifier->decisions;
  LOGTMP ("vivify learning");
  struct unsigneds *ring_clause = &ring->clause;
  unsigned size = SIZE (*ring_clause);
  struct unsigneds *levels = &ring->levels;
  assert (size);
  assert (size < get_clause (ring, candidate)->size);
  unsigned *literals = ring_clause->begin;
  struct watch *res = 0;
  if (ring->level) {
    backtrack (ring, 0);
    CLEAR (*decisions);
  }
  if (size == 1) {
    unsigned unit = literals[0];
    trace_add_unit (&ring->trace, unit);
    ring->statistics.vivify.units++;
    assign_ring_unit (ring, unit);
    if (ring_propagate (ring, false, 0))
      set_inconsistent (ring,
                        "propagation of strengthened clause unit fails");
    else {
      ring->iterating = -1;
      iterate (ring);
    }
  } else if (size == 2) {
    unsigned lit = literals[0], other = literals[1];
    res = new_local_binary_clause (ring, true, lit, other);
    trace_add_binary (&ring->trace, lit, other);
    if (ring->options.vivify_export)
      export_binary_clause (ring, res);
  } else {
    struct watcher *watcher = get_watcher (ring, candidate);
    unsigned glue = SIZE (*levels);
    LOG ("computed glue %u", glue);
    if (glue > watcher->glue) {
      glue = watcher->glue;
      LOG ("but candidate glue %u smaller", glue);
    }
    if (glue == size)
      glue = size - 1;
    struct clause *clause = new_large_clause (size, literals, true, glue);
    LOGCLAUSE (clause, "vivify strengthened");
    clause->origin = ring->id;
    res = watch_first_two_literals_in_large_clause (ring, clause);
    trace_add_clause (&ring->trace, clause);
    if (ring->options.vivify_export)
      export_large_clause (ring, clause);
  }
}

static void sort_vivification_probes (signed char *values, unsigned *counts,
                                      struct unsigneds *sorted) {
  if (SIZE (*sorted) < 2)
    return;
  unsigned *begin = sorted->begin;
  unsigned *end = sorted->end;
  for (unsigned *p = begin; p + 1 != end; p++)
    for (unsigned *q = p + 1; q != end; q++)
      if (better_vivification_literal (counts, *q, *p))
        SWAP (unsigned, *p, *q);
}

static void vivify_watcher (struct vivifier *vivifier, unsigned tier,
                            unsigned idx) {
  struct ring *ring = vivifier->ring;
  struct unsigneds *decisions = &vivifier->decisions;
  assert (SIZE (*decisions) == ring->level);

  struct watcher *watcher = index_to_watcher (ring, idx);
  if (watcher->clause->vivified) {
    LOGCLAUSE (watcher->clause, "already vivified");
    mark_garbage_watcher (ring, watcher);
    return;
  }
  watcher->vivify = false;

  signed char *values = ring->values;
  struct clause *clause = watcher->clause;

  {
    unsigned unit = INVALID_LIT;
    for (all_literals_in_clause (lit, clause)) {
      signed char value = values[lit];
      if (value <= 0)
        continue;
      struct variable *v = VAR (lit);
      if (!v->level) {
        LOGCLAUSE (clause, "root-level satisfied");
        mark_garbage_watcher (ring, watcher);
        return;
      }
      struct watch *reason = v->reason;
      if (reason && !is_binary_pointer (reason) &&
          get_watcher (ring, reason) == watcher)
        unit = lit;
    }
    if (unit != INVALID_LIT) {
      LOG ("vivification candidate is the reason for %s", LOGLIT (unit));
      assert (values[unit] > 0);
      unsigned unit_level = VAR (unit)->level;
      assert (unit_level > 0);
      unsigned level = unit_level - 1;
      assert (level < ring->level);
      backtrack (ring, level);
      RESIZE (*decisions, level);
    }
  }

  LOGCLAUSE (clause, "trying to vivify watcher[%u]", idx);
  ring->statistics.vivify.tried++;

  for (unsigned level = 0; level != SIZE (*decisions); level++) {
    unsigned decision = decisions->begin[level];
    assert (VAR (decision)->level == level + 1);
    assert (!VAR (decision)->reason);
    bool found = false;
    for (all_literals_in_clause (lit, clause))
      if (NOT (lit) == decision) {
        found = true;
        break;
      }
    if (found) {
      assert (VAR (decision)->level == level + 1);
      assert (!VAR (decision)->reason);
      signed char value = values[decision];
      assert (value);
      if (value > 0) {
        LOG ("reusing decision %s", LOGLIT (decision));
        continue;
      }
      LOG ("decision %s with opposite phase", LOGLIT (decision));
    } else
      LOG ("decision %s not in clause", LOGLIT (decision));
    assert (level < ring->level);
    backtrack (ring, level);
    RESIZE (*decisions, level);
    break;
  }

  if (!EMPTY (*decisions))
    ring->statistics.vivify.reused++;

  struct unsigneds *sorted = &vivifier->sorted;
  CLEAR (*sorted);
  for (all_literals_in_clause (lit, clause)) {
    signed char value = values[lit];
    struct variable *v = VAR (lit);
    if (value < 0 && !v->level)
      continue;
    assert (!value || v->level);
    if (value && !v->reason) {
      LOG ("skipping reused decision %s", LOGLIT (lit));
      assert (!v->reason);
      assert (value < 0);
      assert (v->level);
      continue;
    }
    PUSH (*sorted, lit);
  }

  sort_vivification_probes (values, vivifier->counts, sorted);
#ifdef LOGGING
  do {
    LOGPREFIX ("sorted vivification size %zu literals", SIZE (*sorted));
    unsigned *counts = vivifier->counts;
    for (all_elements_on_stack (unsigned, lit, *sorted))
      printf (" %s#%u", LOGLIT (lit), counts[lit]);
    LOGSUFFIX ();
  } while (0);
#endif

  unsigned implied = INVALID;
  struct watch *conflict = 0;

  for (all_elements_on_stack (unsigned, lit, *sorted)) {
    signed char value = values[lit];

    if (value < 0)
      continue;

    if (value > 0) {
      LOG ("vivify implied literal %s", LOGLIT (lit));
      implied = lit;
      break;
    }

    assert (!value);

    ring->level++;
    ring->statistics.contexts[PROBING_CONTEXT].decisions++;
    unsigned not_lit = NOT (lit);
#ifdef LOGGING
    if (ring->stable)
      LOG ("assuming %s score %g", LOGLIT (not_lit),
           ring->heap.nodes[IDX (not_lit)].score);
    else
      LOG ("assuming %s stamp %" PRIu64, LOGLIT (not_lit),
           ring->queue.links[IDX (not_lit)].stamp);
#endif
    assign_decision (ring, not_lit);
    PUSH (*decisions, not_lit);

    conflict = ring_propagate (ring, false, clause);
    if (conflict)
      break;
  }

  assert (!conflict || implied == INVALID);

  for (all_literals_in_clause (lit, clause))
    mark_literal (ring->marks, lit);

  struct watch *candidate = tag_index (true, idx, INVALID_LIT);
  struct watch *subsuming =
      vivify_deduce (vivifier, candidate, conflict, implied);

  for (all_literals_in_clause (lit, clause))
    unmark_literal (ring->marks, lit);

  bool import_before_next_vivification = false;

  if (subsuming) {
    ring->statistics.vivify.succeeded++;
    ring->statistics.vivify.subsumed++;
    LOGWATCH (candidate, "vivify subsumed");
    assert (candidate != subsuming);
    if (!is_binary_pointer (subsuming)) {
      struct watcher *subsuming_watcher = get_watcher (ring, subsuming);
      if (subsuming_watcher->redundant) {
        assert (clause != subsuming_watcher->clause);
        assert (clause->redundant);
        assert (watcher->redundant);
        unsigned watcher_glue = watcher->glue;
        unsigned subsuming_glue = subsuming_watcher->glue;
        if (watcher_glue < subsuming_glue)
          promote_watcher (ring, subsuming_watcher, watcher_glue);
      }
    }
    mark_garbage_watcher (ring, watcher);
  } else if (vivify_shrink (ring, conflict, watcher, &implied)) {
    ring->statistics.vivify.succeeded++;
    ring->statistics.vivify.strengthened++;
    LOGWATCH (candidate, "vivify strengthening");
    (void) vivify_learn (vivifier, candidate);
    watcher = index_to_watcher (ring, idx);
    mark_garbage_watcher (ring, watcher);

    // This is the single unprotected write access to clause data
    // in parallel and thus in principle is a data race.  On the
    // Other hand it just marks this clause to be ignored, actually
    // targeted to be garbage collected in other threads (during
    // vivification). Therefore this condition is never reset and a race
    // might only delay deletion.  Furthermore the 'vivified' flag is a
    // single bit and there is the risk that multiple threads writing
    // concurrently to the word with this bit (after reading the word
    // or byte with the bit first).  However, as 'vivified' is the only data
    // written in parallel all those competing threads will only potentially
    // differ in that bit, and all want to set it true, which is sane.

    // However, if long imported clause glues are increased to MAX_GLUE in
    // the watcher (when watching them) other threads would never try to
    // vivify this clause.  If it is just increased by one it happens.

    watcher->clause->vivified = true;

    // In any case trigger import of new clauses as strengthening and
    // exporting a clause does not happen too frequently and can be
    // considered to play the same role as clause learning during analyzing
    // a regular conflict which also triggers new imports.

    import_before_next_vivification = true;

  } else if (implied != INVALID) {
    ring->statistics.vivify.succeeded++;
    ring->statistics.vivify.implied++;
    LOGCLAUSE (watcher->clause, "vivify implied");
    mark_garbage_watcher (ring, watcher);
  } else
    LOGCLAUSE (clause, "vivification failed on");

  ring->import_after_propagation_and_conflict =
      import_before_next_vivification;

  clear_analyzed (ring);
  CLEAR (ring->clause);
}

void vivify_clauses (struct ring *ring) {
  if (ring->inconsistent)
    return;
  if (!ring->options.vivify)
    return;
  if (!backtrack_propagate_iterate (ring))
    return;
  START (ring, vivify);
  assert (SEARCH_TICKS >= ring->last.probing);

  uint64_t delta_search_ticks = SEARCH_TICKS - ring->last.probing;
  delta_search_ticks = MAX (MIN_ABSOLUTE_FFORT, delta_search_ticks);
  uint64_t delta_probing_ticks = VIVIFY_EFFORT * delta_search_ticks;
  verbose (ring,
           "total vivification effort of %" PRIu64 " = %g * %" PRIu64
           " search ticks",
           delta_probing_ticks, (double) VIVIFY_EFFORT, delta_search_ticks);

  double sum = RELATIVE_VIVIFY_TIER1_EFFORT + RELATIVE_VIVIFY_TIER2_EFFORT;
  uint64_t limit = PROBING_TICKS;

  for (unsigned tier = 1; tier <= 2; tier++) {
    if (ring->inconsistent)
      break;
    if (terminate_ring (ring))
      break;

    double effort;
    if (tier == 2)
      effort = RELATIVE_VIVIFY_TIER2_EFFORT;
    else
      effort = RELATIVE_VIVIFY_TIER1_EFFORT;

    double scale = effort / sum;
    uint64_t scaled_ticks = scale * delta_probing_ticks;
    limit += scaled_ticks;
#ifndef QUIET
    uint64_t probing_ticks_before = PROBING_TICKS;
#endif
    verbose (ring,
             "tier%u vivification limit of %" PRIu64
             " vivification ticks %.0f%%",
             tier, scaled_ticks, 100.0 * scale);

    struct vivifier vivifier;
    init_vivifier (&vivifier, ring);

    size_t rescheduled =
        reschedule_vivification_candidates (&vivifier, tier);
    very_verbose (ring, "rescheduled %zu tier%u vivification candidates",
                  rescheduled, tier);
    size_t scheduled = schedule_vivification_candidates (&vivifier, tier);
    very_verbose (ring,
                  "scheduled %zu tier%u vivification candidates in total",
                  scheduled, tier);
#ifdef QUIET
    (void) rescheduled, (void) scheduled;
#endif

    uint64_t vivified = ring->statistics.vivify.succeeded;
    uint64_t tried = ring->statistics.vivify.tried;

    struct unsigneds *decisions = &vivifier.decisions;

    size_t i = 0;
    while (i != SIZE (vivifier.candidates)) {
      if (PROBING_TICKS > limit)
        break;
      if (terminate_ring (ring))
        break;
      if (import_shared (ring)) {
        if (ring->inconsistent)
          break;
        if (ring->level) {
          backtrack (ring, 0);
          ring->trail.propagate = ring->trail.begin;
        }
        RESIZE (*decisions, ring->level);
        assert (ring->level == SIZE (*decisions));
        if (ring_propagate (ring, false, 0)) {
          set_inconsistent (ring, "propagation of imported clauses "
                                  "during vivification fails");
          break;
        }
      }
      unsigned idx = vivifier.candidates.begin[i++];
      vivify_watcher (&vivifier, tier, idx);
    }

    if (!ring->inconsistent && ring->level) {
      backtrack (ring, 0);
      CLEAR (*decisions);
    }

    size_t final_scheduled = SIZE (vivifier.candidates);
    size_t remain = final_scheduled - i;
    if (remain)
      very_verbose (ring,
                    "incomplete vivification as %zu tier%u "
                    "candidates remain %.0f%%",
                    remain, tier, percent (remain, final_scheduled));
    else
      very_verbose (ring,
                    "all %zu scheduled tier%u "
                    "vivification candidates tried",
                    final_scheduled, tier);

    while (i != final_scheduled) {
      unsigned idx = vivifier.candidates.begin[i++];
      struct watcher *watcher = index_to_watcher (ring, idx);
      watcher->vivify = true;
    }

    release_vivifier (&vivifier);

    vivified = ring->statistics.vivify.succeeded - vivified;
    tried = ring->statistics.vivify.tried - tried;

    very_verbose (ring,
                  "vivified %" PRIu64 " tier%u clauses %.0f%% from %" PRIu64
                  " tried %.0f%% "
                  "after %" PRIu64 " ticks (%s)",
                  vivified, tier, percent (vivified, tried), tried,
                  percent (tried, scheduled),
                  PROBING_TICKS - probing_ticks_before,
                  (PROBING_TICKS > limit ? "limit hit" : "completed"));

    verbose_report (ring, (tier == 1 ? 'u' : 'v'), !vivified);
  }
  STOP (ring, vivify);
}

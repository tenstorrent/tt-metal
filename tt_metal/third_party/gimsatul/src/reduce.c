#include "reduce.h"
#include "barrier.h"
#include "macros.h"
#include "message.h"
#include "report.h"
#include "ring.h"
#include "tiers.h"
#include "trace.h"
#include "utilities.h"

#include "cover.h" // TODO remove

#include <inttypes.h>
#include <math.h>

bool reducing (struct ring *ring) {
  return ring->limits.reduce < SEARCH_CONFLICTS;
}

void check_clause_statistics (struct ring *ring) {
#ifndef NDEBUG
  {
    size_t redundant = 0;
    size_t irredundant = 0;

    for (all_ring_literals (lit)) {
      struct references *watches = &REFERENCES (lit);
      for (all_watches (watch, *watches)) {
        if (!is_binary_pointer (watch))
          continue;
        assert (lit == lit_pointer (watch));
        unsigned other = other_pointer (watch);
        if (lit < other)
          continue;
        assert (redundant_pointer (watch));
        redundant++;
      }

      unsigned *binaries = watches->binaries;
      if (!binaries)
        continue;
      for (unsigned *p = binaries, other; (other = *p) != INVALID; p++)
        if (lit < other)
          irredundant++;
    }

    for (all_watchers (watcher)) {
      if (watcher->garbage)
        continue;
      assert (watcher->clause->redundant == watcher->redundant);
      if (watcher->redundant)
        redundant++;
      else
        irredundant++;
    }

    struct ring_statistics *statistics = &ring->statistics;
    assert (statistics->redundant == redundant);
    assert (statistics->irredundant == irredundant);
  }
#else
  (void) ring;
#endif
}

void check_redundant_offset (struct ring *ring) {
#ifndef NDBEUG
  struct watchers *watchers = &ring->watchers;
  struct watcher *begin = watchers->begin;
  struct watcher *redundant = begin + ring->redundant;
  struct watcher *end = watchers->end;

  for (struct watcher *watcher = begin; watcher != redundant; watcher++)
    assert (!watcher->redundant);

  assert (begin <= redundant);
  assert (redundant <= end);

  for (struct watcher *watcher = redundant; watcher != end; watcher++)
    assert (watcher->redundant);
#else
  (void) ring;
#endif
}

#define all_literals_on_trail_with_reason(LIT) \
  unsigned *P_##LIT = ring->trail.begin, *END_##LIT = ring->trail.end, \
           LIT; \
  P_##LIT != END_##LIT && (LIT = *P_##LIT, true); \
  ++P_##LIT

static void mark_reasons (struct ring *ring, unsigned start) {
  for (all_literals_on_trail_with_reason (lit)) {
    struct variable *v = VAR (lit);
    struct watch *watch = v->reason;
    if (!watch)
      continue;
    if (is_binary_pointer (watch))
      continue;
    unsigned src = index_pointer (watch);
    if (src < start)
      continue;
    struct watcher *watcher = index_to_watcher (ring, src);
    assert (!watcher->reason);
    watcher->reason = true;
  }
}

static unsigned map_idx (unsigned src, unsigned start, unsigned *map) {
  return (src < start) ? src : map[src - start];
}

static void unmark_reasons (struct ring *ring, unsigned start,
                            unsigned *map) {
  for (all_literals_on_trail_with_reason (lit)) {
    struct variable *v = VAR (lit);
    struct watch *watch = v->reason;
    if (!watch)
      continue;
    if (is_binary_pointer (watch))
      continue;
    unsigned src = index_pointer (watch);
    if (src < start)
      continue;
    unsigned dst = map_idx (src, start, map);
    assert (dst);
    struct watcher *watcher = index_to_watcher (ring, dst);
    assert (watcher->reason);
    watcher->reason = false;
    bool redundant = redundant_pointer (watch);
    unsigned other = other_pointer (watch);
    struct watch *mapped = tag_index (redundant, dst, other);
    v->reason = mapped;
  }
}

static void gather_reduce_candidates (struct ring *ring,
                                      struct unsigneds *candidates) {
  struct watchers *watchers = &ring->watchers;
  struct watcher *begin = watchers->begin;
  struct watcher *end = watchers->end;
  struct watcher *redundant = begin + ring->redundant;
  unsigned tier1 = ring->tier1_glue_limit[ring->stable];
  unsigned tier2 = ring->tier2_glue_limit[ring->stable];
  for (struct watcher *watcher = redundant; watcher != end; watcher++) {
    if (!watcher->redundant)
      continue;
    if (watcher->garbage)
      continue;
    const unsigned char used = watcher->used;
    if (used)
      watcher->used = used - 1;
    if (watcher->reason)
      continue;
    const unsigned char glue = watcher->glue;
    if (glue <= tier1 && used)
      continue;
    if (glue <= tier2 && used >= MAX_USED - 1)
      continue;
    unsigned idx = watcher_to_index (ring, watcher);
    PUSH (*candidates, idx);
  }
  verbose (ring, "gathered %zu reduce candidates %.0f%%",
           SIZE (*candidates),
           percent (SIZE (*candidates), ring->statistics.redundant));
}

static void
mark_reduce_candidates_as_garbage (struct ring *ring,
                                   struct unsigneds *candidates) {
  size_t size = SIZE (*candidates);
  const double fraction =
      ring->stable ? REDUCE_FRACTION_STABLE : REDUCE_FRACTION_FOCUSED;
  size_t target = fraction * size;
  size_t reduced = 0;
  unsigned tier1 = ring->tier1_glue_limit[ring->stable];
  unsigned tier2 = ring->tier2_glue_limit[ring->stable];
  for (all_elements_on_stack (unsigned, idx, *candidates)) {
    struct watcher *watcher = index_to_watcher (ring, idx);
    mark_garbage_watcher (ring, watcher);
    ring->statistics.reduced.clauses++;
    if (watcher->glue <= tier1)
      ring->statistics.reduced.tier1++;
    else if (watcher->glue <= tier2)
      ring->statistics.reduced.tier2++;
    else
      ring->statistics.reduced.tier3++;
    if (++reduced == target)
      break;
  }
  verbose (ring, "reduced %zu clauses %.0f%%", reduced,
           percent (reduced, size));
}

static void flush_references (struct ring *ring, bool fixed, unsigned start,
                              unsigned *map) {
#if !defined(QUIET) || !defined(NDEBUG)
  size_t flushed = 0;
#endif
  signed char *values = ring->values;
  struct variable *variables = ring->variables;
  for (all_ring_literals (lit)) {
    signed char lit_value = values[lit];
    if (lit_value > 0) {
      if (variables[IDX (lit)].level)
        lit_value = 0;
    }
    struct references *watches = &REFERENCES (lit);
    struct watch **begin = watches->begin, **q = begin;
    struct watch **end = watches->end;
    for (struct watch **p = begin; p != end; p++) {
      struct watch *watch = *p;
      if (is_binary_pointer (watch)) {
        assert (lit == lit_pointer (watch));
        if (!fixed) {
          *q++ = watch;
          continue;
        }
        unsigned other = other_pointer (watch);
        assert (lit != other);
        signed char other_value = values[other];
        if (other_value > 0) {
          if (variables[IDX (other)].level)
            other_value = 0;
        }
        if (lit_value > 0 || other_value > 0) {
          if (lit < other) {
            bool redundant = redundant_pointer (watch);
            dec_clauses (ring, redundant);
            trace_delete_binary (&ring->trace, lit, other);
          }
#if !defined(QUIET) || !defined(NDEBUG)
          flushed++;
#endif
        } else
          *q++ = watch;
      } else {
        unsigned src = index_pointer (watch);
        unsigned dst = map_idx (src, start, map);
        if (dst) {
          assert (dst);
          bool redundant = redundant_pointer (watch);
          unsigned other = other_pointer (watch);
          struct watch *mapped = tag_index (redundant, dst, other);
          *q++ = mapped;
        } else {
#if !defined(QUIET) || !defined(NDEBUG)
          flushed++;
#endif
        }
      }
    }
    watches->end = q;
    SHRINK_STACK (*watches);
  }
  assert (!(flushed & 1));
  verbose (ring, "flushed %zu garbage watches from watch lists", flushed);
}

void reduce (struct ring *ring) {
  START (ring, reduce);
  check_clause_statistics (ring);
  check_redundant_offset (ring);
  recalculate_tier_limits (ring);
  struct ring_statistics *statistics = &ring->statistics;
  struct ring_limits *limits = &ring->limits;
  statistics->reductions++;
  verbose (ring, "reduction %" PRIu64 " at %" PRIu64 " conflicts",
           statistics->reductions, SEARCH_CONFLICTS);
  bool fixed = ring->last.fixed != ring->statistics.fixed;
  unsigned start = 1;
  if (fixed)
    mark_satisfied_watchers_as_garbage (ring);
  else
    start = ring->redundant;
  mark_reasons (ring, start);
  struct unsigneds candidates;
  INIT (candidates);
  gather_reduce_candidates (ring, &candidates);
  sort_redundant_watcher_indices (ring, SIZE (candidates),
                                  candidates.begin);
  mark_reduce_candidates_as_garbage (ring, &candidates);
  RELEASE (candidates);
  unsigned *map = flush_watchers (ring, start);
  unmark_reasons (ring, start, map);
  flush_references (ring, fixed, start, map);
  free (map);
  reset_last_learned (ring);
  check_clause_statistics (ring);
  check_redundant_offset (ring);
  limits->reduce = SEARCH_CONFLICTS;
  unsigned interval = ring->options.reduce_interval;
  assert (interval);
  uint64_t delta = interval * sqrt (statistics->reductions);
  limits->reduce += delta;
  very_verbose (
      ring, "next reduce limit at %" PRIu64 " after %" PRIu64 " conflicts",
      limits->reduce, delta);
  report (ring, '-');
  STOP (ring, reduce);
}

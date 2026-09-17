#include "allocate.h"
#include "analyze.h"
#include "heap.h"
#include "inline.h"
#include "inlinevector.h"
#include "logging.h"
#include "print.h"
#include "proprobe.h"
#include "report.h"
#include "terminate.h"
#include "trail.h"
#include "transitive.h"
#include "sort.h"

#include <stddef.h>

static void transitive_assign(kissat *solver, unsigned lit) {
  LOG("transitive assign %s", LOGLIT(lit));
  value *values = solver->values;
  const unsigned not_lit = NOT(lit);
  assert(!values[lit]);
  assert(!values[not_lit]);
  values[lit] = 1;
  values[not_lit] = -1;
  PUSH_ARRAY(solver->trail, lit);
}

static void transitive_backtrack(kissat *solver, unsigned *saved) {
  value *values = solver->values;

  unsigned *end_trail = END_ARRAY(solver->trail);
  assert(saved <= end_trail);

  while (end_trail != saved) {
    const unsigned lit = *--end_trail;
    LOG("transitive unassign %s", LOGLIT(lit));
    const unsigned not_lit = NOT(lit);
    assert(values[lit] > 0);
    assert(values[not_lit] < 0);
    values[lit] = values[not_lit] = 0;
  }

  SET_END_OF_ARRAY(solver->trail, saved);
  solver->propagate = saved;
  solver->level = 0;
}

static void prioritize_binaries(kissat *solver) {
  assert(solver->watching);
  statches large;
  INIT_STACK(large);
  watches *all_watches = solver->watches;
  for (all_literals(lit)) {
    assert(EMPTY_STACK(large));
    watches *watches = all_watches + lit;
    watch *begin_watches = BEGIN_WATCHES(*watches), *q = begin_watches;
    const watch *const end_watches = END_WATCHES(*watches), *p = q;
    while (p != end_watches) {
      const watch head = *q++ = *p++;
      if (head.type.binary) {
        continue;
      }
      const watch tail = *p++;
      PUSH_STACK(large, head);
      PUSH_STACK(large, tail);
      q--;
    }
    const watch *const end_large = END_STACK(large);
    watch const *r = BEGIN_STACK(large);
    while (r != end_large) {
      *q++ = *r++;
    }
    assert(q == end_watches);
    CLEAR_STACK(large);
  }
  RELEASE_STACK(large);
}

static bool transitive_reduce(kissat *solver,
      unsigned src, uint64_t limit,
      uint64_t *reduced_ptr, unsigned *units) {
  bool res = false;
  assert(!VALUE(src));
  LOG("transitive reduce %s", LOGLIT(src));
  watches *all_watches = solver->watches;
  watches *src_watches = all_watches + src;
  watch *end_src = END_WATCHES(*src_watches);
  watch *begin_src = BEGIN_WATCHES(*src_watches);
  const size_t size_src_watches = SIZE_WATCHES(*src_watches);
  const unsigned src_ticks =
        1 + kissat_cache_lines(size_src_watches, sizeof(watch));
  ADD(transitive_ticks, src_ticks);
  ADD(probing_ticks, src_ticks);
  ADD(ticks, src_ticks);
  INC(transitive_probes);
  const unsigned not_src = NOT(src);
  unsigned reduced = 0;
  bool failed = false;
  for (watch *p = begin_src; p != end_src; p++) {
    const watch src_watch = *p;
    if (!src_watch.type.binary) {
      break;
    }
    const unsigned dst = src_watch.binary.lit;
    if (dst < src) {
      continue;
    }
    if (VALUE(dst)) {
      continue;
    }
    assert(kissat_propagated(solver));
    unsigned *saved = solver->propagate;
    assert(!solver->level);
    solver->level = 1;
    transitive_assign(solver, not_src);
    const bool redundant = src_watch.binary.redundant;
    bool transitive = false;
    unsigned inner_ticks = 0;
    unsigned *propagate = solver->propagate;
    while (!transitive && !failed && propagate != END_ARRAY(solver->trail)) {
      const unsigned lit = *propagate++;
      LOG("transitive propagate %s", LOGLIT(lit));
      assert(VALUE(lit) > 0);
      const unsigned not_lit = NOT(lit);
      watches *lit_watches = all_watches + not_lit;
      const watch *const end_lit = END_WATCHES(*lit_watches);
      const watch *const begin_lit = BEGIN_WATCHES(*lit_watches);
      const size_t size_lit_watches = SIZE_WATCHES(*lit_watches);
      inner_ticks +=
            1 + kissat_cache_lines(size_lit_watches, sizeof(watch));
      for (const watch *q = begin_lit; q != end_lit; q++) {
        if (p == q) {
          continue;
        }
        const watch lit_watch = *q;
        if (!lit_watch.type.binary) {
          break;
        }
        if (not_lit == src && lit_watch.binary.lit == ILLEGAL_LIT) {
          continue;
        }
        if (!redundant && lit_watch.binary.redundant) {
          continue;
        }
        const unsigned other = lit_watch.binary.lit;
        if (other == dst) {
          transitive = true;
          break;
        }
        const value value = VALUE(other);
        if (value < 0) {
          LOG("both %s and %s reachable from %s",
                LOGLIT(NOT(other)), LOGLIT(other), LOGLIT(src));
          failed = true;
          break;
        }
        if (!value) {
          transitive_assign(solver, other);
        }
      }
    }

    assert(solver->probing);

    assert(solver->propagate <= propagate);
    const unsigned propagated = propagate - solver->propagate;

    ADD(transitive_propagations, propagated);
    ADD(probing_propagations, propagated);
    ADD(propagations, propagated);

    ADD(transitive_ticks, inner_ticks);
    ADD(probing_ticks, inner_ticks);
    ADD(ticks, inner_ticks);

    transitive_backtrack(solver, saved);

    if (transitive) {
      LOGBINARY(src, dst, "transitive reduce");
      INC(transitive_reduced);
      watches *dst_watches = all_watches + dst;
      watch dst_watch = src_watch;
      assert(dst_watch.binary.lit == dst);
      assert(dst_watch.binary.redundant == redundant);
      dst_watch.binary.lit = src;
      REMOVE_WATCHES(*dst_watches, dst_watch);
      kissat_delete_binary(solver,
            src_watch.binary.redundant,
            src_watch.binary.hyper, false, src, dst);
      p->binary.lit = ILLEGAL_LIT;
      reduced++;
      res = true;
    }

    if (failed) {
      break;
    }
    if (solver->statistics.transitive_ticks > limit) {
      break;
    }
    if (TERMINATED(transitive_terminated_1)) {
      break;
    }
  }

  if (reduced) {
    *reduced_ptr += reduced;
    assert(begin_src == BEGIN_WATCHES(WATCHES(src)));
    assert(end_src == END_WATCHES(WATCHES(src)));
    watch *q = begin_src;
    for (const watch *p = begin_src; p != end_src; p++) {
      const watch src_watch = *q++ = *p;
      if (!src_watch.type.binary) {
        *q++ = *++p;
        continue;
      }
      if (src_watch.binary.lit == ILLEGAL_LIT) {
        q--;
      }
    }
    assert(end_src - q == (ptrdiff_t) reduced);
    SET_END_OF_WATCHES(*src_watches, q);
  }

  if (failed) {
    LOG("transitive failed literal %s", LOGLIT(not_src));
    INC(transitive_units);
    *units += 1;
    res = true;

    kissat_learned_unit(solver, src);

    assert(!solver->level);
    (void) kissat_probing_propagate(solver, 0, true);
  }

  return res;
}


static inline bool less_stable_transitive(kissat *solver,
      const flags *const flags, const heap *scores,
      unsigned a, unsigned b) {
#ifdef NDEBUG
  (void) solver;
#endif
  const unsigned i = IDX(a);
  const unsigned j = IDX(b);
  const bool p = flags[i].transitive;
  const bool q = flags[j].transitive;
  if (!p && q) {
    return true;
  }
  if (p && !q) {
    return false;
  }
  const double s = kissat_get_heap_score(scores, i);
  const double t = kissat_get_heap_score(scores, j);
  if (s < t) {
    return true;
  }
  if (s > t) {
    return false;
  }
  return i < j;
}

static inline unsigned less_focused_transitive(kissat *solver,
      const flags *const flags, const links *links,
      unsigned a, unsigned b) {
#ifdef NDEBUG
  (void) solver;
#endif
  const unsigned i = IDX(a);
  const unsigned j = IDX(b);
  const bool p = flags[i].transitive;
  const bool q = flags[j].transitive;
  if (!p && q) {
    return true;
  }
  if (p && !q) {
    return false;
  }
  const unsigned s = links[i].stamp;
  const unsigned t = links[j].stamp;
  return s < t;
}

#define LESS_STABLE_PROBE(A,B) \
  less_stable_transitive (solver, flags, scores, (A), (B))

#define LESS_FOCUSED_PROBE(A,B) \
  less_focused_transitive (solver, flags, links, (A), (B))

static void sort_stable_transitive(kissat *solver, unsigneds *probes) {
  const flags *const flags = solver->flags;
  const heap *const scores = &solver->scores;
  SORT_STACK(unsigned, *probes, LESS_STABLE_PROBE);
}

static void sort_focused_transitive(kissat *solver, unsigneds *probes) {
  const flags *const flags = solver->flags;
  const links *const links = solver->links;
  SORT_STACK(unsigned, *probes, LESS_FOCUSED_PROBE);
}

static void sort_transitive(kissat *solver, unsigneds *probes) {
  if (solver->stable) {
    sort_stable_transitive(solver, probes);
  } else {
    sort_focused_transitive(solver, probes);
  }
}

static void schedule_transitive(kissat *solver, unsigneds *probes) {
  assert(EMPTY_STACK(*probes));
  for (all_variables(idx))
    if (ACTIVE(idx)) {
      PUSH_STACK(*probes, idx);
    }
  sort_transitive(solver, probes);
  kissat_very_verbose(solver, "scheduled %zu transitive probes",
        SIZE_STACK(*probes));
}

void kissat_transitive_reduction(kissat *solver) {
  if (solver->inconsistent) {
    return;
  }
  assert(solver->watching);
  assert(solver->probing);
  assert(!solver->level);
  if (!GET_OPTION(transitive)) {
    return;
  }
  if (TERMINATED(transitive_terminated_2)) {
    return;
  }
  START(transitive);
  INC(transitive_reductions);
#if !defined(NDEBUG) || defined(METRICS)
  assert(!solver->transitive_reducing);
  solver->transitive_reducing = true;
#endif
  prioritize_binaries(solver);
  bool success = false;
  uint64_t reduced = 0;
  unsigned units = 0;

  SET_EFFORT_LIMIT(limit, transitive, transitive_ticks, 0);

#ifndef QUIET
  const unsigned active = solver->active;
  const uint64_t old_ticks = solver->statistics.transitive_ticks;
  kissat_extremely_verbose(solver,
        "starting with %" PRIu64 " transitive ticks",
        old_ticks);
#endif
  unsigned probed = 0;
  unsigneds probes;
  INIT_STACK(probes);
  schedule_transitive(solver, &probes);
  bool terminate = false;
  while (!terminate && !EMPTY_STACK(probes)) {
    const unsigned idx = POP_STACK(probes);
    solver->flags[idx].transitive = false;
    if (!ACTIVE(idx)) {
      continue;
    }
    for (unsigned sign = 0; !terminate && sign < 2; sign++) {
      const unsigned lit = 2 * idx + sign;
      if (solver->values[lit]) {
        continue;
      }
      probed++;
      if (transitive_reduce(solver, lit, limit, &reduced, &units)) {
        success = true;
      }
      if (solver->inconsistent) {
        terminate = true;
      } else if (solver->statistics.transitive_ticks > limit) {
        terminate = true;
      } else if (TERMINATED(transitive_terminated_3)) {
        terminate = true;
      }
    }
  }
  const size_t remain = SIZE_STACK(probes);
  if (remain) {
    if (!GET_OPTION(transitivekeep)) {
      kissat_very_verbose(solver,
            "dropping remaining %zu transitive candidates",
            remain);
      while (!EMPTY_STACK(probes)) {
        const unsigned idx = POP_STACK(probes);
        solver->flags[idx].transitive = false;
      }
    }
  } else {
    kissat_very_verbose(solver, "transitive reduction complete");
  }
  RELEASE_STACK(probes);

#ifndef QUIET
  const uint64_t new_ticks = solver->statistics.transitive_ticks;
  const uint64_t delta_ticks = new_ticks - old_ticks;
  kissat_extremely_verbose(solver,
        "finished at %" PRIu64 " after %" PRIu64
        " transitive ticks", new_ticks, delta_ticks);
#endif
  kissat_phase(solver, "transitive", GET(probings),
        "probed %u (%.0f%%): reduced %" PRIu64 ", units %u",
        probed, kissat_percent(probed, 2 * active), reduced, units);

#if !defined(NDEBUG) || defined(METRICS)
  assert(solver->transitive_reducing);
  solver->transitive_reducing = false;
#endif
  REPORT(!success, 't');
  STOP(transitive);
#ifdef QUIET
  (void) success;
#endif
}

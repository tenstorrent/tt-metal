#include "dominate.h"
#include "fastassign.h"
#include "prophyper.h"

static inline void watch_hyper_delayed(kissat *solver,
      watches *all_watches, unsigneds *delayed) {
  assert(all_watches == solver->watches);
  assert(delayed == &solver->delayed);
  const unsigned *const end_delayed = END_STACK(*delayed);
  unsigned const *d = BEGIN_STACK(*delayed);
  while (d != end_delayed) {
    const unsigned lit = *d++;
    assert(d != end_delayed);
    const watch watch = {.raw = *d++ };
    assert(lit < LITS);
    watches *lit_watches = all_watches + lit;
    if (watch.type.binary) {
      assert(solver->probing);
      assert(watch.binary.hyper);
      assert(watch.binary.redundant);
      LOGBINARY(lit, watch.binary.lit,
            "watching blocking %s in %s",
            LOGLIT(lit), LOGLIT(watch.binary.lit));
      assert(lit < LITS);
      PUSH_WATCHES(*lit_watches, watch);
    } else {
      assert(d != end_delayed);
      const reference ref = *d++;
      const unsigned blocking = watch.blocking.lit;
      LOGREF(ref, "watching %s blocking %s in", LOGLIT(lit),
            LOGLIT(blocking));
      kissat_push_blocking_watch(solver, lit_watches, blocking, ref);
    }
  }
  CLEAR_STACK(*delayed);
}

static inline void delay_watching_hyper(kissat *solver, unsigneds *delayed,
      unsigned lit, unsigned other) {
  assert(delayed == &solver->delayed);
  const watch watch = kissat_binary_watch(other, true, true);
  PUSH_STACK(*delayed, lit);
  PUSH_STACK(*delayed, watch.raw);
}

static inline void kissat_assign_binary_at_level_one(kissat *solver,
      value *values, assigned *assigned,
      bool redundant, unsigned lit,
      unsigned other) {
  assert(solver->probing);
  assert(VALUE(other) < 0);
  assert(solver->level == 1);
  assert(LEVEL(other) == 1);
  kissat_fast_binary_assign(solver, true, 1,
        values, assigned, redundant, lit, other);
}

#define HYPER_PROPAGATION
#define PROPAGATE_LITERAL large_propagate_literal
#define PROPAGATION_TYPE "large"

#include "proplit.h"

static inline clause *binary_propagate_literal(kissat *solver, unsigned lit) {
  assert(solver->probing);
  assert(solver->watching);
  assert(solver->level == 1);

  const watches *const all_watches = solver->watches;
  assigned *const assigned = solver->assigned;
  value *const values = solver->values;

  LOG("binary propagating %s", LOGLIT(lit));
  assert(VALUE(lit) > 0);
  const unsigned not_lit = NOT(lit);

  assert(not_lit < LITS);

  const watches *const watches = all_watches + not_lit;

  const size_t size_watches = SIZE_WATCHES(*watches);
  const uint64_t ticks = kissat_cache_lines(size_watches, sizeof(watch));
  solver->ticks += ticks;

  const watch *const begin = BEGIN_CONST_WATCHES(*watches);
  const watch *const end = END_CONST_WATCHES(*watches);
  watch const *p = begin;

  clause *res = 0;

  while (p != end) {
    watch head = *p++;
    if (!head.type.binary) {
      p++;
      continue;
    }
    const unsigned other = head.binary.lit;
    assert(VALID_INTERNAL_LITERAL(other));
    const value other_value = values[other];
    if (other_value > 0) {
      continue;
    }
    const bool redundant = head.binary.redundant;
    if (other_value < 0) {
      res = kissat_binary_conflict(solver, redundant, not_lit, other);
      break;
    }
    assert(!other_value);
    kissat_assign_binary_at_level_one(solver,
          values, assigned,
          redundant, other, not_lit);
  }
  return res;
}

static inline clause *hyper_propagate(kissat *solver, const clause *ignore) {
  assert(solver->probing);
  assert(solver->watching);
  assert(solver->level == 1);

  clause *res = 0;

  unsigned *propagate_binary = solver->propagate;
  unsigned *propagate_large = propagate_binary;
  unsigned *end_trail;

  while (!res && propagate_large != (end_trail = END_ARRAY(solver->trail))) {
    if (propagate_binary != end_trail) {
      res = binary_propagate_literal(solver, *propagate_binary++);
    } else {
      res = large_propagate_literal(solver, ignore, *propagate_large++);
    }
  }

  solver->propagate = propagate_large;

  return res;
}

clause *kissat_hyper_propagate(kissat *solver, const clause *ignore) {
  assert(solver->probing);
  assert(solver->watching);
  assert(!solver->inconsistent);
  assert(solver->level == 1);
  assert(!solver->unflushed);

  START(propagate);

  solver->ticks = 0;
  const unsigned *start = solver->propagate;
  clause *conflict = hyper_propagate(solver, ignore);
  assert(start <= solver->propagate);
  const unsigned propagated = solver->propagate - start;
  kissat_update_probing_propagation_statistics(solver, propagated);
  ADD(hyper_propagations, propagated);
  ADD(hyper_ticks, solver->ticks);

  STOP(propagate);

  return conflict;
}

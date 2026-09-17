#include "decide.h"
#include "inlineframes.h"
#include "inlineheap.h"
#include "inlinequeue.h"

#include <inttypes.h>

static unsigned last_enqueued_unassigned_variable(kissat *solver) {
  assert(solver->unassigned);
  const links *const links = solver->links;
  const value *const values = solver->values;
  unsigned res = solver->queue.search.idx;
  if (values[LIT(res)]) {
    do {
      res = links[res].prev;
      assert(!DISCONNECTED(res));
    } while (values[LIT(res)]);
    kissat_update_queue(solver, links, res);
  }
#ifdef LOGGING
  const unsigned stamp = links[res].stamp;
  LOG("last enqueued unassigned %s stamp %u", LOGVAR(res), stamp);
#endif
#ifdef CHECK_QUEUE
  for (unsigned i = links[res].next; !DISCONNECTED(i); i = links[i].next) {
    assert(VALUE(LIT(i)));
  }
#endif
  return res;
}

static unsigned largest_score_unassigned_variable(kissat *solver) {
  unsigned res = kissat_max_heap(&solver->scores);
  const value *const values = solver->values;
  while (values[LIT(res)]) {
    kissat_pop_max_heap(solver, &solver->scores);
    res = kissat_max_heap(&solver->scores);
  }
#if defined(LOGGING) || defined(CHECK_HEAP)
  const double score = kissat_get_heap_score(&solver->scores, res);
#endif
  LOG("largest score unassigned %s score %g", LOGVAR(res), score);
#ifdef CHECK_HEAP
  for (all_variables(idx)) {
    if (!ACTIVE(idx)) {
      continue;
    }
    if (VALUE(LIT(idx))) {
      continue;
    }
    const double idx_score = kissat_get_heap_score(&solver->scores, idx);
    assert(score >= idx_score);
  }
#endif
  return res;
}

unsigned kissat_next_decision_variable(kissat *solver) {
  unsigned res;
  if (solver->stable) {
    res = largest_score_unassigned_variable(solver);
  } else {
    res = last_enqueued_unassigned_variable(solver);
  }
  LOG("next decision %s", LOGVAR(res));
  return res;
}

static inline value decide_phase(kissat *solver, unsigned idx) {
  bool force = GET_OPTION(forcephase);

  value *target;
  if (force) {
    target = 0;
  } else if (!GET_OPTION(target)) {
    target = 0;
  } else if (solver->stable || GET_OPTION(target) > 1) {
    target = solver->phases.target + idx;
  } else {
    target = 0;
  }

  value *saved;
  if (force) {
    saved = 0;
  } else if (GET_OPTION(phasesaving)) {
    saved = solver->phases.saved + idx;
  } else {
    saved = 0;
  }

  value res = 0;

  if (!res && target && (res = *target)) {
    LOG("%s uses target decision phase %d", LOGVAR(idx), (int) res);
    INC(target_decisions);
  }

  if (!res && saved && (res = *saved)) {
    LOG("%s uses saved decision phase %d", LOGVAR(idx), (int) res);
    INC(saved_decisions);
  }

  if (!res) {
    res = INITIAL_PHASE;
    LOG("%s uses initial decision phase %d", LOGVAR(idx), (int) res);
    INC(initial_decisions);
  }
  assert(res);

  return res;
}

void kissat_decide(kissat *solver) {
  START(decide);
  assert(solver->unassigned);
  INC(decisions);
  if (solver->stable) {
    INC(stable_decisions);
  } else {
    INC(focused_decisions);
  }
  solver->level++;
  assert(solver->level != INVALID_LEVEL);
  const unsigned idx = kissat_next_decision_variable(solver);
  const value value = decide_phase(solver, idx);
  unsigned lit = LIT(idx);
  if (value < 0) {
    lit = NOT(lit);
  }
  kissat_push_frame(solver, lit);
  assert(solver->level < SIZE_STACK(solver->frames));
  LOG("decide literal %s", LOGLIT(lit));
  kissat_assign_decision(solver, lit);
  STOP(decide);
}

void kissat_internal_assume(kissat *solver, unsigned lit) {
  assert(solver->unassigned);
  assert(!VALUE(lit));
  solver->level++;
  assert(solver->level != INVALID_LEVEL);
  kissat_push_frame(solver, lit);
  assert(solver->level < SIZE_STACK(solver->frames));
  LOG("assuming literal %s", LOGLIT(lit));
  kissat_assign_decision(solver, lit);
}

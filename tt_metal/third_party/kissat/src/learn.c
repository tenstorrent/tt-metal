#include "backtrack.h"
#include "inline.h"
#include "learn.h"
#include "reluctant.h"

#include <inttypes.h>

static unsigned determine_new_level(kissat *solver, unsigned jump) {
  assert(solver->level);
  const unsigned back = solver->level - 1;
  assert(jump <= back);

  const unsigned delta = back - jump;
  const unsigned limit =
        GET_OPTION(chrono) ? (unsigned) GET_OPTION(chronolevels) : UINT_MAX;

  unsigned res;

  if (!delta) {
    res = jump;
    LOG("using identical backtrack and jump level %u", res);
  } else if (delta > limit) {
    res = back;
    LOG("backjumping over %u levels (%u - %u) considered inefficient",
          delta, back, jump);
    LOG("backtracking chronologically to backtrack level %u", res);
    INC(chronological);
  } else {
    res = jump;
    LOG("backjumping over %u levels (%u - %u) considered efficient",
          delta, back, jump);
    LOG("backjumping non-chronologically to jump level %u", res);
  }
  return res;
}

static void learn_unit(kissat *solver, unsigned not_uip) {
  assert(not_uip == PEEK_STACK(solver->clause, 0));
  LOG("learned unit clause %s triggers iteration", LOGLIT(not_uip));
  const unsigned new_level = determine_new_level(solver, 0);
  kissat_backtrack_after_conflict(solver, new_level);
  kissat_learned_unit(solver, not_uip);
  solver->iterating = true;
  INC(learned_units);
}

static void learn_binary(kissat *solver, unsigned not_uip) {
  const unsigned other = PEEK_STACK(solver->clause, 1);
  const unsigned jump_level = LEVEL(other);
  const unsigned new_level = determine_new_level(solver, jump_level);
  kissat_backtrack_after_conflict(solver, new_level);
#ifndef NDEBUG
  const reference ref =
#endif
        kissat_new_redundant_clause(solver, 1);
  assert(ref == INVALID_REF);
  kissat_assign_binary(solver, true, not_uip, other);
  kissat_eager_subsume(solver);
}

static void learn_reference(kissat *solver, unsigned not_uip, unsigned glue) {
  assert(solver->level > 1);
  assert(SIZE_STACK(solver->clause) > 2);
  unsigned *lits = BEGIN_STACK(solver->clause);
  assert(lits[0] == not_uip);
  unsigned *q = lits + 1;
  unsigned jump_lit = *q;
  unsigned jump_level = LEVEL(jump_lit);
  const unsigned *const end = END_STACK(solver->clause);
  const unsigned backtrack_level = solver->level - 1;
  assigned *all_assigned = solver->assigned;
  for (unsigned *p = lits + 2; p != end; p++) {
    const unsigned lit = *p;
    const unsigned idx = IDX(lit);
    const unsigned level = all_assigned[idx].level;
    if (jump_level >= level) {
      continue;
    }
    jump_level = level;
    jump_lit = lit;
    q = p;
    if (level == backtrack_level) {
      break;
    }
  }
  *q = lits[1];
  lits[1] = jump_lit;
  const reference ref = kissat_new_redundant_clause(solver, glue);
  assert(ref != INVALID_REF);
  clause *c = kissat_dereference_clause(solver, ref);
  c->used = 1 + (glue <= (unsigned) GET_OPTION(tier2));
  const unsigned new_level = determine_new_level(solver, jump_level);
  kissat_backtrack_after_conflict(solver, new_level);
  kissat_assign_reference(solver, not_uip, ref, c);
  if (GET_OPTION(eagersubsume)) {
    kissat_eager_subsume(solver);
    kissat_push_clueue(&solver->clueue, ref);
  }
}

void kissat_update_learned(kissat *solver, unsigned glue, unsigned size) {
  assert(!solver->probing);
  INC(clauses_learned);
  LOG("learned[%" PRIu64 "] clause glue %u size %u",
        GET(clauses_learned), glue, size);
  if (solver->stable) {
    kissat_tick_reluctant(&solver->reluctant);
  }
  ADD(literals_learned, size);
#ifndef QUIET
  UPDATE_AVERAGE(size, size);
#endif
  UPDATE_AVERAGE(fast_glue, glue);
  UPDATE_AVERAGE(slow_glue, glue);
}

void kissat_learn_clause(kissat *solver) {
  const unsigned not_uip = PEEK_STACK(solver->clause, 0);
  const unsigned size = SIZE_STACK(solver->clause);
  const size_t glue = SIZE_STACK(solver->levels);
  assert(glue <= UINT_MAX);
  if (!solver->probing) {
    kissat_update_learned(solver, glue, size);
  }
  assert(size > 0);
  if (size == 1) {
    learn_unit(solver, not_uip);
  } else if (size == 2) {
    learn_binary(solver, not_uip);
  } else {
    learn_reference(solver, not_uip, glue);
  }
}

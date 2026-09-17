#ifndef _inlineassign_h_INLCUDED
#define _inlineassign_h_INLCUDED

#include "internal.h"
#include "inlinevector.h"

typedef struct kissat kissat;

#ifdef FAST_ASSIGN
#define kissat_assign kissat_fast_assign
#endif

static inline void
kissat_assign(kissat *solver, const bool probing, const unsigned level,
#ifdef FAST_ASSIGN
      value *values, assigned *assigned,
#endif
      bool binary, bool redundant, unsigned lit, unsigned reason) {
  assert(binary || !redundant);
  const unsigned not_lit = NOT(lit);

  watches watches = WATCHES(not_lit);
  if (!kissat_empty_vector(&watches)) {
    watch *w = BEGIN_WATCHES(watches);
    __builtin_prefetch(w, 0, 1);
  }

#ifndef FAST_ASSIGN
  value *values = solver->values;
#endif
  assert(!values[lit]);
  assert(!values[not_lit]);

  values[lit] = 1;
  values[not_lit] = -1;

  assert(solver->unassigned > 0);
  solver->unassigned--;

  if (!level) {
    kissat_mark_fixed_literal(solver, lit);
    assert(solver->unflushed < UINT_MAX);
    solver->unflushed++;
    if (reason != UNIT_REASON) {
      CHECK_AND_ADD_UNIT(lit);
      ADD_UNIT_TO_PROOF(lit);
    }
  }

  const size_t trail = SIZE_ARRAY(solver->trail);
  PUSH_ARRAY(solver->trail, lit);

  const unsigned idx = IDX(lit);

#if !defined(HYPER_PROPAGATION) && !defined(PROBING_PROPAGATION)
  if (!probing) {
    const bool negated = NEGATED(lit);
    const value value = BOOL_TO_VALUE(negated);
    SAVED(idx) = value;
  }
#endif

  assert(level <= MAX_LEVEL);
  assert(trail <= MAX_TRAIL);

  struct assigned b;
  b.level = level;
  b.analyzed = false;
  b.poisoned = false;
  b.removable = false;
  b.shrinkable = false;
  b.trail = trail;
  b.binary = binary;
  b.redundant = redundant;
  b.reason = reason;

#ifndef FAST_ASSIGN
  assigned *assigned = solver->assigned;
#endif
  struct assigned *a = assigned + idx;
  *a = b;
}

static inline unsigned kissat_assignment_level(kissat *solver,
      value *values, assigned *assigned,
      unsigned lit, clause *reason) {
  unsigned res = 0;
  for (all_literals_in_clause(other, reason)) {
    if (other == lit) {
      continue;
    }
    assert(values[other] < 0), (void) values;
    const unsigned other_idx = IDX(other);
    struct assigned *a = assigned + other_idx;
    const unsigned level = a->level;
    if (res < level) {
      res = level;
    }
  }
#ifdef NDEBUG
  (void) solver;
#endif
  return res;
}

#endif

#ifndef _fastassign_h_INCLUDED
#define _fastassign_h_INCLUDED

#define FAST_ASSIGN

#include "inline.h"
#include "inlineassign.h"

static inline void kissat_fast_binary_assign(kissat *solver,
      const bool probing, const unsigned level,
      value *values, assigned *assigned,
      bool redundant, unsigned lit, unsigned other) {
  kissat_fast_assign(solver, probing, level, values, assigned,
        true, redundant, lit, other);
  LOGBINARY(lit, other, "assign %s %s reason",
        LOGLIT(lit), redundant ? "redundant" : "irredundant");
}

static inline void kissat_fast_assign_reference(kissat *solver,
      value *values, assigned *assigned,
      unsigned lit, reference ref, clause *reason) {
  assert(reason == kissat_dereference_clause(solver, ref));
  const unsigned level =
        kissat_assignment_level(solver, values, assigned, lit, reason);
  assert(level <= solver->level);
  assert(ref != DECISION_REASON);
  assert(ref != UNIT_REASON);
  kissat_fast_assign(solver, solver->probing, level,
        values, assigned, false, false, lit, ref);
  LOGREF(ref, "assign %s reason", LOGLIT(lit));
}

#endif

#ifndef _assume_h_INCLUDED
#define _assume_h_INCLUDED

#include <stdbool.h>

#include "internal.h"

void kissat_reset_assumptions(kissat *solver);
void kissat_assume_literal(kissat *solver, int elit, unsigned ilit);
int kissat_assign_assumption(kissat *solver);
int kissat_propagate_assumptions(kissat *solver);

static inline bool kissat_assuming(kissat *solver) {
  return solver->level < SIZE_STACK(solver->assumptions);
}

#endif

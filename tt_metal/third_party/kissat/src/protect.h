#ifndef _protect_h_INCLUDED
#define _protect_h_INCLUDED

#define PROTECT(IDX) (solver->protect[assert((IDX) < VARS), IDX])

typedef struct kissat kissat;

void kissat_protect_variable(kissat *solver, unsigned idx);
void kissat_unprotect_variable(kissat *solver, unsigned idx);
void kissat_move_protection(kissat *solver, unsigned from_idx, unsigned to_idx);

#endif

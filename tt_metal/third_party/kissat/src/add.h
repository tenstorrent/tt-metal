#ifndef _add_h_INCLUDED
#define _add_h_INCLUDED

#include <stdbool.h>

typedef struct kissat kissat;

bool kissat_add_external_literal(kissat *solver, int elit);
void kissat_finish_external_clause(kissat *solver, bool restored);

#endif

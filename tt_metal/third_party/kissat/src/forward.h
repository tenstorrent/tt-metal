#ifndef _forward_h_INCLUDED
#define _forward_h_INCLUDED

#include <stdbool.h>

struct kissat;

bool kissat_forward_subsume_temporary(struct kissat *);
bool kissat_forward_subsume_during_elimination(struct kissat *);

#endif

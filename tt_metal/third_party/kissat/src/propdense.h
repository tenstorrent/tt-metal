#ifndef _propdense_h_INCLUDED
#define _propdense_h_INCLUDED

#include <limits.h>
#include <stdbool.h>

struct kissat;

bool kissat_dense_propagate(struct kissat *);

#endif

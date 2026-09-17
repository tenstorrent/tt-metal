#ifndef _walk_h_INCLUDED
#define _walk_h_INCLUDED

#include <stdbool.h>

struct kissat;

bool kissat_walking(struct kissat *);
void kissat_walk(struct kissat *);
int kissat_walk_initially(struct kissat *);

#endif

#ifndef _probe_h_INCLUDED
#define _probe_h_INCLUDED

#include <stdbool.h>

struct kissat;

bool kissat_probing(struct kissat *);
int kissat_probe(struct kissat *);

#endif

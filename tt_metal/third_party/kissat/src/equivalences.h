#ifndef _equivs_h_INCLUDED
#define _equivs_h_INCLUDED

#include <stdbool.h>

struct kissat;

bool kissat_find_equivalence_gate(struct kissat *, unsigned lit);

#endif

#ifndef _eliminate_h_INCLUDED
#define _eliminate_h_INCLUDED

#include <stdbool.h>

struct ruler;
struct simplifier;
bool eliminate_variables (struct simplifier *, unsigned round);
void try_to_increase_elimination_bound (struct ruler *);

#endif

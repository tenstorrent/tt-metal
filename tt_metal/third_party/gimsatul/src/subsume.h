#ifndef _subsume_h_INCLUDED
#define _subsume_h_INCLUDED

#include <stdbool.h>

struct simplifier;

bool subsume_clauses (struct simplifier *, unsigned round);

#endif

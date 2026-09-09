#ifndef _substitute_h_INCLUDED
#define _substitute_h_INCLUDED

#include <stdbool.h>

struct simplifier;

bool equivalent_literal_substitution (struct simplifier *, unsigned round);

#endif

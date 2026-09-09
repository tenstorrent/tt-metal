#ifndef _definition_h_INCLUDED
#define _definition_h_INCLUDED

#include <stdbool.h>

struct simplifier;
bool find_definition (struct simplifier *, unsigned lit);

#endif

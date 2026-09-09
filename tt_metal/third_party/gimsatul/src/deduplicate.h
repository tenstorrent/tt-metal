#ifndef _deduplicate_h_INCLUDED
#define _deduplicate_h_INCLUDED

#include <stdbool.h>

struct simplifier;

bool remove_duplicated_binaries (struct simplifier *, unsigned round);

#endif

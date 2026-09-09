#ifndef _vivify_h_INCLUDED
#define _vivify_h_INCLUDED

#include <stdbool.h>

#include "options.h"
#include "watches.h"

struct ring;
struct watch;

void vivify_clauses (struct ring *);

#endif

#ifndef _propagate_h_INCLUDED
#define _propagate_h_INCLUDED

#include <stdbool.h>

struct clause;
struct ring;

struct watch *ring_propagate (struct ring *, bool stop_at_conflict,
                              struct clause *ignored_large_clause);

#endif

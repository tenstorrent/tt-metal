#ifndef _search_h_INCLUDED
#define _search_h_INCLUDED

#include <stdbool.h>

struct ring;

int search (struct ring *);
void iterate (struct ring *);
bool terminate_ring (struct ring *ring);
bool backtrack_propagate_iterate (struct ring *ring);

#endif

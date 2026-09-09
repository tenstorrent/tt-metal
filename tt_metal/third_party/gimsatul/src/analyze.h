#ifndef _analyze_h_INCLUDED
#define _analyze_h_INCLUDED

#include <stdbool.h>

struct ring;
struct watch;

bool analyze (struct ring *, struct watch *conflict);
void clear_analyzed (struct ring *);

#endif

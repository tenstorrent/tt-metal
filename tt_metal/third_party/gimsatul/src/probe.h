#ifndef _probe_h_INCLUDED
#define _probe_h_INCLUDED

#include <stdbool.h>

struct ring;

bool probing (struct ring *);
int probe (struct ring *);

#endif

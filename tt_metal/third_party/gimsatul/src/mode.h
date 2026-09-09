#ifndef _switch_h_INCLUDED
#define _switch_h_INCLUDED

#include <stdbool.h>

struct ring;

bool switching_mode (struct ring *);
void switch_mode (struct ring *);

#endif

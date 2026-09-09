#ifndef _restart_h_INCLUDED
#define _restart_h_INCLUDED

#include <stdbool.h>

struct ring;

bool restarting (struct ring *);
void restart (struct ring *);

#endif

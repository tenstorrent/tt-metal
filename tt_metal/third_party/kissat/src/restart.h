#ifndef _restart_h_INCLUDED
#define _restart_h_INCLUDED

#include <stdbool.h>

struct kissat;

bool kissat_restarting(struct kissat *);
void kissat_restart(struct kissat *);
void kissat_new_focused_restart_limit(struct kissat *);

#endif

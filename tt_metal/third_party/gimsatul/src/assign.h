#ifndef _assign_h_INCLUDED
#define _assign_h_INCLUDED

struct ring;
struct watch;

void assign_with_reason (struct ring *, unsigned lit, struct watch *reason);
void assign_ring_unit (struct ring *, unsigned unit);
void assign_decision (struct ring *, unsigned decision);

#endif

#ifndef _clone_h_INCLUDED
#define _clone_h_INCLUDED

struct ring;
struct ruler;
void copy_ring (struct ring *dst);
void copy_ruler (struct ring *dst);
void clone_rings (struct ruler *);

#endif

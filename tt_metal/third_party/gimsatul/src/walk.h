#ifndef _walk_h_INCLUDED
#define _walk_h_INCLUDED

struct ring;
struct clause;

struct counter {
  unsigned count;
  struct clause *clause;
};

void local_search (struct ring *);

#endif

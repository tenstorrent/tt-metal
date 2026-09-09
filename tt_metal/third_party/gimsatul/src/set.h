#ifndef _set_h_INCLUDED
#define _set_h_INCLUDED

#include <stdint.h>
#include <stdlib.h>

struct set {
  size_t size;
  size_t deleted;
  size_t allocated;
  void **table;
  struct {
    size_t (*function) (void *state, void *ptr);
    void *state;
  } hash;
};

void set_insert (struct set *, void *);
void set_remove (struct set *, void *);
void *random_set (uint64_t *, struct set *);

#endif

#ifndef _clueue_h_INCLUDED
#define _clueue_h_INCLUDED

#include "reference.h"

#include <assert.h>

typedef struct clueue clueue;

struct clueue {
  unsigned size, next;
  reference *elements;
};

struct kissat;

void kissat_clear_clueue(struct kissat *, clueue *);
void kissat_init_clueue(struct kissat *, clueue *, unsigned size);
void kissat_release_clueue(struct kissat *, clueue *);

static inline void kissat_push_clueue(clueue *clueue, reference element) {
  if (!clueue->size) {
    return;
  }
  assert(clueue->next < clueue->size);
  clueue->elements[clueue->next++] = element;
  if (clueue->next == clueue->size) {
    clueue->next = 0;
  }
}

void kissat_eager_subsume(struct kissat *);

#endif

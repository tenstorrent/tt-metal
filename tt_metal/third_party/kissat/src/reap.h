#ifndef _reap_h_INCLUDED
#define _reap_h_INCLUDED

#include <stdbool.h>

#include "stack.h"

typedef struct reap reap;

struct reap {
  size_t num_elements;
  unsigned last_deleted;
  unsigned min_bucket;
  unsigned max_bucket;
  unsigneds buckets[33];
};

struct kissat;

void kissat_init_reap(struct kissat *, reap *);
void kissat_release_reap(struct kissat *solver, reap *);

static inline bool kissat_empty_reap(reap *reap) {
  return !reap->num_elements;
}

static inline size_t kissat_size_reap(reap *reap) {
  return reap->num_elements;
}

#endif

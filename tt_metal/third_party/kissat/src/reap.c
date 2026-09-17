#include "allocate.h"
#include "internal.h"
#include "logging.h"
#include "reap.h"

#include <string.h>

void kissat_init_reap(kissat *solver, reap *reap) {
  LOG("initializing radix heap");
  memset(reap, 0, sizeof * reap);
  assert(!reap->num_elements);
  assert(!reap->last_deleted);
  reap->min_bucket = 32;
  assert(!reap->max_bucket);
#ifndef LOGGING
  (void) solver;
#endif
}

void kissat_release_reap(kissat *solver, reap *reap) {
  LOG("releasing reap");
  for (unsigned i = 0; i < 33; i++) {
    RELEASE_STACK(reap->buckets[i]);
  }
  reap->num_elements = 0;
  reap->last_deleted = 0;
  reap->min_bucket = 32;
  reap->max_bucket = 0;
}

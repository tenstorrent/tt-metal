#ifndef _proprobe_h_INCLUDED
#define _proprobe_h_INCLUDED

#include <stdbool.h>

struct kissat;
struct clause;

struct clause *kissat_probing_propagate(struct kissat *,
      struct clause *, bool flush);

#endif

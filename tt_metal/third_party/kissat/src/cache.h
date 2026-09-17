#ifndef _cache_h_INCLUDED
#define _cache_h_INCLUDED

#include "bits.h"
#include "stack.h"
#include "value.h"

#include <stdbool.h>
#include <stdint.h>

struct kissat;

typedef struct cache cache;
typedef struct line line;

struct line {
  unsigned vars;
  unsigned unsatisfied;
  uint64_t signature;
  uint64_t inserted;
  bits *bits;
};

// *INDENT-OFF*

typedef STACK (line*) lineptrs;
typedef STACK (line) lines;

// *INDENT-ON*

#define kissat_size_

struct cache {
  bool valid;
  bool looked;
  unsigned vars;
  uint64_t inserted;
  size_t last_looked_up_position;
  lines lines;
};

void kissat_clear_cache(struct kissat *);
void kissat_release_cache(struct kissat *);
bool kissat_insert_cache(struct kissat *, unsigned unsatisfied);
void kissat_update_cache(struct kissat *, unsigned unsatisfied);
bits *kissat_lookup_cache(struct kissat *);

#endif

#ifndef _extend_h_INCLUDED
#define _extend_h_INCLUDED

#include <limits.h>

#include "stack.h"
#include "utilities.h"

typedef struct extension extension;

struct extension {
  signed int lit: 30;
  bool autarky: 1;
  bool blocking: 1;
};

// *INDENT-OFF*
typedef STACK (extension) extensions;
// *INDENT-ON*

static inline extension kissat_extension(bool blocking, int lit) {
  assert(ABS(lit) < (1 << 29));
  extension res;
  res.autarky = false;
  res.blocking = blocking;
  res.lit = lit;
  return res;
}

static inline extension kissat_autarky_extension(bool blocking, int lit) {
  assert(ABS(lit) < (1 << 29));
  extension res;
  res.autarky = true;
  res.blocking = blocking;
  res.lit = lit;
  return res;
}

struct kissat;

void kissat_undo_eliminated_assignment(struct kissat *solver);
void kissat_extend(struct kissat *solver);


#endif

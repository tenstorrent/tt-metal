#ifndef _inlineframes_h_INCLUDEDd
#define _inlineframes_h_INCLUDEDd

#include "allocate.h"
#include "internal.h"

static inline void kissat_push_frame(kissat *solver, unsigned decision) {
  assert(!solver->level || decision != UINT_MAX);
  const size_t trail = SIZE_ARRAY(solver->trail);
  assert(trail <= MAX_TRAIL);
  frame frame;
  frame.decision = decision;
  frame.promote = false;
  frame.trail = trail;
  frame.used = 0;
  PUSH_STACK(solver->frames, frame);
}

#endif

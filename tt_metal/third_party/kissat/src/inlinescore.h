#ifndef _inlinescore_h_INCLUDED
#define _inlinescore_h_INCLUDED

#include "inlineheap.h"
#include "inlinevector.h"

static inline void kissat_update_variable_score(kissat *solver, heap *schedule,
      unsigned idx) {
  if (!GET_OPTION(eliminateheap)) {
    return;
  }
  assert(schedule->size);
  const unsigned lit = LIT(idx);
  const unsigned not_lit = NOT(lit);
  size_t pos = SIZE_WATCHES(WATCHES(lit));
  size_t neg = SIZE_WATCHES(WATCHES(not_lit));
  double new_score = ((double) pos) * neg + pos + neg;
  LOG("new elimination score %g for variable %u (pos %zu and neg %zu)",
        new_score, idx, pos, neg);
  kissat_update_heap(solver, schedule, idx, -new_score);
}

#endif

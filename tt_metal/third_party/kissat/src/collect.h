#ifndef _collect_h_INCLUDED
#define _collect_h_INCLUDED

#include "internal.h"

void kissat_dense_collect(kissat *);
void kissat_sparse_collect(kissat *, bool compact, reference start);

static inline void kissat_defrag_watches(kissat *solver) {
  kissat_defrag_vectors(solver, LITS, solver->watches);
}

static inline void kissat_defrag_watches_if_needed(kissat *solver) {
  const size_t size = SIZE_STACK(solver->vectors.stack);
  const size_t size_limit = GET_OPTION(defragsize);
  if (size <= size_limit) {
    return;
  }

  const size_t usable = solver->vectors.usable;
  const size_t usable_limit = (size * GET_OPTION(defraglim)) / 100;
  if (usable <= usable_limit) {
    return;
  }

  INC(vectors_defrags_needed);
  kissat_defrag_watches(solver);
}

#endif

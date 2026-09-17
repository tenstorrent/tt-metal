#ifndef _dense_h_INCLUDED
#define _dense_h_INCLUDED

#include "watch.h"

void kissat_enter_dense_mode(struct kissat *,
      litpairs *saved_irredundant_binary_clauses,
      litwatches *saved_redundant_binary_clauses);

void kissat_resume_sparse_mode(struct kissat *, bool flush_eliminated,
      litpairs *, litwatches *);

#endif

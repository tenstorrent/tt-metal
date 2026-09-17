#ifndef _terminate_h_INCLUDED
#define _terminate_h_INCLUDED

#include "internal.h"

#ifndef QUIET
void kissat_report_termination(kissat *, const char *name,
      const char *file, long lineno,
      const char *fun);
#endif

static inline bool kissat_terminated(kissat *solver, int bit, const char *name,
      const char *file, long lineno, const char *fun) {
  assert(0 <= bit), assert(bit < 64);
#ifdef COVERAGE
  const uint64_t mask = (uint64_t) 1 << bit;
  if (!(solver->termination.flagged & mask)) {
    return false;
  }
  solver->termination.flagged = ~(uint64_t) 0;
#else
  if (!solver->termination.flagged) {
    return false;
  }
#endif
#ifndef QUIET
  kissat_report_termination(solver, name, file, lineno, fun);
#else
  (void) file;
  (void) fun;
  (void) lineno;
  (void) name;
#endif
#if !defined (COVERAGE) && defined(NDEBUG)
  (void) bit;
#endif
  return true;
}

#define TERMINATED(BIT) \
  kissat_terminated (solver, BIT, #BIT, __FILE__, __LINE__, __func__)

#define autarky_terminated_1 0
#define autarky_terminated_2 1
#define autarky_terminated_3 2
#define autarky_terminated_4 3
#define backbone_terminated_1 4
#define backbone_terminated_2 5
#define backbone_terminated_3 6
#define backward_terminated_1 7
#define eliminate_terminated_1 8
#define eliminate_terminated_2 9
#define failed_terminated_1 10
#define failed_terminated_2 11
#define forward_terminated_1 12
#define kitten_terminated_1 13
#define rephase_terminated_1 14
#define rephase_terminated_2 15
#define search_terminated_1 16
#define substitute_terminated_1 17
#define sweep_terminated_1 18
#define sweep_terminated_2 19
#define sweep_terminated_3 20
#define ternary_terminated_1 21
#define ternary_terminated_2 22
#define ternary_terminated_3 23
#define transitive_terminated_1 24
#define transitive_terminated_2 25
#define transitive_terminated_3 26
#define vivify_terminated_1 27
#define vivify_terminated_2 28
#define vivify_terminated_3 29
#define vivify_terminated_4 30
#define walk_terminated_1 31
#define walk_terminated_2 32
#define xors_terminated_1 33

#endif

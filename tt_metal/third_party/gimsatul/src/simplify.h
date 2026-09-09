#ifndef _simplify_h_INCLUDED
#define _simplify_h_INCLUDED

#include "macros.h"
#include "ruler.h"
#include "stack.h"

#include <assert.h>

/*------------------------------------------------------------------------*/

struct simplifier {
  struct ruler *ruler;
  signed char *marks;
  bool *eliminated;
  struct unsigneds resolvent;
  struct clauses gate[2], nogate[2];
};

/*------------------------------------------------------------------------*/

void add_resolvent (struct simplifier *);
void recycle_clause (struct simplifier *, struct clause *, unsigned except);
void recycle_clauses (struct simplifier *, struct clauses *,
                      unsigned except);
void simplify_ruler (struct ruler *);

/*------------------------------------------------------------------------*/

bool simplifying (struct ring *);
int simplify_ring (struct ring *);

/*------------------------------------------------------------------------*/

static inline void mark_eliminate_literal (struct simplifier *simplifier,
                                           unsigned lit) {
  unsigned idx = IDX (lit);
  bool *eliminate = simplifier->ruler->eliminate;
  assert (eliminate);
  if (eliminate[idx])
    return;
#ifdef LOGGING
  struct ruler *ruler = simplifier->ruler;
  ROG ("marking %s to be eliminated", ROGVAR (idx));
#endif
  eliminate[idx] = true;
}

static inline void mark_eliminate_clause (struct simplifier *simplifier,
                                          struct clause *clause) {
  for (all_literals_in_clause (lit, clause))
    mark_eliminate_literal (simplifier, lit);
}

static inline void mark_subsume_literal (struct simplifier *simplifier,
                                         unsigned lit) {
  unsigned idx = IDX (lit);
  bool *subsume = simplifier->ruler->subsume;
  assert (subsume);
  if (subsume[idx])
    return;
#ifdef LOGGING
  struct ruler *ruler = simplifier->ruler;
  ROG ("marking %s to be subsumed", ROGVAR (idx));
#endif
  subsume[idx] = true;
}

static inline void mark_subsume_clause (struct simplifier *simplifier,
                                        struct clause *clause) {
  for (all_literals_in_clause (lit, clause))
    mark_subsume_literal (simplifier, lit);
}

static inline bool
subsumption_ticks_limit_hit (struct simplifier *simplifier) {
  struct ruler *ruler = simplifier->ruler;
  struct ruler_statistics *statistics = &ruler->statistics;
  struct ruler_limits *limits = &ruler->limits;
  return statistics->ticks.subsumption > limits->subsumption;
}

static inline bool
elimination_ticks_limit_hit (struct simplifier *simplifier) {
  struct ruler *ruler = simplifier->ruler;
  struct ruler_statistics *statistics = &ruler->statistics;
  struct ruler_limits *limits = &ruler->limits;
  return statistics->ticks.elimination > limits->elimination;
}

#endif

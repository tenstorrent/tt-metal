#ifndef _learn_h_INCLUDED
#define _learn_h_INCLUDED

struct kissat;

void kissat_learn_clause(struct kissat *);
void kissat_update_learned(struct kissat *, unsigned glue, unsigned size);

#endif

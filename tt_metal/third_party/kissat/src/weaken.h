#ifndef _weaken_h_INCLUDED
#define _weaken_h_INCLUDED

struct clause;
struct kissat;

void kissat_weaken_unit(struct kissat *, unsigned lit);
void kissat_weaken_binary(struct kissat *, unsigned lit, unsigned other);
void kissat_weaken_clause(struct kissat *, unsigned lit, struct clause *);

#endif

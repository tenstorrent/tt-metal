#ifndef _dominate_h_INCLUDED
#define _dominate_h_INCLUDED

struct kissat;
struct clause;

unsigned kissat_find_dominator(struct kissat *, unsigned, struct clause *);

#endif

#ifndef _analyze_h_INCLUDED
#define _analyze_h_INCLUDED

#include <stdbool.h>

struct clause;
struct kissat;

int kissat_analyze(struct kissat *, struct clause *);
void kissat_reset_only_analyzed_literals(struct kissat *);

#endif

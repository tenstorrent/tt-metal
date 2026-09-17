#ifndef _gates_h_INCLUDED
#define _gates_h_INCLUDED

#include <stdlib.h>
#include <stdbool.h>

struct kissat;
struct clause;

bool kissat_find_gates(struct kissat *, unsigned lit, bool eq_only);
void kissat_get_antecedents(struct kissat *, unsigned lit);

size_t kissat_mark_binaries(struct kissat *, unsigned lit);
void kissat_unmark_binaries(struct kissat *, unsigned lit);

#ifndef METRICS
#define GATE_ELIMINATED(...) true
#else
#define GATE_ELIMINATED(NAME) (&solver->statistics.NAME ## _eliminated)
#endif

#endif

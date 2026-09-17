#ifndef _eliminate_hpp_INCLUDED
#define _eliminate_hpp_INCLUDED

#include <stdbool.h>

struct kissat;
struct clause;

void kissat_update_after_removing_variable(struct kissat *, unsigned);
void kissat_update_after_removing_clause(struct kissat *, struct clause *,
      unsigned except);

void kissat_flush_units_while_connected(struct kissat *);

bool kissat_eliminating(struct kissat *);
int kissat_eliminate(struct kissat *);

void kissat_eliminate_binary(
      struct kissat *, bool weakened, unsigned, unsigned);
void kissat_eliminate_clause(
      struct kissat *, bool weakened, struct clause *, unsigned);

#endif

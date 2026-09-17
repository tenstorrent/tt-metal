#ifndef _flags_h_INCLUDED
#define _flags_h_INCLUDED

#include <stdbool.h>

typedef struct flags flags;

struct flags {
  bool active: 1;
  bool backbone0: 1;
  bool backbone1: 1;
  bool eliminate: 1;
  bool eliminated: 1;
  bool fixed: 1;
  bool probe: 1;
  bool subsume: 1;
  bool sweep: 1;
  bool transitive: 1;
  int assumed: 2;
};

#define FLAGS(IDX) \
  (assert ((IDX) < VARS), (solver->flags + (IDX)))

#define ACTIVE(IDX) (FLAGS(IDX)->active)
#define ELIMINATED(IDX) (FLAGS(IDX)->eliminated)

struct kissat;

void kissat_activate_literal(struct kissat *, unsigned);
void kissat_activate_literals(struct kissat *, unsigned, unsigned *);

void kissat_mark_eliminated_variable(struct kissat *, unsigned idx);
void kissat_mark_substituted_variable(struct kissat *solver,
      unsigned idx, unsigned replacement_lit);
void kissat_mark_fixed_literal(struct kissat *, unsigned lit);
void kissat_mark_autarkic_literal(struct kissat *, unsigned lit);

void kissat_mark_added_literals(struct kissat *, unsigned, unsigned *);
void kissat_mark_removed_literals(struct kissat *, unsigned, unsigned *);

#endif

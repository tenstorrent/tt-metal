#ifndef _assign_h_INCLUDED
#define _assign_h_INCLUDED

#include <stdbool.h>

#define DECISION_REASON	UINT_MAX
#define UNIT_REASON	(DECISION_REASON - 1)

#define INVALID_LEVEL UINT_MAX

#define MAX_LEVEL ((1u<<28)-1)
#define MAX_TRAIL ((1u<<30)-1)

typedef struct assigned assigned;
struct clause;

struct assigned {
  unsigned level: 28;

  bool analyzed: 1;
  bool poisoned: 1;
  bool removable: 1;
  bool shrinkable: 1;

  unsigned trail: 30;

  bool binary: 1;
  bool redundant: 1;

  unsigned reason;
};

#define ASSIGNED(LIT) \
  (assert (VALID_INTERNAL_LITERAL (LIT)), \
   solver->assigned + IDX (LIT))

#define LEVEL(LIT) \
  (ASSIGNED(LIT)->level)

#define REASON(LIT) \
  (ASSIGNED(LIT)->reason)

#ifndef FAST_ASSIGN

#include "reference.h"

struct kissat;
struct clause;

void kissat_assign_unit(struct kissat *, unsigned lit, const char *);
void kissat_learned_unit(struct kissat *, unsigned lit);
void kissat_original_unit(struct kissat *, unsigned lit);

void kissat_assign_decision(struct kissat *, unsigned lit);

void kissat_assign_binary(struct kissat *, bool, unsigned, unsigned);

void kissat_assign_reference(struct kissat *, unsigned lit,
      reference, struct clause *);

#endif

#endif

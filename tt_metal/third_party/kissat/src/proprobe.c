#include "fastassign.h"
#include "proprobe.h"
#include "trail.h"

#define PROPAGATE_LITERAL probing_propagate_literal
#define PROPAGATION_TYPE "probing"
#define PROBING_PROPAGATION

#include "proplit.h"

clause *kissat_probing_propagate(kissat *solver, clause *ignore, bool flush) {
  assert(solver->probing);
  assert(solver->watching);
  assert(!solver->inconsistent);

  START(propagate);

  clause *conflict = 0;
  unsigned *propagate = solver->propagate;
  solver->ticks = 0;
  while (!conflict && propagate != END_ARRAY(solver->trail)) {
    const unsigned lit = *propagate++;
    conflict = probing_propagate_literal(solver, ignore, lit);
  }

  const unsigned propagated = propagate - solver->propagate;
  solver->propagate = propagate;
  kissat_update_probing_propagation_statistics(solver, propagated);
  kissat_update_conflicts_and_trail(solver, conflict, flush);

  STOP(propagate);

  return conflict;
}

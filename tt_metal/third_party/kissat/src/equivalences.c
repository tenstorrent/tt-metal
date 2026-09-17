#include "inlinevector.h"
#include "gates.h"
#include "logging.h"

bool kissat_find_equivalence_gate(kissat *solver, unsigned lit) {
  if (!GET_OPTION(equivalences)) {
    return false;
  }
  if (!kissat_mark_binaries(solver, lit)) {
    return false;
  }
  value *marks = solver->marks;
  unsigned not_lit = NOT(lit);
  watches *watches = &WATCHES(not_lit);
  unsigned replace = INVALID_LIT;
  for (all_binary_large_watches(watch, *watches)) {
    if (!watch.type.binary) {
      continue;
    }
    const unsigned other = watch.binary.lit;
    const unsigned not_other = NOT(other);
    if (!marks[not_other]) {
      continue;
    }
    replace = other;
    break;
  }
  kissat_unmark_binaries(solver, lit);
  if (replace == INVALID_LIT) {
    return false;
  }
  LOG("found equivalence gate %s = %s", LOGLIT(lit), LOGLIT(replace));
  solver->found_equivalence = replace;

  const watch watch1 = kissat_binary_watch(replace, false, false);
  PUSH_STACK(solver->gates[1], watch1);

  const watch watch0 = kissat_binary_watch(NOT(replace), false, false);
  PUSH_STACK(solver->gates[0], watch0);
  solver->gate_eliminated = GATE_ELIMINATED(equivalences);
  INC(equivalences_extracted);
  return true;
}

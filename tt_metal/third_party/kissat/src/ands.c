#include "ands.h"
#include "eliminate.h"
#include "gates.h"
#include "inline.h"

bool kissat_find_and_gate(kissat *solver, unsigned lit, unsigned negative) {
  if (!GET_OPTION(ands)) {
    return false;
  }
  size_t marked = kissat_mark_binaries(solver, lit);
  if (!marked) {
    return false;
  }
  if (marked < 2) {
    kissat_unmark_binaries(solver, lit);
    return false;
  }

  unsigned not_lit = NOT(lit);
  watches *not_watches = &WATCHES(not_lit);

  ward *const arena = BEGIN_STACK(solver->arena);
  value *marks = solver->marks;
  const value *const values = solver->values;

  clause *base = 0;
  for (all_binary_large_watches(watch, *not_watches)) {
    if (watch.type.binary) {
      continue;
    }
    const reference ref = watch.large.ref;
    assert(ref < SIZE_STACK(solver->arena));
    clause *c = (clause *)(arena + ref);
    assert(!c->garbage);
    base = c;
    for (all_literals_in_clause(other, c)) {
      if (other == not_lit) {
        continue;
      }
      const value value = values[other];
      if (value > 0) {
        kissat_eliminate_clause(solver, false, c, INVALID_LIT);
        base = 0;
        break;
      }
      if (value < 0) {
        continue;
      }
      const unsigned not_other = NOT(other);
      signed char mark = marks[not_other];
      if (mark) {
        continue;
      }
      base = 0;
      break;
    }
    if (base) {
      break;
    }
  }
  if (!base) {
    kissat_unmark_binaries(solver, lit);
    return false;
  }
  LOGCLS(base, "found and gate %s base clause", LOGLIT(not_lit));
  for (all_literals_in_clause(other, base)) {
    if (other == not_lit) {
      continue;
    }
    if (values[other]) {
      continue;
    }
    const unsigned not_other = NOT(other);
    assert(marks[not_other]);
    marks[not_other] = 0;
  }
  watch tmp = kissat_binary_watch(0, false, false);
  watches *watches = &WATCHES(lit);
  for (all_binary_large_watches(watch, *watches)) {
    if (!watch.type.binary) {
      continue;
    }
    const unsigned other = watch.binary.lit;
    assert(!solver->values[other]);
    if (marks[other]) {
      marks[other] = 0;
      continue;
    }
    tmp.binary.lit = other;
    PUSH_STACK(solver->gates[negative], tmp);
  }
  tmp = kissat_large_watch(kissat_reference_clause(solver, base));
  PUSH_STACK(solver->gates[!negative], tmp);
  solver->gate_eliminated = GATE_ELIMINATED(ands);
  INC(ands_extracted);
  return true;
}

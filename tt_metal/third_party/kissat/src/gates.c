#include "ands.h"
#include "definition.h"
#include "eliminate.h"
#include "equivalences.h"
#include "gates.h"
#include "ifthenelse.h"
#include "inline.h"
#include "xors.h"

size_t kissat_mark_binaries(kissat *solver, unsigned lit) {
  value *marks = solver->marks;
  size_t res = 0;
  watches *watches = &WATCHES(lit);
  for (all_binary_large_watches(watch, *watches)) {
    if (!watch.type.binary) {
      continue;
    }
    const unsigned other = watch.binary.lit;
    assert(!solver->values[other]);
    if (marks[other]) {
      continue;
    }
    marks[other] = 1;
    res++;
  }
  return res;
}

void kissat_unmark_binaries(kissat *solver, unsigned lit) {
  value *marks = solver->marks;
  watches *watches = &WATCHES(lit);
  for (all_binary_large_watches(watch, *watches))
    if (watch.type.binary) {
      marks[watch.binary.lit] = 0;
    }
}

bool kissat_find_gates(kissat *solver, unsigned lit, bool eq_only) {
  solver->gate_eliminated = 0;
  solver->resolve_gate = false;
  if (!GET_OPTION(extract)) {
    return false;
  }
  INC(gates_checked);
  const unsigned not_lit = NOT(lit);
  if (EMPTY_WATCHES(WATCHES(not_lit))) {
    return false;
  }
  bool res = false;
  if (kissat_find_equivalence_gate(solver, lit)) {
    res = true;
  } else if (!eq_only) {
    if (kissat_find_and_gate(solver, lit, 0)) {
      res = true;
    } else if (kissat_find_and_gate(solver, not_lit, 1)) {
      res = true;
    } else if (kissat_find_if_then_else_gate(solver, lit, 0)) {
      res = true;
    } else if (kissat_find_if_then_else_gate(solver, not_lit, 1)) {
      res = true;
    } else if (kissat_find_xor_gate(solver, lit, 0)) {
      res = true;
    } else if (kissat_find_xor_gate(solver, not_lit, 1)) {
      res = true;
    } else if (kissat_find_definition(solver, lit)) {
      res = true;
    }
  }
  if (res) {
    INC(gates_extracted);
  }
  return res;
}

static void get_antecedents(kissat *solver, unsigned lit, unsigned negative) {
  assert(!solver->watching);
  assert(!negative || negative == 1);

  statches *gates = solver->gates + negative;
  watches *watches = &WATCHES(lit);

  statches *antecedents = solver->antecedents + negative;
  assert(EMPTY_STACK(*antecedents));

  const watch *const begin_gates = BEGIN_STACK(*gates);
  const watch *const end_gates = END_STACK(*gates);
  watch const *g = begin_gates;

  const watch *const begin_watches = BEGIN_WATCHES(*watches);
  const watch *const end_watches = END_WATCHES(*watches);
  watch const *w = begin_watches;

  while (w != end_watches) {
    const watch watch = *w++;
    if (g != end_gates && g->raw == watch.raw) {
      g++;
    } else {
      PUSH_STACK(*antecedents, watch);
    }
  }

  assert(g == end_gates);
#ifdef LOGGING
  size_t size_gates = SIZE_STACK(*gates);
  size_t size_antecedents = SIZE_STACK(*antecedents);
  size_t size_watches = SIZE_WATCHES(*watches);
  LOG("got %zu antecedent %.0f%% and %zu gate clauses %.0f%% "
        "out of %zu watches of literal %s",
        size_antecedents, kissat_percent(size_antecedents, size_watches),
        size_gates, kissat_percent(size_gates, size_watches),
        size_watches, LOGLIT(lit));
#endif
}

void kissat_get_antecedents(kissat *solver, unsigned lit) {
  get_antecedents(solver, lit, 0);
  get_antecedents(solver, NOT(lit), 1);
}

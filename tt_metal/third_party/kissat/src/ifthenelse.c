#include "eliminate.h"
#include "gates.h"
#include "ifthenelse.h"
#include "inline.h"

static bool get_ternary_clause(kissat *solver, reference ref,
      unsigned *p, unsigned *q, unsigned *r) {
  clause *clause = kissat_dereference_clause(solver, ref);
  if (clause->garbage) {
    return false;
  }
  const value *const values = solver->values;
  unsigned a = INVALID_LIT, b = INVALID_LIT, c = INVALID_LIT;
  unsigned found = 0;
  for (all_literals_in_clause(other, clause)) {
    const value value = values[other];
    if (value > 0) {
      kissat_eliminate_clause(solver, false, clause, INVALID_LIT);
      return false;
    }
    if (value < 0) {
      continue;
    }
    if (++found == 1) {
      a = other;
    } else if (found == 2) {
      b = other;
    } else if (found == 3) {
      c = other;
    } else {
      return false;
    }
  }
  if (found != 3) {
    return false;
  }
  assert(a != INVALID_LIT);
  assert(b != INVALID_LIT);
  assert(c != INVALID_LIT);
  *p = a;
  *q = b;
  *r = c;
  return true;
}

static bool match_ternary_ref(kissat *solver, reference ref,
      unsigned a, unsigned b, unsigned c) {
  clause *clause = kissat_dereference_clause(solver, ref);
  if (clause->garbage) {
    return false;
  }
  const value *const values = solver->values;
  unsigned found = 0;
  for (all_literals_in_clause(other, clause)) {
    const value value = values[other];
    if (value > 0) {
      kissat_eliminate_clause(solver, false, clause, INVALID_LIT);
      return false;
    }
    if (value < 0) {
      continue;
    }
    if (a != other && b != other && c != other) {
      return false;
    }
    found++;
  }
  if (found == 3) {
    return true;
  }
  solver->resolve_gate = true;
  return true;
}

static bool match_ternary_watch(kissat *solver, watch watch,
      unsigned a, unsigned b, unsigned c) {
  if (watch.type.binary) {
    const unsigned other = watch.binary.lit;
    if (other != b && other != c) {
      return false;
    }
    solver->resolve_gate = true;
    return true;
  } else {
    const reference ref = watch.large.ref;
    return match_ternary_ref(solver, ref, a, b, c);
  }
}

static inline const watch *find_ternary_clause(kissat *solver, uint64_t *steps,
      unsigned a, unsigned b, unsigned c) {
  watches *watches = &WATCHES(a);
  const watch *const begin = BEGIN_WATCHES(*watches);
  const watch *const end = END_WATCHES(*watches);
  for (const watch *p = begin; p != end; p++) {
    *steps += 1;
    if (match_ternary_watch(solver, *p, a, b, c)) {
      return p;
    }
  }
  return 0;
}

bool kissat_find_if_then_else_gate(kissat *solver,
      unsigned lit, unsigned negative) {
  if (!GET_OPTION(ifthenelse)) {
    return false;
  }
  watches *watches = &WATCHES(lit);
  const watch *const begin = BEGIN_WATCHES(*watches);
  const watch *const end = END_WATCHES(*watches);
  if (begin == end) {
    return false;
  }
  uint64_t large_clauses = 0;
  for (const watch *p = begin; p != end; p++)
    if (!p->type.binary) {
      large_clauses++;
    }
  const uint64_t limit = solver->bounds.eliminate.occurrences;
  if (large_clauses * large_clauses > limit) {
    return false;
  }
  const watch *const last = end - 1;
  uint64_t steps = 0;
  for (const watch *p1 = begin; steps < limit && p1 != last; p1++) {
    watch w1 = *p1;
    if (w1.type.binary) {
      continue;
    }
    unsigned a1, b1, c1;
    if (!get_ternary_clause(solver, p1->large.ref, &a1, &b1, &c1)) {
      continue;
    }
    if (b1 == lit) {
      SWAP(unsigned, a1, b1);
    }
    if (c1 == lit) {
      SWAP(unsigned, a1, c1);
    }
    assert(a1 == lit);
    for (const watch *p2 = p1 + 1; steps < limit && p2 != end; p2++) {
      watch w2 = *p2;
      if (w2.type.binary) {
        continue;
      }
      unsigned a2, b2, c2;
      if (!get_ternary_clause(solver, p2->large.ref, &a2, &b2, &c2)) {
        continue;
      }
      if (b2 == lit) {
        SWAP(unsigned, a2, b2);
      }
      if (c2 == lit) {
        SWAP(unsigned, a2, c2);
      }
      assert(a2 == lit);
      if (STRIP(b1) == STRIP(c2)) {
        SWAP(unsigned, b2, c2);
      }
      if (STRIP(c1) == STRIP(c2)) {
        continue;
      }
      const unsigned not_b2 = NOT(b2);
      if (b1 != not_b2) {
        continue;
      }
      solver->resolve_gate = false;
      const unsigned not_lit = NOT(lit);
      const unsigned not_c1 = NOT(c1);
      const watch *const p3 =
            find_ternary_clause(solver, &steps, not_lit, b1, not_c1);
      if (!p3) {
        continue;
      }
      const unsigned not_c2 = NOT(c2);
      const watch *const p4 =
            find_ternary_clause(solver, &steps, not_lit, b2, not_c2);
      if (!p4) {
        continue;
      }
      watch w3 = p3 < p4 ? *p3 : *p4;
      watch w4 = p3 < p4 ? *p4 : *p3;
      LOGWATCH(lit, w1, "1st if-then-else");
      LOGWATCH(lit, w2, "2nd if-then-else");
      LOGWATCH(not_lit, w3, "3rd if-then-else");
      LOGWATCH(not_lit, w4, "4th if-then-else");
      LOG("found if-then-else gate %s = (%s ? %s : %s)",
            LOGLIT(lit), LOGLIT(NOT(b1)),
            LOGLIT(not_c1), LOGLIT(not_c2));
      solver->gate_eliminated = GATE_ELIMINATED(if_then_else);
      PUSH_STACK(solver->gates[negative], w1);
      PUSH_STACK(solver->gates[negative], w2);
      PUSH_STACK(solver->gates[!negative], w3);
      PUSH_STACK(solver->gates[!negative], w4);
      INC(if_then_else_extracted);
      return true;
    }
  }
  return false;
}

#include "eliminate.h"
#include "gates.h"
#include "inline.h"
#include "sort.h"
#include "terminate.h"
#include "xors.h"

static void mark_literals(kissat *solver, value *marks,
      unsigned size, unsigned *lits, unsigned signs) {
  assert(size < 32);
  for (unsigned i = 0, bit = 1; i < size; i++, bit <<= 1) {
    unsigned sign = ((bit & signs) != 0);
    const unsigned lit = lits[i] ^ sign;
    assert(!marks[lit]);
    marks[lit] = 1;
    LOG("marked %s", LOGLIT(lit));
  }
#ifndef LOGGING
  (void) solver;
#endif
}

static void unmark_literals(kissat *solver, value *marks,
      unsigned size, unsigned *lits, unsigned signs) {
  assert(size < 32);
  for (unsigned i = 0, bit = 1; i < size; i++, bit <<= 1) {
    unsigned sign = ((bit & signs) != 0);
    const unsigned lit = lits[i] ^ sign;
    assert(marks[lit]);
    marks[lit] = 0;
    LOG("unmarked %s", LOGLIT(lit));
  }
#ifndef LOGGING
  (void) solver;
#endif
}

static unsigned copy_literals(kissat *solver, unsigned lit,
      const value *const values, unsigned *lits, clause *c) {
  assert(!c->garbage);
  const unsigned *const end = c->lits + c->size;
  unsigned *q = lits;
#ifndef NDEBUG
  bool found_lit = false;
#endif
  for (const unsigned *p = c->lits; p != end; p++) {
    const unsigned other = *p;
    if (other == lit) {
#ifndef NDEBUG
      assert(!found_lit);
      assert(!values[other]);
      found_lit = true;
#endif
    } else {
      const value value = values[other];
      if (value > 0) {
        kissat_eliminate_clause(solver, false, c, other);
        LOG("found satisfied %s", LOGLIT(other));
        return UINT_MAX;
      }
      if (value < 0) {
        LOG("skipping falsified %s", LOGLIT(other));
        continue;
      }
      LOG("copying %s", LOGLIT(other));
      *q++ = other;
    }
  }
  assert(found_lit);
  *q++ = lit;
  const unsigned size = q - lits;
  LOGLITS(size, lits, "copied %u", size);
#ifndef LOGGING
  (void) solver;
#endif
  return size;
}

static bool odd_parity(unsigned size, unsigned signs) {
  bool res = false;
  for (unsigned i = 0; i < size; i++)
    if (signs & (1u << i)) {
      res = !res;
    }
  return res;
}

static unsigned next_marking(kissat *solver, value *marks,
      unsigned size, unsigned *lits, unsigned prev) {
  LOG("prev signs %s", FORMAT_SIGNS(size, prev));
  assert(2 < size);
  assert(size < 32);
  const unsigned limit = (1u << size);

  assert(prev < limit);

  unsigned next;

  for (next = prev + 1; odd_parity(size, next); next++)
    ;

  LOG("next signs %s", FORMAT_SIGNS(size, next));

  for (unsigned i = 0, bit = 1; i < size; i++, bit <<= 1) {
    const unsigned lit = lits[i];
    const unsigned not_lit = NOT(lit);
    if (!(prev & bit) && (next & bit)) {
      assert(marks[lit]);
      assert(!marks[not_lit]);
      marks[lit] = 0;
      LOG("unmarked %s", LOGLIT(lit));
      marks[not_lit] = 1;
      LOG("marked %s", LOGLIT(not_lit));
    } else if ((prev & bit) && !(next & bit)) {
      assert(marks[not_lit]);
      assert(!marks[lit]);
      marks[not_lit] = 0;
      LOG("unmarked %s", LOGLIT(not_lit));
      marks[lit] = 1;
      LOG("marked %s", LOGLIT(lit));
    }
  }
#ifndef LOGGING
  (void) solver;
#endif
  return next == limit ? 0 : next;
}

static bool match_lits_ref(kissat *solver, const value *marks,
      const value *values,
      unsigned size, reference ref) {
  clause *c = kissat_dereference_clause(solver, ref);
  if (c->garbage) {
    return false;
  }
  unsigned found = 0;
  for (all_literals_in_clause(lit, c)) {
    const value value = values[lit];
    if (value > 0) {
      kissat_eliminate_clause(solver, false, c, INVALID_LIT);
      return false;
    }
    if (value < 0) {
      continue;
    }
    if (!marks[lit]) {
      return false;
    }
    found++;
  }
  assert(found <= size);
  if (found < size) {
    solver->resolve_gate = true;
  }
  return true;
}

static bool match_lits_watch(kissat *solver,
      const value *const marks, const value *values,
      unsigned size, watch watch) {
  if (watch.type.binary) {
    const unsigned other = watch.binary.lit;
    if (!marks[other]) {
      return false;
    }
    assert(size > 2);
    solver->resolve_gate = true;
    return true;
  } else {
    const reference ref = watch.large.ref;
    return match_lits_ref(solver, marks, values, size, ref);
  }
}

static inline watch *find_lits_watch(kissat *solver, watch *begin, watch *end,
      const value *const marks, const value *values,
      unsigned size, uint64_t *steps) {
  assert(begin <= end);
  for (watch *p = begin; p != end; p++) {
    if (match_lits_watch(solver, marks, values, size, *p)) {
      *steps += p - begin + 1;
      return p;
    }
  }
  *steps += end - begin;
  return 0;
}

#define LESS_POINTER(P,Q) \
  ((P) < (Q))

static void sort_watch_pointers(kissat *solver, patches *patches) {
  SORT_STACK(watch *, *patches, LESS_POINTER);
}

bool kissat_find_xor_gate(kissat *solver, unsigned lit, unsigned negative) {
  if (!GET_OPTION(xors)) {
    return false;
  }

  const unsigned size_limit = solver->bounds.xor.clause_size;
  if (size_limit < 3) {
    return false;
  }
  assert(size_limit < 32);

  watches *watches0 = &WATCHES(lit);
  watch *begin0 = BEGIN_WATCHES(*watches0);
  watch *end0 = END_WATCHES(*watches0);
  if (begin0 == end0) {
    return false;
  }

  uint64_t large_clauses0 = 0;
  for (watch *p = begin0; p != end0; p++)
    if (!p->type.binary && large_clauses0++) {
      break;
    }
  if (large_clauses0 < 2) {
    return false;
  }

  const unsigned not_lit = NOT(lit);
  watches *watches1 = &WATCHES(not_lit);
  watch *begin1 = BEGIN_WATCHES(*watches1);
  watch *end1 = END_WATCHES(*watches1);

  uint64_t large_clauses1 = 0;
  for (watch *p = begin1; p != end1; p++)
    if (!p->type.binary && large_clauses1++) {
      break;
    }
  if (large_clauses1 < 2) {
    return false;
  }

  unsigned lits[size_limit];

  const value *const values = solver->values;
  value *marks = solver->marks;

  const unsigned steps_limit = solver->bounds.eliminate.occurrences;

  uint64_t steps = 0;

  for (watch *p = begin0; p != end0; p++) {
    if (p->type.binary) {
      continue;
    }

    if (steps > steps_limit) {
      break;
    }

    if (TERMINATED(xors_terminated_1)) {
      break;
    }

    steps++;
    clause *c = kissat_dereference_clause(solver, p->large.ref);
    if (c->garbage) {
      continue;
    }
    if (c->size > size_limit) {
      continue;
    }

    unsigned size = copy_literals(solver, lit, values, lits, c);
    if (size == UINT_MAX) {
      continue;
    }

    assert(size <= 32);
    if (size < 3) {
      continue;
    }

    solver->resolve_gate = false;

    assert(EMPTY_STACK(solver->xorted[0]));
    assert(EMPTY_STACK(solver->xorted[1]));
    PUSH_STACK(solver->xorted[0], p);

    unsigned signs = 0;
    mark_literals(solver, marks, size, lits, signs);

    while (steps <= steps_limit &&
          (signs = next_marking(solver, marks, size, lits, signs))) {
      if (marks[lit]) {
        watch *q = find_lits_watch(solver, begin0, end0,
                    marks, values, size, &steps);
        if (!q) {
          LOGLITS(size, lits,
                "could not match signs %s of copied",
                FORMAT_SIGNS(size, signs));
          break;
        }

        LOGWATCH(lit, *q, "literal %s XOR", LOGLIT(lit));
        PUSH_STACK(solver->xorted[0], q);
      } else {
        watch *q = find_lits_watch(solver, begin1, end1,
                    marks, values, size, &steps);
        if (!q) {
          break;
        }

        LOGWATCH(not_lit, *q, "found %s literal XOR",
              LOGLIT(not_lit));
        PUSH_STACK(solver->xorted[1], q);
      }
    }

    unmark_literals(solver, marks, size, lits, signs);

    unsigned nsort[2] = {
      SIZE_STACK(solver->xorted[0]), SIZE_STACK(solver->xorted[1])
    };

    if (nsort[0] + nsort[1] == (1u << (size - 1))) {
      assert(nsort[0] == (1u << (size - 2)));
      assert(nsort[1] == (1u << (size - 2)));

      sort_watch_pointers(solver, &solver->xorted[0]);

      watch const *prev = 0;
      for (unsigned i = 0; i < nsort[0]; i++) {
        const watch *const p0 = PEEK_STACK(solver->xorted[0], i);
        const watch w0 = *p0;
        if (p0 == prev) {
          LOGWATCH(lit, w0, "dropping repeated");
        } else {
          LOGWATCH(lit, w0, "%s %s XOR",
                FORMAT_ORDINAL(i + 1), LOGLIT(lit));
          PUSH_STACK(solver->gates[negative], w0);
        }
        prev = p0;
      }

      sort_watch_pointers(solver, &solver->xorted[1]);
      prev = 0;
      for (unsigned i = 0; i < nsort[1]; i++) {
        const watch *const p1 = PEEK_STACK(solver->xorted[1], i);
        const watch w1 = *p1;
        if (p1 == prev) {
          LOGWATCH(not_lit, w1, "dropping repeated");
        } else {
          LOGWATCH(not_lit, w1, "%s %s XOR",
                FORMAT_ORDINAL(i + 1), LOGLIT(not_lit));
          PUSH_STACK(solver->gates[!negative], w1);
        }
        prev = p1;
      }

      assert(!EMPTY_STACK(solver->gates[0]));
#ifdef LOGGING
      assert(size > 1);
      assert(lits[size - 1] == lit);
      lits[0] = NOT(lits[0]);
      LOGXOR(lit, size - 1, lits, "found");
      lits[0] = NOT(lits[0]);
#endif
    }

    CLEAR_STACK(solver->xorted[0]);
    CLEAR_STACK(solver->xorted[1]);

    if (EMPTY_STACK(solver->gates[0])) {
      continue;
    }

    solver->gate_eliminated = GATE_ELIMINATED(xors);
    INC(xors_extracted);

    return true;
  }

  assert(EMPTY_STACK(solver->xorted[0]));
  assert(EMPTY_STACK(solver->xorted[1]));

  return false;
}

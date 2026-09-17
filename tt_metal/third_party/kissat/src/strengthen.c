#include "inline.h"
#include "promote.h"
#include "strengthen.h"

static clause *large_on_the_fly_strengthen(kissat *solver, clause *c,
      unsigned lit) {
  assert(solver->antecedent_size > 3);
  LOGCLS(c, "large on-the-fly strengthening "
        "by removing %s from", LOGLIT(lit));
  unsigned *lits = c->lits;
  assert(lits[0] == lit || lits[1] == lit);
  INC(on_the_fly_strengthened);
#ifndef NDEBUG
  clause *old_next = kissat_next_clause(c);
#endif
  if (lits[0] == lit) {
    SWAP(unsigned, lits[0], lits[1]);
  }
  assert(lits[1] == lit);
  const reference ref = kissat_reference_clause(solver, c);
  kissat_unwatch_blocking(solver, lit, ref);
  SHRINK_CLAUSE_IN_PROOF(c, lit, lits[0]);
  CHECK_SHRINK_CLAUSE(c, lit, lits[0]);
  {
    const unsigned old_size = c->size;
    unsigned new_size = 1;
    const bool irredundant = !c->redundant;
    for (unsigned i = 2; i < old_size; i++) {
      const unsigned other = lits[i];
      assert(VALUE(other) < 0);
      if (!LEVEL(other)) {
        continue;
      }
      lits[new_size++] = other;
      if (irredundant) {
        kissat_mark_added_literal(solver, other);
      }
    }
    assert(new_size > 2);
    c->size = new_size;
    c->searched = 2;
    if (c->redundant && c->glue >= new_size) {
      kissat_promote_clause(solver, c, new_size - 1);
    }
    if (!c->shrunken) {
      c->shrunken = true;
      lits[old_size - 1] = INVALID_LIT;
    }
  }
  LOGCLS(c, "unsorted on-the-fly strengthened");
  {
    assert(VALUE(lits[1]) < 0);
    unsigned highest_pos = 1;
    unsigned highest_level = LEVEL(lits[1]);
    const unsigned size = c->size;
    for (unsigned i = 2; i < size; i++) {
      const unsigned other = lits[i];
      assert(VALUE(other) < 0);
      const unsigned level = LEVEL(other);
      if (level <= highest_level) {
        continue;
      }
      highest_pos = i;
      highest_level = level;
    }
    if (highest_pos != 1) {
      SWAP(unsigned, lits[1], lits[highest_pos]);
    }
    LOGCLS(c, "sorted on-the-fly strengthened");
    kissat_watch_blocking(solver, lits[1], lits[0], ref);
  }
  {
    watches *watches = &WATCHES(lits[0]);
#ifndef NDEBUG
    const watch *const end_of_watches = END_WATCHES(*watches);
#endif
    watch *p = BEGIN_WATCHES(*watches);
    assert(solver->watching);
    for (;;) {
      assert(p != end_of_watches);
      const watch head = *p++;
      if (head.type.binary) {
        continue;
      }
      assert(p != end_of_watches);
      const watch tail = *p++;
      if (tail.large.ref == ref) {
        break;
      }
    }
    p[-2].blocking.lit = lits[1];
    LOGREF(ref, "updating watching %s now blocking %s in",
          LOGLIT(lits[0]), LOGLIT(lits[1]));
  }
#ifndef NDEBUG
  clause *new_next = kissat_next_clause(c);
  assert(old_next == new_next);
#endif
  LOGCLS(c, "conflicting");
  INC(conflicts);
  return c;
}

static clause *binary_on_the_fly_strengthen(kissat *solver, clause *c,
      unsigned lit) {
  assert(solver->antecedent_size == 3);
  LOGCLS(c, "binary on-the-fly strengthening "
        "by removing %s from", LOGLIT(lit));
  unsigned first = INVALID_LIT, second = INVALID_LIT;
#ifndef NDEBUG
  bool found = false;
#endif
  for (all_literals_in_clause(other, c)) {
    if (other == lit) {
#ifndef NDEBUG
      assert(!found);
      found = true;
#endif
      continue;
    }
    assert(VALUE(other) < 0);
    if (!LEVEL(other)) {
      continue;
    }
    if (first == INVALID_LIT) {
      first = other;
    } else {
      assert(second == INVALID_LIT);
      second = other;
#ifndef NDEBUG
      break;
#endif
    }
  }
  assert(found);
  assert(second != INVALID_LIT);
  const bool redundant = c->redundant;
  LOGBINARY(first, second, "on-the-fly strengthened");
  kissat_new_binary_clause(solver, redundant, first, second);
  const reference ref = kissat_reference_clause(solver, c);
  kissat_unwatch_blocking(solver, c->lits[0], ref);
  kissat_unwatch_blocking(solver, c->lits[1], ref);
  kissat_mark_clause_as_garbage(solver, false, c);
  clause *conflict =
        kissat_binary_conflict(solver, redundant, first, second);
  INC(conflicts);
  return conflict;
}

clause *kissat_on_the_fly_strengthen(kissat *solver, clause *c, unsigned lit) {
  assert(!c->garbage);
  assert(solver->antecedent_size > 2);
  if (!c->redundant) {
    kissat_mark_removed_literal(solver, lit);
  }
  clause *res;
  if (solver->antecedent_size == 3) {
    res = binary_on_the_fly_strengthen(solver, c, lit);
  } else {
    res = large_on_the_fly_strengthen(solver, c, lit);
  }
  return res;
}

void kissat_on_the_fly_subsume(kissat *solver, clause *c, clause *d) {
  LOGCLS(c, "on-the-fly subsuming");
  LOGCLS(d, "on-the-fly subsumed");
  assert(c != d);
  assert(!d->garbage);
  assert(c->size > 1);
  assert(c->size <= d->size);
  kissat_mark_clause_as_garbage(solver, false, d);
  INC(on_the_fly_subsumed);
  if (d->redundant) {
    return;
  }
  if (!c->redundant) {
    return;
  }
  if (c->size == 2) {
    assert(c == &solver->conflict);
    const unsigned *const lits = c->lits;
    LOGBINARY(lits[0], lits[1], "turned irredundant");
    for (unsigned i = 0; i < 2; i++) {
      const unsigned lit = lits[i];
      watches *watches = &WATCHES(lit);
      watch *p = LAST_WATCH_POINTER(*watches);
      assert(p->type.binary);
      assert(p->binary.redundant);
      assert(p->binary.lit == lits[!i]);
      p->binary.redundant = false;
      assert(!p->binary.hyper);
    }
  } else {
    c->redundant = false;
    LOGCLS(c, "turned");
    clause *last_irredundant = kissat_last_irredundant_clause(solver);
    if (!last_irredundant || last_irredundant < c) {
      LOGCLS(c, "updating last irredundant clause as");
      reference ref = kissat_reference_clause(solver, c);
      solver->last_irredundant = ref;
    }
  }
  statistics *statistics = &solver->statistics;
  assert(statistics->clauses_irredundant < UINT64_MAX);
  statistics->clauses_irredundant++;
  assert(statistics->clauses_redundant > 0);
  statistics->clauses_redundant--;
}

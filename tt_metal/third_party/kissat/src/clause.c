#include "allocate.h"
#include "collect.h"
#include "inline.h"

#include <string.h>

static void inc_clause(kissat *solver, bool original, bool redundant) {
  if (redundant) {
    INC(clauses_redundant);
  } else {
    INC(clauses_irredundant);
  }
  INC(clauses_added);
  if (original) {
    INC(clauses_original);
  }
}

static void dec_clause(kissat *solver, bool redundant) {
  if (redundant) {
    DEC(clauses_redundant);
  } else {
    DEC(clauses_irredundant);
  }
}

static void init_clause(kissat *solver, clause *res,
      bool redundant, unsigned glue, unsigned size) {
  assert(size <= UINT_MAX);
  assert(redundant || !glue);

  glue = MIN(MAX_GLUE, glue);

  const unsigned tier1 = GET_OPTION(tier1);
  const bool keep = (glue <= tier1);

  res->glue = glue;

  res->garbage = false;
  res->hyper = false;
  res->keep = keep;
  res->reason = false;
  res->redundant = redundant;
  res->shrunken = false;
  res->subsume = false;
  res->sweeped = false;
  res->vivify = false;

  res->used = 0;

  res->searched = 2;
  res->size = size;
#ifdef NOPTIONS
  (void) solver;
#endif
}

void kissat_connect_clause(kissat *solver, clause *c) {
  watches *all_watches = solver->watches;
  const reference ref = kissat_reference_clause(solver, c);
  kissat_inlined_connect_clause(solver, all_watches, c, ref);
}

static reference new_binary_clause(kissat *solver,
      bool original, bool redundant,
      unsigned first, unsigned second) {
  assert(first != second);
  assert(first != NOT(second));
  assert(!original || !redundant);
  kissat_watch_binary(solver, redundant, false, first, second);
  if (!redundant) {
    kissat_mark_added_literal(solver, first);
    kissat_mark_added_literal(solver, second);
  }
  inc_clause(solver, original, redundant);
  if (!original) {
    CHECK_AND_ADD_BINARY(first, second);
    ADD_BINARY_TO_PROOF(first, second);
  }
  return INVALID_REF;
}

static reference new_large_clause(kissat *solver,
      bool original, bool redundant, unsigned glue,
      unsigned size, unsigned *lits) {
  assert(size > 2);
  reference res = kissat_allocate_clause(solver, size);
  clause *c = kissat_unchecked_dereference_clause(solver, res);
  init_clause(solver, c, redundant, glue, size);
  memcpy(c->lits, lits, size * sizeof(unsigned));
  LOGREF(res, "new");
  if (solver->watching) {
    kissat_watch_reference(solver, lits[0], lits[1], res);
  } else {
    kissat_connect_clause(solver, c);
  }
  if (redundant) {
    if (!c->keep && solver->first_reducible == INVALID_REF) {
      solver->first_reducible = res;
    }
  } else {
    kissat_mark_added_literals(solver, size, lits);
    solver->last_irredundant = res;
  }
  inc_clause(solver, original, redundant);
  if (!original) {
    CHECK_AND_ADD_CLAUSE(c);
    ADD_CLAUSE_TO_PROOF(c);
  }
  return res;
}

static reference new_clause(kissat *solver,
      bool original, bool redundant,
      unsigned glue, unsigned size, unsigned *lits) {
  reference res;
  if (size == 2) {
    res = new_binary_clause(solver, original, redundant, lits[0], lits[1]);
  } else {
    res = new_large_clause(solver, original, redundant, glue, size, lits);
  }
  kissat_defrag_watches_if_needed(solver);
  return res;
}

void kissat_new_binary_clause(kissat *solver,
      bool redundant, unsigned first, unsigned second) {
  (void) new_binary_clause(solver, false, redundant, first, second);
}

reference kissat_new_original_clause(kissat *solver) {
  const unsigned size = SIZE_STACK(solver->clause);
  unsigned *lits = BEGIN_STACK(solver->clause);
  kissat_sort_literals(solver, size, lits);
  reference res = new_clause(solver, true, false, 0, size, lits);
  return res;
}

reference kissat_new_irredundant_clause(kissat *solver) {
  const unsigned size = SIZE_STACK(solver->clause);
  unsigned *lits = BEGIN_STACK(solver->clause);
  return new_clause(solver, false, false, 0, size, lits);
}

reference kissat_new_redundant_clause(kissat *solver, unsigned glue) {
  const unsigned size = SIZE_STACK(solver->clause);
  unsigned *lits = BEGIN_STACK(solver->clause);
  return new_clause(solver, false, true, glue, size, lits);
}

static void mark_clause_as_garbage(kissat *solver, bool weakened, clause *c) {
  assert(!c->garbage);
  LOGCLS(c, "garbage");
  if (!c->redundant) {
    kissat_mark_removed_literals(solver, c->size, c->lits);
  }
  if (!weakened || !GET_OPTION(incremental)) {
    REMOVE_CHECKER_CLAUSE(c);
    DELETE_CLAUSE_FROM_PROOF(c);
  }
  if (c->hyper) {
    assert(c->size == 3);
    assert(c->redundant);
    DEC(hyper_ternaries);
  }
  dec_clause(solver, c->redundant);
  c->garbage = true;
}

void kissat_mark_clause_as_garbage(kissat *solver, bool weakened, clause *c) {
  assert(!c->garbage);
  mark_clause_as_garbage(solver, weakened, c);
  size_t bytes = kissat_actual_bytes_of_clause(c);
  ADD(arena_garbage, bytes);
}

clause *kissat_delete_clause(kissat *solver, clause *c) {
  LOGCLS(c, "delete");
  assert(c->size > 2);
  assert(c->garbage);
  size_t bytes = kissat_actual_bytes_of_clause(c);
  SUB(arena_garbage, bytes);
  INC(clauses_deleted);
  return (clause *)((char *) c + bytes);
}

void kissat_delete_binary(kissat *solver,
      bool redundant, bool hyper, bool weakened, unsigned a, unsigned b) {
  LOGBINARY(a, b, "delete");
  if (!redundant) {
    kissat_mark_removed_literal(solver, a);
    kissat_mark_removed_literal(solver, b);
  } else if (hyper) {
    DEC(hyper_binaries);
  }
  if (!weakened || !GET_OPTION(incremental)) {
    REMOVE_CHECKER_BINARY(a, b);
    DELETE_BINARY_FROM_PROOF(a, b);
  }
  dec_clause(solver, redundant);
  INC(clauses_deleted);
}

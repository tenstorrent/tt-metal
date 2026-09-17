#include "allocate.h"
#include "clueue.h"
#include "internal.h"
#include "logging.h"

#include <string.h>

void kissat_clear_clueue(kissat *solver, clueue *clueue) {
  if (!clueue->size) {
    return;
  }
  const size_t bytes = clueue->size * sizeof * clueue->elements;
  memset(clueue->elements, 0xff, bytes);
  LOG("cleared clueue of size %u", clueue->size);
#ifndef LOGGING
  (void) solver;
#endif
}

void kissat_release_clueue(kissat *solver, clueue *clueue) {
  if (!clueue->size) {
    return;
  }
  const size_t bytes = clueue->size * sizeof * clueue->elements;
  kissat_free(solver, clueue->elements, bytes);
}

void kissat_init_clueue(kissat *solver, clueue *clueue, unsigned size) {
  assert(size);
  assert(!clueue->size);
  clueue->size = size;
  clueue->elements = kissat_malloc(solver, size * sizeof * clueue->elements);
  assert(!clueue->next);
  LOG("initialized clueue of size %u", size);
  kissat_clear_clueue(solver, clueue);
}

#define all_clueue(REF, CLUEUE) \
  reference REF, * REF ## _PTR = (CLUEUE).elements,  \
                 * REF ## _END = REF ## _PTR + (CLUEUE).size; \
  REF ## _PTR != REF ## _END && (REF = *REF ## _PTR, true); \
  REF ## _PTR++

void kissat_eager_subsume(kissat *solver) {
  assert(!solver->probing);
  clueue *clueue = &solver->clueue;
  if (!clueue->size) {
    return;
  }
  LOGTMP("eagerly subsuming");
  value *marks = solver->marks;
  for (all_stack(unsigned, lit, solver->clause)) {
    assert(!marks[lit]);
    marks[lit] = 1;
  }
  const unsigned size = SIZE_STACK(solver->clause);
  assert(size > 1);
  const unsigned bound = size - 1;
  ward *arena = BEGIN_STACK(solver->arena);
  for (all_clueue(ref, solver->clueue)) {
    if (ref == INVALID_REF) {
      continue;
    }
    clause *c = (clause *)(arena + ref);
    assert(kissat_clause_in_arena(solver, c));
    if (c->garbage) {
      goto REMOVE;
    }
    if (!c->redundant) {
      goto REMOVE;
    }
    if (c->size + 1 < bound) {
      continue;
    }
    unsigned needed = size;
    for (all_literals_in_clause(lit, c))
      if (marks[lit] && !--needed) {
        break;
      }
    if (needed) {
      continue;
    }
    LOGCLS(c, "eagerly subsumed");
    INC(subsumed);
    INC(eager_subsumed);
    kissat_mark_clause_as_garbage(solver, false, c);
  REMOVE:
    *ref_PTR = INVALID_REF;
  }
  for (all_stack(unsigned, lit, solver->clause)) {
    assert(marks[lit]);
    marks[lit] = 0;
  }
}

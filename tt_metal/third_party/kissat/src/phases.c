#include "allocate.h"
#include "internal.h"
#include "logging.h"

#include <string.h>

#define realloc_phases(NAME) \
do { \
  solver->phases.NAME = \
    kissat_realloc (solver, solver->phases.NAME, old_size, new_size); \
} while (0)

#define increase_phases(NAME) \
do { \
  assert (old_size < new_size); \
  realloc_phases (NAME); \
  memset (solver->phases.NAME + old_size, 0, new_size - old_size); \
} while (0)

void kissat_increase_phases(kissat *solver, unsigned new_size) {
  const unsigned old_size = solver->size;
  assert(old_size < new_size);
  LOG("increasing phases from %u to %u", old_size, new_size);
  increase_phases(best);
  increase_phases(saved);
  increase_phases(target);
}

void kissat_decrease_phases(kissat *solver, unsigned new_size) {
  const unsigned old_size = solver->size;
  assert(old_size > new_size);
  LOG("decreasing phases from %u to %u", old_size, new_size);
  realloc_phases(best);
  realloc_phases(saved);
  realloc_phases(target);
}

#define release_phases(NAME, SIZE) \
  kissat_free (solver, solver->phases.NAME, SIZE)

void kissat_release_phases(kissat *solver) {
  const unsigned size = solver->size;
  release_phases(best, size);
  release_phases(saved, size);
  release_phases(target, size);
}

static void save_phases(kissat *solver, value *phases) {
  const value *const values = solver->values;
  const value *const end = phases + VARS;
  value const *v = values;
  for (value *p = phases, tmp; p != end; p++, v += 2)
    if ((tmp = *v)) {
      *p = tmp;
    }
  assert(v == values + LITS);
}

void kissat_save_best_phases(kissat *solver) {
  assert(sizeof(value) == 1);
  LOG("saving %u best values", VARS);
  save_phases(solver, solver->phases.best);
}

void kissat_save_saved_phases(kissat *solver) {
  assert(sizeof(value) == 1);
  LOG("saving %u saved values", VARS);
  save_phases(solver, solver->phases.saved);
}

void kissat_save_target_phases(kissat *solver) {
  assert(sizeof(value) == 1);
  LOG("saving %u target values", VARS);
  save_phases(solver, solver->phases.target);
}

#include "allocate.h"
#include "inline.h"
#include "require.h"
#include "resize.h"

#include <limits.h>
#include <string.h>

#define NREALLOC_GENERIC(TYPE, NAME, ELEMENTS_PER_BLOCK) \
do { \
  const size_t block_size = sizeof (TYPE); \
  solver->NAME = \
    kissat_nrealloc (solver, solver->NAME, old_size, new_size, \
                     ELEMENTS_PER_BLOCK * block_size); \
} while (0)

#define CREALLOC_GENERIC(TYPE, NAME, ELEMENTS_PER_BLOCK) \
do { \
  const size_t block_size = sizeof (TYPE); \
  TYPE *NAME = kissat_calloc (solver, \
                              ELEMENTS_PER_BLOCK * new_size, block_size); \
  if (old_size) \
    { \
      const size_t bytes = ELEMENTS_PER_BLOCK * old_size * block_size; \
      memcpy (NAME, solver->NAME, bytes); \
    } \
  kissat_dealloc (solver, solver->NAME, \
                  ELEMENTS_PER_BLOCK * old_size, block_size); \
  solver->NAME = NAME; \
} while (0)

#define NREALLOC_VARIABLE_INDEXED(TYPE, NAME) \
  NREALLOC_GENERIC (TYPE, NAME, 1)

#define NREALLOC_LITERAL_INDEXED(TYPE, NAME) \
  NREALLOC_GENERIC (TYPE, NAME, 2)

#define CREALLOC_VARIABLE_INDEXED(TYPE, NAME) \
  CREALLOC_GENERIC (TYPE, NAME, 1)

#define CREALLOC_LITERAL_INDEXED(TYPE, NAME) \
  CREALLOC_GENERIC (TYPE, NAME, 2)

static void reallocate_trail(kissat *solver, unsigned old_size,
      unsigned new_size) {
  unsigned propagated = solver->propagate - BEGIN_ARRAY(solver->trail);
  REALLOCATE_ARRAY(solver->trail, old_size, new_size);
  solver->propagate = BEGIN_ARRAY(solver->trail) + propagated;
}

void kissat_increase_size(kissat *solver, unsigned new_size) {
  assert(solver->vars <= new_size);
  const unsigned old_size = solver->size;
  if (old_size >= new_size) {
    return;
  }

#ifdef METRICS
  LOG("%s before increasing size from %u to %u",
        FORMAT_BYTES(kissat_allocated(solver)), old_size, new_size);
#endif
  CREALLOC_VARIABLE_INDEXED(assigned, assigned);
  CREALLOC_VARIABLE_INDEXED(flags, flags);
  CREALLOC_VARIABLE_INDEXED(unsigned, protect);
  NREALLOC_VARIABLE_INDEXED(links, links);

  CREALLOC_LITERAL_INDEXED(mark, marks);
  CREALLOC_LITERAL_INDEXED(value, values);
  CREALLOC_LITERAL_INDEXED(watches, watches);

  reallocate_trail(solver, old_size, new_size);

  kissat_resize_heap(solver, &solver->scores, new_size);
  kissat_increase_phases(solver, new_size);

  solver->size = new_size;

#ifdef METRICS
  LOG("%s after increasing size from %u to %u",
        FORMAT_BYTES(kissat_allocated(solver)), old_size, new_size);
#endif
}

void kissat_decrease_size(kissat *solver) {
  const unsigned old_size = solver->size;
  const unsigned new_size = solver->vars;

#ifdef METRICS
  LOG("%s before decreasing size from %u to %u",
        FORMAT_BYTES(kissat_allocated(solver)), old_size, new_size);
#endif

  NREALLOC_VARIABLE_INDEXED(assigned, assigned);
  NREALLOC_VARIABLE_INDEXED(flags, flags);
  NREALLOC_VARIABLE_INDEXED(unsigned, protect);
  NREALLOC_VARIABLE_INDEXED(links, links);

  NREALLOC_LITERAL_INDEXED(mark, marks);
  NREALLOC_LITERAL_INDEXED(value, values);
  NREALLOC_LITERAL_INDEXED(watches, watches);

  reallocate_trail(solver, old_size, new_size);

  kissat_resize_heap(solver, &solver->scores, new_size);
  kissat_decrease_phases(solver, new_size);

  solver->size = new_size;

#ifdef METRICS
  LOG("%s after decreasing size from %u to %u",
        FORMAT_BYTES(kissat_allocated(solver)), old_size, new_size);
#endif
}

void kissat_enlarge_variables(kissat *solver, unsigned new_vars) {
  if (solver->vars >= new_vars) {
    return;
  }
  assert(new_vars <= INTERNAL_MAX_VAR + 1);
  LOG("enlarging variables from %u to %u", solver->vars, new_vars);
  const size_t old_size = solver->size;
  if (old_size < new_vars) {
    LOG("old size %zu below requested new number of variables %u",
          old_size, new_vars);
    size_t new_size;
    if (!old_size) {
      new_size = new_vars;
    } else {
      if (kissat_is_power_of_two(old_size)) {
        assert(old_size <= UINT_MAX / 2);
        new_size = 2 * old_size;
      } else {
        assert(1 < old_size);
        new_size = 2;
      }
      while (new_size < new_vars) {
        assert(new_size <= UINT_MAX / 2);
        new_size *= 2;
      }
    }
    kissat_increase_size(solver, new_size);
  }
  solver->vars = new_vars;
  kissat_invalidate_cache(solver);
}

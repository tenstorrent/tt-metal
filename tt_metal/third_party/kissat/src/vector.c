#include "allocate.h"
#include "error.h"
#include "collect.h"
#include "inlinevector.h"
#include "logging.h"
#include "print.h"
#include "rank.h"

#include <inttypes.h>
#include <stddef.h>

#ifndef COMPACT

static void fix_vector_pointers_after_moving_stack(kissat *solver,
      ptrdiff_t moved) {
#ifdef LOGGING
  uint64_t bytes = moved < 0 ? -moved : moved;
  LOG("fixing begin and end pointers of all watches "
        "since the global watches stack has been moved by %s",
        FORMAT_BYTES(bytes));
#endif
  struct vector *begin_watches = solver->watches;
  struct vector *end_watches = begin_watches + LITS;
  for (struct vector *p = begin_watches; p != end_watches; p++) {
#define FIX_POINTER(PTR) \
do { \
char * old_char_ptr_value = (char*) (PTR); \
if (!old_char_ptr_value) \
break; \
char * new_char_ptr_value = old_char_ptr_value + moved; \
unsigned * new_unsigned_ptr_value = (unsigned *) new_char_ptr_value; \
(PTR) = new_unsigned_ptr_value; \
} while (0)
    FIX_POINTER(p->begin);
    FIX_POINTER(p->end);
  }
}

#endif

unsigned *kissat_enlarge_vector(kissat *solver, vector *vector) {
  unsigneds *stack = &solver->vectors.stack;
  const size_t old_vector_size = kissat_size_vector(vector);
#ifdef LOGGING
  const size_t old_offset = kissat_offset_vector(solver, vector);
  LOG2("enlarging vector %zu[%zu] at %p",
        old_offset, old_vector_size, (void *) vector);
#endif
  assert(old_vector_size < MAX_VECTORS / 2);
  const size_t new_vector_size = old_vector_size ? 2 * old_vector_size : 1;
  size_t old_stack_size = SIZE_STACK(*stack);
  size_t capacity = CAPACITY_STACK(*stack);
  assert(kissat_is_power_of_two(MAX_VECTORS));
  assert(capacity <= MAX_VECTORS);
  size_t available = capacity - old_stack_size;
  if (new_vector_size > available) {
#if !defined(QUIET) || !defined(COMPACT)
    unsigned *old_begin_stack = BEGIN_STACK(*stack);
#endif
    unsigned enlarged = 0;
    do {
      assert(kissat_is_zero_or_power_of_two(capacity));

      if (capacity == MAX_VECTORS)
        kissat_fatal("maximum vector stack size "
              "of 2^%u entries %s exhausted", LD_MAX_VECTORS,
              FORMAT_BYTES(MAX_VECTORS * sizeof(unsigned)));
      enlarged++;
      kissat_stack_enlarge(solver, (chars *) stack, sizeof(unsigned));

      capacity = CAPACITY_STACK(*stack);
      available = capacity - old_stack_size;
    } while (new_vector_size > available);

    if (enlarged) {
      INC(vectors_enlarged);
#if !defined(QUIET) || !defined(COMPACT)
      unsigned *new_begin_stack = BEGIN_STACK(*stack);
      const ptrdiff_t moved =
            (char *) new_begin_stack - (char *) old_begin_stack;
#endif
#ifndef QUIET
      kissat_phase(solver, "vectors",
            GET(vectors_enlarged),
            "enlarged to %s entries %s (%s)",
            FORMAT_COUNT(capacity),
            FORMAT_BYTES(capacity * sizeof(unsigned)),
            (moved ? "moved" : "in place"));
#endif
#ifndef COMPACT
      if (moved) {
        fix_vector_pointers_after_moving_stack(solver, moved);
      }
#endif
    }
    assert(capacity <= MAX_VECTORS);
    assert(new_vector_size <= available);
  }
  unsigned *begin_old_vector = kissat_begin_vector(solver, vector);
  unsigned *begin_new_vector = END_STACK(*stack);
  unsigned *middle_new_vector = begin_new_vector + old_vector_size;
  unsigned *end_new_vector = begin_new_vector + new_vector_size;
  assert(end_new_vector <= stack->allocated);
  const size_t old_bytes = old_vector_size * sizeof(unsigned);
  const size_t delta_size = new_vector_size - old_vector_size;
  assert(MAX_SIZE_T / sizeof(unsigned) >= delta_size);
  const size_t delta_bytes = delta_size * sizeof(unsigned);
  memcpy(begin_new_vector, begin_old_vector, old_bytes);
  memset(begin_old_vector, 0xff, old_bytes);
  solver->vectors.usable += old_vector_size;
  kissat_add_usable(solver, delta_size);
  memset(middle_new_vector, 0xff, delta_bytes);
#ifdef COMPACT
  const uint64_t offset = SIZE_STACK(*stack);
  assert(offset <= MAX_VECTORS);
  vector->offset = offset;
  LOG2("enlarged vector at %p to %u[%u]",
        (void *) vector, vector->offset, vector->size);
#else
  vector->begin = begin_new_vector;
  vector->end = middle_new_vector;
#ifdef LOGGING
  const size_t new_offset = vector->begin - stack->begin;
  LOG2("enlarged vector at %p to %zu[%zu]",
        (void *) vector, new_offset, old_vector_size);
#endif
#endif
  stack->end = end_new_vector;
  assert(begin_new_vector < end_new_vector);
  assert(kissat_size_vector(vector) == old_vector_size);
  return middle_new_vector;
}

#ifdef COMPACT

typedef unsigned rank;

static inline rank rank_offset(vector *unsorted, unsigned i) {
  return unsorted[i].offset;
}

#else

typedef uintptr_t rank;

static inline rank rank_offset(vector *unsorted, unsigned i) {
  const unsigned *begin = unsorted[i].begin;
  return (uintptr_t) begin;
}

#endif

#define RANK_OFFSET(A) \
  rank_offset (unsorted, (A))

void kissat_defrag_vectors(kissat *solver,
      size_t size_unsorted, vector *unsorted) {
  unsigneds *stack = &solver->vectors.stack;
  const size_t size_vectors = SIZE_STACK(*stack);
  if (size_vectors < 2) {
    return;
  }
  START(defrag);
  INC(defragmentations);
  LOG("defragmenting vectors size %zu capacity %zu usable %zu",
        size_vectors, CAPACITY_STACK(*stack), solver->vectors.usable);
  size_t bytes = size_unsorted * sizeof(unsigned);
  unsigned *sorted = kissat_malloc(solver, bytes);
  unsigned size_sorted = 0;
  for (unsigned i = 0; i < size_unsorted; i++) {
    vector *vector = unsorted + i;
    if (kissat_empty_vector(vector))
#ifdef COMPACT
      vector->offset = 0;
#else
      vector->begin = vector->end = 0;
#endif
    else {
      sorted[size_sorted++] = i;
    }
  }
  RADIX_SORT(unsigned, rank, size_sorted, sorted, RANK_OFFSET);
  unsigned *old_begin_stack = BEGIN_STACK(*stack);
  unsigned *p = old_begin_stack + 1;
  for (unsigned i = 0; i < size_sorted; i++) {
    unsigned j = sorted[i];
    vector *vector = unsorted + j;
    const size_t size = kissat_size_vector(vector);
    unsigned *new_end_of_vector = p + size;
#ifdef COMPACT
    const unsigned old_offset = vector->offset;
    const unsigned new_offset = p - old_begin_stack;
    assert(new_offset <= old_offset);
    vector->offset = new_offset;
    const unsigned *const q = old_begin_stack + old_offset;
#else
    if (!size) {
      vector->begin = vector->end = 0;
      continue;
    }
    const unsigned *const q = vector->begin;
    vector->begin = p;
    vector->end = new_end_of_vector;
#endif
    assert(MAX_SIZE_T / sizeof(unsigned) >= size);
    memmove(p, q, size * sizeof(unsigned));
    p = new_end_of_vector;
  }
  kissat_free(solver, sorted, bytes);
#ifndef QUIET
  const size_t freed = END_STACK(*stack) - p;
  double freed_fraction = kissat_percent(freed, size_vectors);
  kissat_phase(solver, "defrag", GET(defragmentations),
        "freed %zu usable entries %.0f%% thus %s",
        freed, freed_fraction,
        FORMAT_BYTES(freed * sizeof(unsigned)));
  assert(freed == solver->vectors.usable);
#endif
  SET_END_OF_STACK(*stack, p);
#ifndef COMPACT
  assert(old_begin_stack == BEGIN_STACK(*stack));
#endif
  SHRINK_STACK(*stack);
#ifndef COMPACT
  unsigned *new_begin_stack = BEGIN_STACK(*stack);
  const ptrdiff_t moved = (char *) new_begin_stack - (char *) old_begin_stack;
  if (moved) {
    fix_vector_pointers_after_moving_stack(solver, moved);
  }
#endif
  solver->vectors.usable = 0;
  kissat_check_vectors(solver);
  STOP(defrag);
}

void kissat_remove_from_vector(kissat *solver, vector *vector,
      unsigned remove) {
  unsigned *begin = kissat_begin_vector(solver, vector), *p = begin;
  const unsigned *const end = kissat_end_vector(solver, vector);
  assert(p != end);
  while (*p != remove) {
    p++, assert(p != end);
  }
  while (++p != end) {
    p[-1] = *p;
  }
  p[-1] = INVALID_VECTOR_ELEMENT;
#ifdef COMPACT
  vector->size--;
#else
  assert(vector->begin < vector->end);
  vector->end--;
#endif
  kissat_inc_usable(solver);
  kissat_check_vectors(solver);
#ifndef CHECK_VECTORS
  (void) solver;
#endif
}

void kissat_resize_vector(kissat *solver, vector *vector, size_t new_size) {
  const size_t old_size = kissat_size_vector(vector);
  assert(new_size <= old_size);
  if (new_size == old_size) {
    return;
  }
#ifdef COMPACT
  vector->size = new_size;
#else
  vector->end = vector->begin + new_size;
#endif
  unsigned *begin = kissat_begin_vector(solver, vector);
  unsigned *end = begin + new_size;
  size_t delta = old_size - new_size;
  kissat_add_usable(solver, delta);
  size_t bytes = delta * sizeof(unsigned);
  memset(end, 0xff, bytes);
  kissat_check_vectors(solver);
#ifndef CHECK_VECTORS
  (void) solver;
#endif
}

#ifdef CHECK_VECTORS

void kissat_check_vector(vectors *vectors, vector *vector) {
  assert(vectors == solver->vectors);
  const unsigned *const begin = kissat_begin_vector(vectors, vector);
  const unsigned *const end = kissat_end_vector(vectors, vector);
  for (const unsigned *p = begin; p != end; p++) {
    assert(*p != INVALID_VECTOR_ELEMENT);
  }
}

void kissat_check_vectors(kissat *solver) {
  for (all_literals(lit)) {
    vector *vector = &WATCHES(lit);
    kissat_check_vector(&solver->vectors, vector);
  }
  vectors *vectors = &solver->vectors;
  unsigneds *stack = &vectors->stack;
  const unsigned *const begin = BEGIN_STACK(*stack);
  const unsigned *const end = END_STACK(*stack);
  if (begin == end) {
    return;
  }
  size_t invalid = 0;
  for (const unsigned *p = begin + 1; p != end; p++)
    if (*p == INVALID_VECTOR_ELEMENT) {
      invalid++;
    }
  assert(invalid == solver->vectors.usable);
}

#endif

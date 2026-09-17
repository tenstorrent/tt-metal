#ifndef _inlinevector_h_INCLUDED
#define _inlinevector_h_INCLUDED

#include "internal.h"

static inline unsigned *kissat_begin_vector(kissat *solver, vector *vector) {
#ifdef COMPACT
  return BEGIN_STACK(solver->vectors.stack) + vector->offset;
#else
  (void) solver;
  return vector->begin;
#endif
}

static inline unsigned *kissat_end_vector(kissat *solver, vector *vector) {
#ifdef COMPACT
  return kissat_begin_vector(solver, vector) + vector->size;
#else
  (void) solver;
  return vector->end;
#endif
}

static inline const unsigned *kissat_begin_const_vector(kissat *solver,
      const vector *vector) {
#ifdef COMPACT
  return BEGIN_STACK(solver->vectors.stack) + vector->offset;
#else
  (void) solver;
  return vector->begin;
#endif
}

static inline const unsigned *kissat_end_const_vector(kissat *solver,
      const vector *vector) {
#ifdef COMPACT
  return kissat_begin_const_vector(solver, vector) + vector->size;
#else
  (void) solver;
  return vector->end;
#endif
}

#if defined(LOGGING) || defined(TEST_VECTOR)

static inline size_t kissat_offset_vector(kissat *solver, vector *vector) {
#ifdef COMPACT
  (void) solver;
  return vector->offset;
#else
  unsigned *begin_vector = vector->begin;
  unsigned *begin_stack = BEGIN_STACK(solver->vectors.stack);
  return begin_vector ? begin_vector - begin_stack : 0;
#endif
}

#endif

static inline size_t kissat_size_vector(const vector *vector) {
#ifdef COMPACT
  return vector->size;
#else
  return vector->end - vector->begin;
#endif
}

static inline bool kissat_empty_vector(vector *vector) {
#ifdef COMPACT
  return !vector->size;
#else
  return vector->end == vector->begin;
#endif
}

static inline void kissat_inc_usable(kissat *solver) {
  assert(MAX_SECTOR > solver->vectors.usable);
  solver->vectors.usable++;
}

static inline void kissat_add_usable(kissat *solver, size_t inc) {
  assert(MAX_SECTOR - inc >= solver->vectors.usable);
  solver->vectors.usable += inc;
}

static inline unsigned *kissat_last_vector_pointer(kissat *solver,
      vector *vector) {
  assert(!kissat_empty_vector(vector));
#ifdef COMPACT
  assert(vector->size);
  unsigned *begin = kissat_begin_vector(solver, vector);
  return begin + vector->size - 1;
#else
  (void) solver;
  return vector->end - 1;
#endif
}

#ifdef TEST_VECTOR

static inline void kissat_pop_vector(kissat *solver, vector *vector) {
  assert(!kissat_empty_vector(vector));
#ifdef COMPACT
  unsigned *p = kissat_last_vector_pointer(solver, vector);
  vector->size--;
  *p = INVALID_VECTOR_ELEMENT;
#else
  *--vector->end = INVALID_VECTOR_ELEMENT;
  (void) solver;
#endif
  kissat_inc_usable(solver);
}

#endif

static inline void kissat_release_vector(kissat *solver, vector *vector) {
  kissat_resize_vector(solver, vector, 0);
}

static inline void kissat_dec_usable(kissat *solver) {
  assert(solver->vectors.usable > 0);
  solver->vectors.usable--;
}

static inline void kissat_push_vectors(kissat *solver, vector *vector,
      unsigned e) {
  unsigneds *stack = &solver->vectors.stack;
  assert(e != INVALID_VECTOR_ELEMENT);
  if (
#ifdef COMPACT
        !vector->size && !vector->offset
#else
        !vector->begin
#endif
  ) {
    if (EMPTY_STACK(*stack)) {
      PUSH_STACK(*stack, 0);
    }
    if (FULL_STACK(*stack)) {
      unsigned *end = kissat_enlarge_vector(solver, vector);
      assert(*end == INVALID_VECTOR_ELEMENT);
      *end = e;
      kissat_dec_usable(solver);
    } else {
#ifdef COMPACT
      assert((uint64_t) SIZE_STACK(*stack) < MAX_VECTORS);
      vector->offset = SIZE_STACK(*stack);
      assert(vector->offset);
      *stack->end++ = e;
#else
      assert(stack->end < stack->allocated);
      *(vector->begin = stack->end++) = e;
#endif
    }
#if !defined(COMPACT)
    vector->end = vector->begin;
#endif
  } else {
    unsigned *end = kissat_end_vector(solver, vector);
    if (end == END_STACK(*stack)) {
      if (FULL_STACK(*stack)) {
        end = kissat_enlarge_vector(solver, vector);
        assert(*end == INVALID_VECTOR_ELEMENT);
        *end = e;
        kissat_dec_usable(solver);
      } else {
        *stack->end++ = e;
      }
    } else {
      if (*end != INVALID_VECTOR_ELEMENT) {
        end = kissat_enlarge_vector(solver, vector);
      }
      assert(*end == INVALID_VECTOR_ELEMENT);
      *end = e;
      kissat_dec_usable(solver);
    }
  }
#ifndef COMPACT
  vector->end++;
#else
  vector->size++;
#endif
  kissat_check_vectors(solver);
}

#ifdef TEST_VECTOR

#define all_vector(E,V) \
  unsigned E, *       E ## _PTR = kissat_begin_vector (solver, &V), \
              * const E ## _END = kissat_end_vector (solver, &V); \
  E ## _PTR != E ## _END && (E = *E ## _PTR, true); \
  E ## _PTR++

#endif

#endif

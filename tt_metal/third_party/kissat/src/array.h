#ifndef _array_h_INCLUDED
#define _array_h_INCLUDED

#include "allocate.h"
#include "stack.h"

#define ARRAY(TYPE) \
  struct { TYPE * begin; TYPE * end; }

#define ALLOCATE_ARRAY(A,N) \
do { \
  const size_t TMP_N = (N); \
  (A).begin = (A).end = kissat_nalloc (solver, TMP_N, sizeof *(A).begin); \
} while (0)

#define EMPTY_ARRAY EMPTY_STACK
#define SIZE_ARRAY SIZE_STACK

#define PUSH_ARRAY(A,E) \
do { \
  *(A).end++ = (E); \
} while (0)

#define REALLOCATE_ARRAY(A,O,N) \
do { \
  const size_t SIZE = SIZE_ARRAY (A); \
  (A).begin = kissat_nrealloc (solver, (A).begin, \
                               (O), (N), sizeof *(A).begin); \
  (A).end = (A).begin + SIZE; \
} while (0)

#define RELEASE_ARRAY(A,N) \
do { \
  const size_t TMP_NIZE = (N); \
  DEALLOC((A).begin, TMP_NIZE); \
} while (0)

#define CLEAR_ARRAY CLEAR_STACK
#define TOP_ARRAY TOP_STACK
#define PEEK_ARRAY PEEK_STACK
#define POKE_ARRAY POKE_STACK
#define POP_ARRAY POP_STACK
#define BEGIN_ARRAY BEGIN_STACK
#define END_ARRAY END_STACK
#define RESIZE_ARRAY RESIZE_STACK
#define SET_END_OF_ARRAY SET_END_OF_STACK

// *INDENT-OFF*

typedef ARRAY (unsigned) unsigned_array;

// *INDENT-ON*

#endif

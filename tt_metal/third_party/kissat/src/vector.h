#ifndef _vector_h_INCLUDED
#define _vector_h_INCLUDED

#include "stack.h"
#include "utilities.h"

#include <limits.h>

#ifdef COMPACT
#define LD_MAX_VECTORS (sizeof (word) == 8 ? 32u : 28u)
#else
#define LD_MAX_VECTORS (sizeof (word) == 8 ? 48u : 28u)
#endif

#define MAX_VECTORS (((uint64_t) 1) << LD_MAX_VECTORS)

#define INVALID_VECTOR_ELEMENT UINT_MAX

#define MAX_SECTOR MAX_SIZE_T

typedef struct vector vector;
typedef struct vectors vectors;

struct vectors {
  unsigneds stack;
  size_t usable;
};

struct vector {
#ifdef COMPACT
  unsigned offset;
  unsigned size;
#else
  unsigned *begin;
  unsigned *end;
#endif
};

struct kissat;

#ifdef CHECK_VECTORS
void kissat_check_vectors(struct kissat *);
#else
#define kissat_check_vectors(...) do { } while (0)
#endif

unsigned *kissat_enlarge_vector(struct kissat *, vector *);
void kissat_defrag_vectors(struct kissat *, size_t, vector *);
void kissat_remove_from_vector(struct kissat *, vector *, unsigned);
void kissat_resize_vector(struct kissat *, vector *, size_t);

#endif

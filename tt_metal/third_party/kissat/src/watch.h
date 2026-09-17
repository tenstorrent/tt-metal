#ifndef _watch_h_INCLUDED
#define _watch_h_INCLUDED

#include "endianness.h"
#include "reference.h"
#include "stack.h"
#include "vector.h"

#include <stdbool.h>

typedef union watch watch;

typedef struct watch_type watch_type;
typedef struct binary_watch binary_watch;
typedef struct blocking_watch blocking_watch;
typedef struct large_watch large_watch;

struct watch_type {
#ifdef KISSAT_IS_BIG_ENDIAN
  bool binary: 1;
  unsigned padding: 2;
  unsigned lit: 29;
#else
  unsigned lit: 29;
  unsigned padding: 2;
  bool binary: 1;
#endif
};

struct binary_watch {
#ifdef KISSAT_IS_BIG_ENDIAN
  bool binary: 1;
  bool redundant: 1;
  bool hyper: 1;
  unsigned lit: 29;
#else
  unsigned lit: 29;
  bool hyper: 1;
  bool redundant: 1;
  bool binary: 1;
#endif
};

struct large_watch {
#ifdef KISSAT_IS_BIG_ENDIAN
  bool binary: 1;
  unsigned ref: 31;
#else
  unsigned ref: 31;
  bool binary: 1;
#endif
};

struct blocking_watch {
#ifdef KISSAT_IS_BIG_ENDIAN
  bool binary: 1;
  unsigned padding: 2;
  unsigned lit: 29;
#else
  unsigned lit: 29;
  unsigned padding: 2;
  bool binary: 1;
#endif
};

union watch {
  watch_type type;
  binary_watch binary;
  blocking_watch blocking;
  large_watch large;
  unsigned raw;
};

typedef vector watches;

typedef struct litwatch litwatch;
typedef struct litpair litpair;

// *INDENT-OFF*
typedef STACK (litwatch) litwatches;
typedef STACK (litpair) litpairs;
// *INDENT-ON*

struct litwatch {
  unsigned lit;
  watch watch;
};

struct litpair {
  unsigned lits[2];
};

static inline litpair kissat_litpair(unsigned lit, unsigned other) {
  litpair res;
  res.lits[0] = lit < other ? lit : other;
  res.lits[1] = lit < other ? other : lit;
  return res;
}

static inline watch kissat_binary_watch(unsigned lit, bool redundant,
      bool hyper) {
  assert(redundant || !hyper);
  watch res;
  res.binary.lit = lit;
  res.binary.redundant = redundant;
  res.binary.hyper = hyper;
  res.binary.binary = true;
  assert(res.type.binary);
  return res;
}

static inline watch kissat_large_watch(reference ref) {
  watch res;
  res.large.ref = ref;
  res.large.binary = false;
  assert(!res.type.binary);
  return res;
}

static inline watch kissat_blocking_watch(unsigned lit) {
  watch res;
  res.blocking.lit = lit;
  res.blocking.padding = 0;
  res.blocking.binary = false;
  assert(!res.type.binary);
  return res;
}

#define EMPTY_WATCHES(W) kissat_empty_vector (&W)
#define SIZE_WATCHES(W) kissat_size_vector (&W)

#define PUSH_WATCHES(W,E) \
do { \
  assert (sizeof (E) == sizeof (unsigned)); \
  kissat_push_vectors (solver, &(W), (E).raw); \
} while (0)

#define LAST_WATCH_POINTER(WS) \
  (watch *) kissat_last_vector_pointer (solver, &WS)

#define BEGIN_WATCHES(WS) \
  ((union watch*) kissat_begin_vector (solver, &(WS)))

#define END_WATCHES(WS) \
  ((union watch*) kissat_end_vector (solver, &(WS)))

#define BEGIN_CONST_WATCHES(WS) \
  ((union watch*) kissat_begin_const_vector (solver, &(WS)))

#define END_CONST_WATCHES(WS) \
  ((union watch*) kissat_end_const_vector (solver, &(WS)))

#define RELEASE_WATCHES(WS) \
  kissat_release_vector (solver, &(WS))

#define SET_END_OF_WATCHES(WS,P) \
do { \
  size_t SIZE = (unsigned*)(P) - kissat_begin_vector (solver, &WS); \
  kissat_resize_vector (solver, &WS, SIZE); \
} while (0)

#define REMOVE_WATCHES(W,E) \
  kissat_remove_from_vector (solver, &(W), (E).raw)

#define WATCHES(LIT) (solver->watches[assert ((LIT) < LITS), (LIT)])

// This iterator is currently only used in 'testreferences.c'.
//
#define all_binary_blocking_watch_ref(WATCH,REF,WATCHES) \
  watch WATCH, \
    * WATCH ## _PTR = (assert (solver->watching), BEGIN_WATCHES (WATCHES)), \
    * const WATCH ## _END = END_WATCHES (WATCHES); \
  WATCH ## _PTR != WATCH ## _END && \
    ((WATCH = *WATCH ## _PTR), \
     (REF = WATCH.type.binary ? INVALID_REF : \
	    WATCH ## _PTR[1].large.ref), true); \
  WATCH ## _PTR += 1u + !WATCH.type.binary

#define all_binary_blocking_watches(WATCH,WATCHES) \
  watch WATCH, \
    * WATCH ## _PTR = (assert (solver->watching), BEGIN_WATCHES (WATCHES)), \
    * const WATCH ## _END = END_WATCHES (WATCHES); \
  WATCH ## _PTR != WATCH ## _END && ((WATCH = *WATCH ## _PTR), true); \
  WATCH ## _PTR += 1u + !WATCH.type.binary

#define all_binary_large_watches(WATCH,WATCHES) \
  watch WATCH, \
    * WATCH ## _PTR = (assert (!solver->watching), BEGIN_WATCHES (WATCHES)), \
    * const WATCH ## _END = END_WATCHES (WATCHES); \
  WATCH ## _PTR != WATCH ## _END && ((WATCH = *WATCH ## _PTR), true); \
  ++WATCH ## _PTR

void kissat_remove_blocking_watch(struct kissat *, watches *, reference);

void kissat_flush_large_watches(struct kissat *);
void kissat_watch_large_clauses(struct kissat *);

void kissat_connect_irredundant_large_clauses(struct kissat *);

void kissat_flush_large_connected(struct kissat *);

#endif

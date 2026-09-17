#ifndef _sort_h_INCLUDED
#define _sort_h_INCLUDED

#include "utilities.h"

#define GREATER_SWAP(TYPE,P,Q,LESS) \
do { \
  if (LESS (Q, P)) \
    SWAP (TYPE, P, Q); \
} while(0)

#define SORTER (solver->sorter)

#define PARTITION(TYPE,L,R,A,LESS) \
do { \
  const size_t L_PARTITION = (L); \
  I_QUICK_SORT = L_PARTITION - 1; \
  \
  size_t J_PARTITION = (R); \
  \
  TYPE PIVOT_PARTITION = A[J_PARTITION]; \
  \
  for (;;) \
    { \
      while (LESS (A[++I_QUICK_SORT], PIVOT_PARTITION)) \
	; \
      while (LESS (PIVOT_PARTITION, A[--J_PARTITION])) \
	if (J_PARTITION == L_PARTITION) \
	  break; \
      \
      if (I_QUICK_SORT >= J_PARTITION) \
	break; \
      \
      SWAP (TYPE, A[I_QUICK_SORT], A[J_PARTITION]); \
    } \
  \
  SWAP (TYPE, A[I_QUICK_SORT], A[R]); \
} while(0)

#define QUICK_SORT_LIMIT 10

#define QUICK_SORT(TYPE,N,A,LESS) \
do { \
  assert (N); \
  assert (EMPTY_STACK (SORTER)); \
  \
  size_t L_QUICK_SORT = 0; \
  size_t R_QUICK_SORT = N - 1; \
  \
  if (R_QUICK_SORT - L_QUICK_SORT <= QUICK_SORT_LIMIT) \
    break; \
  \
  for (;;) \
    { \
      const size_t M = L_QUICK_SORT + (R_QUICK_SORT - L_QUICK_SORT) / 2; \
      \
      SWAP (TYPE, A[M], A[R_QUICK_SORT - 1]); \
      \
      GREATER_SWAP (TYPE, A[L_QUICK_SORT],     A[R_QUICK_SORT - 1], LESS); \
      GREATER_SWAP (TYPE, A[L_QUICK_SORT],     A[R_QUICK_SORT],     LESS); \
      GREATER_SWAP (TYPE, A[R_QUICK_SORT - 1], A[R_QUICK_SORT],     LESS); \
      \
      size_t I_QUICK_SORT; \
      \
      PARTITION (TYPE, L_QUICK_SORT + 1, R_QUICK_SORT - 1, A, LESS); \
      assert (L_QUICK_SORT < I_QUICK_SORT); \
      assert (I_QUICK_SORT <= R_QUICK_SORT); \
      \
      size_t LL_QUICK_SORT; \
      size_t RR_QUICK_SORT; \
      \
      if (I_QUICK_SORT - L_QUICK_SORT < R_QUICK_SORT - I_QUICK_SORT) \
	{ \
	  LL_QUICK_SORT = I_QUICK_SORT + 1; \
	  RR_QUICK_SORT = R_QUICK_SORT; \
	  R_QUICK_SORT = I_QUICK_SORT - 1; \
	} \
      else \
	{ \
	  LL_QUICK_SORT = L_QUICK_SORT; \
	  RR_QUICK_SORT = I_QUICK_SORT - 1; \
	  L_QUICK_SORT = I_QUICK_SORT + 1; \
	} \
      if (R_QUICK_SORT - L_QUICK_SORT > QUICK_SORT_LIMIT) \
	{ \
	  assert (RR_QUICK_SORT - LL_QUICK_SORT > QUICK_SORT_LIMIT); \
	  PUSH_STACK (SORTER, LL_QUICK_SORT); \
	  PUSH_STACK (SORTER, RR_QUICK_SORT); \
	} \
      else if (RR_QUICK_SORT - LL_QUICK_SORT > QUICK_SORT_LIMIT) \
	{ \
	  L_QUICK_SORT = LL_QUICK_SORT; \
	  R_QUICK_SORT = RR_QUICK_SORT; \
	} \
      else if (!EMPTY_STACK (SORTER)) \
	{ \
	  R_QUICK_SORT = POP_STACK (SORTER); \
	  L_QUICK_SORT = POP_STACK (SORTER); \
	} else break; \
    } \
} while (0)

#define INSERTION_SORT(TYPE,N,A,LESS) \
do { \
  size_t L_INSERTION_SORT = 0; \
  size_t R_INSERTION_SORT = N - 1; \
  \
  for (size_t I_INSERTION_SORT = R_INSERTION_SORT; \
       I_INSERTION_SORT > L_INSERTION_SORT; \
       I_INSERTION_SORT--) \
    GREATER_SWAP (TYPE, A[I_INSERTION_SORT - 1], A[I_INSERTION_SORT], LESS); \
  \
  for (size_t I_INSERTION_SORT = L_INSERTION_SORT + 2; \
        I_INSERTION_SORT <= R_INSERTION_SORT; \
	I_INSERTION_SORT++) \
    { \
      TYPE PIVOT_INSERTION_SORT = A[I_INSERTION_SORT]; \
      \
      size_t J_INSERTION_SORT = I_INSERTION_SORT; \
      \
      while (LESS (PIVOT_INSERTION_SORT, A[J_INSERTION_SORT - 1])) \
	{ \
	  A[J_INSERTION_SORT] = A[J_INSERTION_SORT - 1]; \
	  J_INSERTION_SORT--; \
	} \
      \
      A[J_INSERTION_SORT] = PIVOT_INSERTION_SORT; \
    } \
} while (0)

#ifdef NDEBUG
#define CHECK_SORTED(...) do { } while(0)
#else
#define CHECK_SORTED(N,A,LESS) \
do { \
  for (size_t I_CHECK_SORTED = 0; I_CHECK_SORTED < N - 1; I_CHECK_SORTED++) \
    assert (!LESS (A[I_CHECK_SORTED+1], A[I_CHECK_SORTED])); \
} while(0)
#endif

#define SORT(TYPE,N,A,LESS) \
do { \
  const size_t N_SORT = (N); \
  if (!N_SORT) \
    break; \
  START (sort); \
  TYPE * A_SORT = (A); \
  QUICK_SORT (TYPE, N_SORT, A_SORT, LESS); \
  INSERTION_SORT (TYPE, N_SORT, A_SORT, LESS); \
  CHECK_SORTED (N_SORT, A_SORT, LESS); \
  STOP (sort); \
} while (0)

#define SORT_STACK(TYPE,S,LESS) \
do { \
  const size_t N_SORT_STACK = SIZE_STACK (S); \
  if (N_SORT_STACK <= 1) \
    break; \
  TYPE * A_SORT_STACK = BEGIN_STACK (S); \
  SORT (TYPE, N_SORT_STACK, A_SORT_STACK, LESS); \
} while (0)

#endif

#ifndef _rank_h_INCLUDED
#define _rank_h_INCLUDED

#include "allocate.h"

#include <string.h>

#ifdef NDEBUG
#define CHECK_RANKED(...) do { } while (0)
#else
#define CHECK_RANKED(N,A,RANK) \
do { \
  assert (0 < (N)); \
  for (size_t I_CHECK_RANKED = 0; I_CHECK_RANKED < N-1; I_CHECK_RANKED++) \
    assert (RANK (A[I_CHECK_RANKED]) <= RANK (A[I_CHECK_RANKED + 1])); \
} while (0)
#endif

#define RADIX_SORT(VTYPE,RTYPE,N,V,RANK) \
do { \
  const size_t N_RADIX = (N); \
  if (N_RADIX <= 1) \
    break; \
  \
  START (radix); \
  \
  VTYPE * V_RADIX = (V); \
  \
  const size_t LENGTH_RADIX = 8; \
  const size_t WIDTH_RADIX = (1 << LENGTH_RADIX); \
  const RTYPE MASK_RADIX = WIDTH_RADIX - 1; \
  \
  size_t COUNT_RADIX[WIDTH_RADIX]; \
  \
  VTYPE * TMP_RADIX = 0; \
  const size_t BYTES_TMP_RADIX = N_RADIX * sizeof (VTYPE); \
  \
  VTYPE * A_RADIX = V_RADIX; \
  VTYPE * B_RADIX = 0; \
  VTYPE * C_RADIX = A_RADIX; \
  \
  RTYPE MLOWER_RADIX = 0; \
  RTYPE MUPPER_RADIX = MASK_RADIX; \
  \
  bool BOUNDED_RADIX = false; \
  RTYPE UPPER_RADIX = 0; \
  RTYPE LOWER_RADIX = ~UPPER_RADIX; \
  RTYPE SHIFT_RADIX = MASK_RADIX; \
  \
  for (size_t I_RADIX = 0; \
       I_RADIX < 8 * sizeof (RTYPE); \
       I_RADIX += LENGTH_RADIX, \
       SHIFT_RADIX <<= LENGTH_RADIX) \
    { \
      if (BOUNDED_RADIX && \
	  (LOWER_RADIX & SHIFT_RADIX) == (UPPER_RADIX & SHIFT_RADIX)) \
	continue; \
      \
      memset (COUNT_RADIX + MLOWER_RADIX, 0, \
              (MUPPER_RADIX - MLOWER_RADIX + 1) * sizeof *COUNT_RADIX); \
      \
      VTYPE * END_RADIX = C_RADIX + N_RADIX; \
      \
      bool SORTED_RADIX = true; \
      RTYPE LAST_RADIX = 0; \
      \
      for (VTYPE * P_RADIX = C_RADIX; P_RADIX != END_RADIX; P_RADIX++) \
	{ \
	  RTYPE R_RADIX = RANK (*P_RADIX); \
	  if (!BOUNDED_RADIX) \
	    { \
	      LOWER_RADIX &= R_RADIX; \
	      UPPER_RADIX |= R_RADIX; \
	    } \
	  RTYPE S_RADIX = R_RADIX >> I_RADIX; \
	  RTYPE M_RADIX = S_RADIX & MASK_RADIX; \
	  if (SORTED_RADIX && LAST_RADIX > M_RADIX) \
	    SORTED_RADIX = false; \
	  else \
	    LAST_RADIX = M_RADIX; \
	  COUNT_RADIX[M_RADIX]++; \
	} \
      \
      MLOWER_RADIX = (LOWER_RADIX >> I_RADIX) & MASK_RADIX; \
      MUPPER_RADIX = (UPPER_RADIX >> I_RADIX) & MASK_RADIX; \
      \
      if (!BOUNDED_RADIX) \
	{ \
	  BOUNDED_RADIX = true;  \
	  if ((LOWER_RADIX & SHIFT_RADIX) == (UPPER_RADIX & SHIFT_RADIX)) \
	    continue; \
	} \
      \
      if (SORTED_RADIX) \
	continue; \
      \
      size_t POS_RADIX = 0; \
      for (size_t J_RADIX = MLOWER_RADIX; J_RADIX <= MUPPER_RADIX; J_RADIX++) \
	{ \
	  const size_t DELTA_RADIX = COUNT_RADIX[J_RADIX]; \
	  COUNT_RADIX[J_RADIX] = POS_RADIX; \
	  POS_RADIX += DELTA_RADIX; \
	} \
      \
      if (!TMP_RADIX) \
	{ \
	  assert (C_RADIX == A_RADIX); \
	  TMP_RADIX = kissat_malloc (solver, BYTES_TMP_RADIX); \
	  B_RADIX = TMP_RADIX; \
	} \
      \
      assert (B_RADIX == TMP_RADIX); \
      \
      VTYPE * D_RADIX = (C_RADIX == A_RADIX) ? B_RADIX : A_RADIX; \
      \
      for (VTYPE * P_RADIX = C_RADIX; P_RADIX != END_RADIX; P_RADIX++) \
	{ \
	  RTYPE R_RADIX = RANK (*P_RADIX); \
	  RTYPE S_RADIX = R_RADIX >> I_RADIX; \
	  RTYPE M_RADIX = S_RADIX & MASK_RADIX; \
	  const size_t POS_RADIX = COUNT_RADIX[M_RADIX]++; \
	  D_RADIX[POS_RADIX] = *P_RADIX; \
	} \
      \
      C_RADIX = D_RADIX; \
    } \
  \
  if (C_RADIX == B_RADIX) \
    memcpy (A_RADIX, B_RADIX, N_RADIX * sizeof *A_RADIX); \
  \
  if (TMP_RADIX) \
    kissat_free (solver, TMP_RADIX, BYTES_TMP_RADIX); \
  \
  CHECK_RANKED (N_RADIX, V_RADIX, RANK); \
  STOP (radix); \
} while (0)

#define RADIX_STACK(VTYPE,RTYPE,S,RANK) \
do { \
  const size_t N_RADIX_STACK = SIZE_STACK (S); \
  VTYPE * A_RADIX_STACK = BEGIN_STACK (S); \
  RADIX_SORT(VTYPE,RTYPE,N_RADIX_STACK,A_RADIX_STACK,RANK); \
} while (0)

#endif

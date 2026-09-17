#ifndef _literal_h_INCLUDED
#define _literal_h_INCLUDED

#include <limits.h>

#define LD_MAX_VAR 28u

#define EXTERNAL_MAX_VAR ((1<<LD_MAX_VAR) - 1)
#define INTERNAL_MAX_VAR ((1u<<LD_MAX_VAR) - 2)

#define INTERNAL_MAX_LIT ((INTERNAL_MAX_VAR<<1) + 1)

#define ILLEGAL_LIT (INTERNAL_MAX_LIT + 1)

#define INVALID_IDX UINT_MAX
#define INVALID_LIT UINT_MAX

#define VALID_INTERNAL_INDEX(IDX) \
  ((IDX) < VARS)

#define VALID_INTERNAL_LITERAL(LIT) \
  ((LIT) < LITS)

#define VALID_EXTERNAL_LITERAL(LIT) \
  ((LIT) && ((LIT) != INT_MIN) && ABS (LIT) <= EXTERNAL_MAX_VAR)

#define IDX(LIT) \
  (assert (VALID_INTERNAL_LITERAL (LIT)), (((unsigned)(LIT)) >> 1))

#define LIT(IDX) \
  (assert (VALID_INTERNAL_INDEX (IDX)), ((IDX) << 1))

#define NOT(LIT) \
   (assert (VALID_INTERNAL_LITERAL (LIT)), ((LIT) ^ 1u))

#define NEGATED(LIT) \
  (assert (VALID_INTERNAL_LITERAL (LIT)), ((LIT) & 1u))

#define STRIP(LIT) \
  (assert (VALID_INTERNAL_LITERAL (LIT)), ((LIT) & ~1u))

#endif

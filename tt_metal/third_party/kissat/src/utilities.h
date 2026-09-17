#ifndef _utilities_h_INCLUDED
#define _utilities_h_INCLUDED

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef uintptr_t word;
typedef uintptr_t w2rd[2];

#define WORD_ALIGNMENT_MASK (sizeof (word)-1)
#define W2RD_ALIGNMENT_MASK (sizeof (w2rd)-1)

#define WORD_FORMAT PRIuPTR

#define MAX_SIZE_T (~ (size_t) 0)

#define ASSUMED_LD_CACHE_LINE_BYTES 7u

static inline word kissat_cache_lines(word n, size_t size) {
  if (!n) {
    return 0;
  }
#ifdef NDEBUG
  (void) size;
#endif
  assert(size == 4);
  assert(ASSUMED_LD_CACHE_LINE_BYTES > 2);
  const unsigned shift = ASSUMED_LD_CACHE_LINE_BYTES - 2u;
  const word mask = (((word) 1) << shift) - 1;
  const word masked = n + mask;
  const word res = masked >> shift;
  return res;
}

static inline double kissat_average(double a, double b) {
  return b ? a / b : 0.0;
}

static inline double kissat_percent(double a, double b) {
  return kissat_average(100.0 * a, b);
}

static inline bool kissat_aligned_word(word word) {
  return !(word & WORD_ALIGNMENT_MASK);
}

static inline bool kissat_aligned_pointer(const void *p) {
  return kissat_aligned_word((word) p);
}

static inline word kissat_align_word(word w) {
  word res = w;
  if (res & WORD_ALIGNMENT_MASK) {
    res = 1 + (res | WORD_ALIGNMENT_MASK);
  }
  return res;
}

static inline word kissat_align_w2rd(word w) {
  word res = w;
  if (res & W2RD_ALIGNMENT_MASK) {
    res = 1 + (res | W2RD_ALIGNMENT_MASK);
  }
  return res;
}

bool kissat_has_suffix(const char *str, const char *suffix);

static inline bool kissat_is_power_of_two(uint64_t w) {
  return w && !(w & (w - 1));
}

static inline bool kissat_is_zero_or_power_of_two(word w) {
  return !(w & (w - 1));
}

static inline unsigned kissat_leading_zeroes_of_unsigned(unsigned x) {
  return x ? __builtin_clz(x) : sizeof(unsigned) * 8;
}

static inline unsigned kissat_leading_zeroes_of_word(word x) {
  if (!x) {
    return sizeof(word) * 8;
  }
  if (sizeof(word) == sizeof(unsigned long long)) {
    return __builtin_clzll(x);
  }
  if (sizeof(word) == sizeof(unsigned long)) {
    return __builtin_clzl(x);
  }
  return __builtin_clz(x);
}

static inline unsigned kissat_log2_floor_of_word(word x) {
  return x ? sizeof(word) * 8 - 1 - kissat_leading_zeroes_of_word(x) : 0;
}

static inline unsigned kissat_log2_ceiling_of_word(word x) {
  if (!x) {
    return 0;
  }
  unsigned tmp = kissat_log2_floor_of_word(x);
  return tmp + ! !(x ^ (((word) 1) << tmp));
}

static inline unsigned kissat_leading_zeroes_of_uint64(uint64_t x) {
  if (!x) {
    return sizeof(uint64_t) * 8;
  }
  if (sizeof(uint64_t) == sizeof(unsigned long long)) {
    return __builtin_clzll(x);
  }
  if (sizeof(uint64_t) == sizeof(unsigned long)) {
    return __builtin_clzl(x);
  }
  return __builtin_clz(x);
}

static inline unsigned kissat_log2_floor_of_uint64(uint64_t x) {
  return x ?
        sizeof(uint64_t) * 8 - 1 - kissat_leading_zeroes_of_uint64(x) : 0;
}

static inline unsigned kissat_log2_ceiling_of_uint64(uint64_t x) {
  if (!x) {
    return 0;
  }
  unsigned tmp = kissat_log2_floor_of_uint64(x);
  return tmp + ! !(x ^ (((uint64_t) 1) << tmp));
}

#define SWAP(TYPE,A,B) \
do { \
  TYPE TMP_SWAP = (A); \
  (A) = (B); \
  (B) = (TMP_SWAP); \
} while (0)

#define MIN(A,B) \
  ((A) > (B)  ? (B) : (A))

#define MAX(A,B) \
  ((A) < (B)  ? (B) : (A))

#define ABS(A) \
  (assert ((int)(A) != INT_MIN), (A) < 0 ? -(A) : (A))

#endif

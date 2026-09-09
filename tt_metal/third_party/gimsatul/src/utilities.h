#ifndef _utilities_h_INCLUDED
#define _utilities_h_INCLUDED

#include "macros.h"
#include "options.h"

#include <assert.h>
#include <stdlib.h>

static inline double average (double a, double b) { return b ? a / b : 0; }

static inline double percent (double a, double b) {
  return average (100 * a, b);
}

double logn (uint64_t count);
double nlogn (uint64_t count);
double nlog2n (uint64_t count);
double nlog3n (uint64_t count);
double nlog4n (uint64_t count);

unsigned gcd (unsigned, unsigned);
unsigned log2ceil (unsigned);

static inline void mark_literal (signed char *marks, unsigned lit) {
  unsigned idx = IDX (lit);
  assert (!marks[idx]);
  marks[idx] = SGN (lit) ? -1 : 1;
}

static inline void unmark_literal (signed char *marks, unsigned lit) {
  unsigned idx = IDX (lit);
  assert (marks[idx]);
  marks[idx] = 0;
}

static inline signed char marked_literal (signed char *marks,
                                          unsigned lit) {
  unsigned idx = IDX (lit);
  signed char res = marks[idx];
  if (SGN (lit))
    res = -res;
  return res;
}

static inline unsigned unmap_literal (unsigned *unmap, unsigned lit) {
  if (!unmap)
    return lit;
  unsigned idx = IDX (lit);
  unsigned unmapped_idx = unmap[idx];
  unsigned res = LIT (unmapped_idx);
  res ^= SGN (lit);
  return res;
}

static inline int only_export_literal (unsigned unsigned_lit) {
  unsigned unsigned_idx = IDX (unsigned_lit);
  assert (unsigned_idx < (unsigned) INT_MAX);
  int signed_lit = unsigned_idx + 1;
  if (SGN (unsigned_lit))
    signed_lit *= -1;
  return signed_lit;
}

static inline int unmap_and_export_literal (unsigned *unmap,
                                            unsigned unsigned_lit) {
  unsigned unmapped_idx;
  if (unmap) {
    unsigned unsigned_idx = IDX (unsigned_lit);
    unmapped_idx = unmap[unsigned_idx];
  } else
    unmapped_idx = IDX (unsigned_lit);
  assert (unmapped_idx < (unsigned) INT_MAX);
  int signed_lit = unmapped_idx + 1;
  if (SGN (unsigned_lit))
    signed_lit *= -1;
  return signed_lit;
}

static inline size_t cache_lines (void *p, void *q) {
  if (p == q)
    return 0;
  assert (p >= q);
  size_t bytes = (char *) p - (char *) q;
  size_t res = (bytes + (CACHE_LINE_SIZE - 1)) / CACHE_LINE_SIZE;
  return res;
}

static inline bool is_power_of_two (size_t n) {
  return n && !(n & (n - 1));
}

#endif

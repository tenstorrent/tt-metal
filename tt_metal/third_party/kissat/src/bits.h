#ifndef _bits_h_INCLUDED
#define _bits_h_INCLUDED

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

typedef unsigned bits;

struct kissat;

bits *kissat_new_bits(struct kissat *, size_t size);
void kissat_delete_bits(struct kissat *, bits *, size_t size);

static inline size_t kissat_bits_size_in_words(size_t size) {
  assert(sizeof(bits) == 4);
  return (size >> 5) + ! !(size & 31);
}

static inline bool kissat_get_bit(const bits *bits, size_t size, size_t bit) {
  assert(bit < size);
  const size_t x = (bit >> 5);
  assert(x < kissat_bits_size_in_words(size));
  const unsigned word = bits[x];
  const unsigned y = (bit & 31);
  const unsigned mask = (1u << y);
  const unsigned masked = word & mask;
  const bool res = ! !masked;
  (void) size;
  return res;
}

static inline void kissat_set_bit_to_true(bits *bits, size_t size, size_t bit) {
  assert(bit < size);
  const size_t x = (bit >> 5);
  assert(x < kissat_bits_size_in_words(size));
  unsigned word = bits[x];
  const unsigned y = (bit & 31);
  const unsigned mask = (1u << y);
  word |= mask;
  bits[x] = word;
  (void) size;
}

static inline void kissat_set_bit_to_false(bits *bits, size_t size,
      size_t bit) {
  assert(bit < size);
  const size_t x = (bit >> 5);
  assert(x < kissat_bits_size_in_words(size));
  unsigned word = bits[x];
  const unsigned y = (bit & 31);
  const unsigned mask = (1u << y);
  word &= ~mask;
  bits[x] = word;
  (void) size;
}

static inline void kissat_set_bit_explicitly(bits *bits, size_t size,
      size_t bit, bool value) {
  assert(bit < size);
  const size_t x = (bit >> 5);
  assert(x < kissat_bits_size_in_words(size));
  unsigned word = bits[x];
  const unsigned y = (bit & 31);
  const unsigned clear = (1 << y);
  const unsigned set = (value << y);
  word &= ~clear;
  word |= set;
  bits[x] = word;
  (void) size;
}

#endif

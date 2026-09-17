#include "allocate.h"
#include "bits.h"
#include "internal.h"

bits *kissat_new_bits(kissat *solver, size_t size) {
  const unsigned words = kissat_bits_size_in_words(size);
  bits *res;
  CALLOC(res, words);
  return res;
}

void kissat_delete_bits(kissat *solver, bits *bits, size_t size) {
  const unsigned words = kissat_bits_size_in_words(size);
  DEALLOC(bits, words);
}

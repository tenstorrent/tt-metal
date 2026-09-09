#ifndef _random_h_INCLUDED
#define _random_h_INCLUDED

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

static inline uint64_t random64 (uint64_t *random) {
  uint64_t res = *random, next = res;
  next *= 6364136223846793005ul;
  next += 1442695040888963407ul;
  *random = next;
  return res;
}

static inline unsigned random32 (uint64_t *random) {
  return random64 (random) >> 32;
}

static inline bool random_bit (uint64_t *random) {
  return random32 (random) & 1;
}

static inline size_t random_modulo (uint64_t *random, size_t mod) {
  assert (mod);
  size_t tmp = random64 (random);
  size_t res = tmp % mod;
  assert (res < mod);
  return res;
}

static inline double random_double (uint64_t *random) {
  return random32 (random) / 4294967296.0;
}

#endif

#include "utilities.h"

#include <assert.h>
#include <math.h>

double logn (uint64_t count) {
  assert (count > 0);
  double res = log10 (count + 9);
  assert (res >= 1);
  return res;
}

double nlogn (uint64_t count) { return count * logn (count); }

double nlog2n (uint64_t count) {
  double f = logn (count);
  return count * f * f;
}

double nlog3n (uint64_t count) {
  double f = logn (count);
  return count * f * f * f;
}

double nlog4n (uint64_t count) {
  double f = logn (count);
  return count * f * f * f * f;
}

unsigned gcd (unsigned a, unsigned b) {
  while (b) {
    unsigned r = a % b;
    a = b, b = r;
  }
  return a;
}

unsigned log2ceil (unsigned n) {
  assert (n);
  return 32 - __builtin_clz (n - 1);
}

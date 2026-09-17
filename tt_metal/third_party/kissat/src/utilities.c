#include "utilities.h"

#include <string.h>

bool kissat_has_suffix(const char *str, const char *suffix) {
  const char *p = str;
  while (*p) {
    p++;
  }
  const char *q = suffix;
  while (*q) {
    q++;
  }
  while (p > str && q > suffix)
    if (*--p != *--q) {
      return false;
    }
  return q == suffix;
}

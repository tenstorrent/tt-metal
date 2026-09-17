#ifndef _parse_h_INCLUDED
#define _parse_h_INCLUDED

#include "file.h"

enum strictness {
  RELAXED_PARSING = 0,
  NORMAL_PARSING = 1,
  PEDANTIC_PARSING = 2,
};

typedef enum strictness strictness;

struct kissat;

const char *kissat_parse_dimacs(struct kissat *, strictness, file *,
      uint64_t *linenoptr, int *max_var_ptr);

#endif

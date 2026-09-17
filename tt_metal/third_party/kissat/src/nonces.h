#ifndef _nonces_h_INCLUDED
#define _nonces_h_INCLUDED

#include "stack.h"

#include <stdint.h>

typedef STACK(uint64_t)
nonces;

struct kissat;

void kissat_init_nonces(struct kissat *);

#endif

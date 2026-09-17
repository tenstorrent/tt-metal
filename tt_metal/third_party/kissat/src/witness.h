#ifndef _witness_h_INCLUDED
#define _witness_h_INCLUDED

#include <stdbool.h>

struct kissat;

void kissat_print_witness(struct kissat *, int max_var, bool partial);

#endif

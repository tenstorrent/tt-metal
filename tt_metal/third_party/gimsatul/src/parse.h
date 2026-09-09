#ifndef _parse_h_INCLUDED
#define _parse_h_INCLUDED

struct options;
struct ruler;

void parse_dimacs_header (struct options *options, int *variables_ptr,
                          int *clauses_ptr);

void parse_dimacs_body (struct ruler *ruler, int variables, int expected);

#endif

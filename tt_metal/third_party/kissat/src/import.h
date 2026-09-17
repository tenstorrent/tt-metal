#ifndef _import_h_INLCUDED
#define _import_h_INLCUDED

struct kissat;

unsigned kissat_import_literal(struct kissat *solver, int lit);

int kissat_external_representative(struct kissat *solver, int elit);

#endif

#ifndef _report_h_INCLUDED
#define _report_h_INCLUDED

#include <stdbool.h>

#ifdef QUIET

#define REPORT(...) do { } while (0)

#else

struct kissat;

void kissat_report(struct kissat *, bool verbose, char type);

#define REPORT(LEVEL,TYPE) \
  kissat_report (solver, (LEVEL), (TYPE))

#endif

#endif

#include "print.h"
#include "terminate.h"

#ifndef QUIET

void kissat_report_termination(kissat *solver, const char *name,
      const char *file, long lineno, const char *fun) {
  kissat_very_verbose(solver, "%s:%ld: %s: 'TERMINATED (%s)' triggered",
        file, lineno, fun, name);
}

#else
int kissat_terminate_dummy_to_avoid_warning;
#endif

#ifndef NOPTIONS

#include "config.h"
#include "kissat.h"
#include "options.h"

#include <stdio.h>
#include <string.h>

int kissat_has_configuration(const char *name) {
  if (!strcmp(name, "basic")) {
    return 1;
  }
  if (!strcmp(name, "default")) {
    return 1;
  }
  if (!strcmp(name, "plain")) {
    return 1;
  }
  if (!strcmp(name, "sat")) {
    return 1;
  }
  if (!strcmp(name, "unsat")) {
    return 1;
  }
  return 0;
}

void kissat_configuration_usage(void) {
#define FMT "  --%-8s %s"
  printf(FMT "\n", "basic", "basic CDCL solving "
        "('--plain' but no restarts, minimize, reduce)");
  printf(FMT "\n", "default", "default configuration");
  printf(FMT "\n", "plain",
        "plain CDCL solving without advanced techniques");
  printf(FMT " ('--target=%d --restartint=%d')\n", "sat",
        "target satisfiable instances", (int) TARGET_SAT,
        (int) RESTARTINT_SAT);
  printf(FMT " ('--stable=%d')\n", "unsat", "target unsatisfiable instances",
        (int) STABLE_UNSAT);
}

static void set_plain_options(kissat *solver) {
  kissat_set_option(solver, "bumpreasons", 0);
  kissat_set_option(solver, "chrono", 0);
  kissat_set_option(solver, "compact", 0);
  kissat_set_option(solver, "eagersubsume", 0);
  kissat_set_option(solver, "otfs", 0);
  kissat_set_option(solver, "rephase", 0);
  kissat_set_option(solver, "restartreusetrail", 0);
  kissat_set_option(solver, "simplify", 0);
  kissat_set_option(solver, "stable", 0);
  kissat_set_option(solver, "tumble", 0);
}

int kissat_set_configuration(kissat *solver, const char *name) {
  if (!strcmp(name, "basic")) {
    set_plain_options(solver);
    kissat_set_option(solver, "restart", 0);
    kissat_set_option(solver, "reduce", 0);
    kissat_set_option(solver, "minimize", 0);
    return 1;
  }
  if (!strcmp(name, "default")) {
    return 1;
  }
  if (!strcmp(name, "plain")) {
    set_plain_options(solver);
    return 1;
  }
  if (!strcmp(name, "sat")) {
    kissat_set_option(solver, "target", TARGET_SAT);
    kissat_set_option(solver, "restartint", RESTARTINT_SAT);
    return 1;
  }
  if (!strcmp(name, "unsat")) {
    kissat_set_option(solver, "stable", STABLE_UNSAT);
    return 1;
  }
  return 0;
}

#else
int kissat_config_dummy_to_avoid_warning;
#endif

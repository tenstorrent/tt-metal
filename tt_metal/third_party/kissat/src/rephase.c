#include "autarky.h"
#include "backtrack.h"
#include "decide.h"
#include "internal.h"
#include "logging.h"
#include "print.h"
#include "rephase.h"
#include "report.h"
#include "terminate.h"
#include "walk.h"

#include <inttypes.h>
#include <string.h>

void kissat_reset_best_assigned(kissat *solver) {
  if (!solver->best_assigned) {
    return;
  }
  kissat_extremely_verbose(solver,
        "resetting best assigned trail height %u to 0",
        solver->best_assigned);
  solver->best_assigned = 0;
}

void kissat_reset_target_assigned(kissat *solver) {
  if (!solver->target_assigned) {
    return;
  }
  kissat_extremely_verbose(solver,
        "resetting target assigned trail height %u to 0",
        solver->target_assigned);
  solver->target_assigned = 0;
}

bool kissat_rephasing(kissat *solver) {
  if (!solver->enabled.rephase) {
    return false;
  }
  if (!solver->stable) {
    return false;
  }
  return CONFLICTS > solver->limits.rephase.conflicts;
}

static char rephase_best(kissat *solver) {
  const value *const best = solver->phases.best;
  const value *const end_of_best = best + VARS;
  value const *b;

  value *const saved = solver->phases.saved;
  value *s;

  value tmp;

  for (s = saved, b = best; b != end_of_best; s++, b++)
    if ((tmp = *b)) {
      *s = tmp;
    }

  INC(rephased_best);

  return 'B';
}

static char rephase_original(kissat *solver) {
  assert(GET_OPTION(rephaseoriginal));
  const value initial_phase = INITIAL_PHASE;
  for (all_phases(saved, p)) {
    *p = initial_phase;
  }
  INC(rephased_original);
  return 'O';
}

static char rephase_inverted(kissat *solver) {
  assert(GET_OPTION(rephaseinverted));
  const value inverted_initial_phase = -INITIAL_PHASE;
  for (all_phases(saved, p)) {
    *p = inverted_initial_phase;
  }
  INC(rephased_inverted);
  return 'I';
}

static char rephase_walking(kissat *solver) {
  assert(GET_OPTION(rephasewalking));
  assert(kissat_walking(solver));
  STOP(rephase);
  kissat_walk(solver);
  START(rephase);
  INC(rephased_walking);
  return 'W';
}

static char reset_phases(kissat *solver) {
  uint64_t count = solver->rephased.count++;

  const bool inverted = GET_OPTION(rephaseinverted);
  const bool original = GET_OPTION(rephaseoriginal);
  const bool best = GET_OPTION(rephasebest);
  const bool walking = GET_OPTION(rephasewalking) && kissat_walking(solver);

  uint64_t prefix = inverted + original;
  prefix *= GET_OPTION(rephaseprefix);

  char (*functions[6])(kissat *);
  uint64_t candidates = 0;

#define PUSH1(NAME) \
  if (NAME) \
    { \
      assert (candidates < sizeof functions / sizeof *functions); \
      functions[candidates++] = rephase_ ## NAME; \
    }

#define PUSH3(NAME) \
  if (NAME) \
    { \
      PUSH1 (best); \
      PUSH1 (walking); \
      PUSH1 (NAME); \
    }

  if (count < prefix) {
    PUSH1(original);
    PUSH1(inverted);
  } else {
    PUSH3(original);
    PUSH3(inverted);
    count -= prefix;
  }

  if (!candidates) {
    PUSH1(best);
    PUSH1(walking);
  }

  char type;

  // Since 'enabled.rephase' is true one of the rephase methods is enabled.
  // However 'rephasewalking' could be the only one and the derived 'walking'
  // depends also on the size of the formula, i.e., 'kissat_walking' could
  // return 'false'.  As a consequence there is no candidate if only the
  // 'rephasewalking' is true but the formula is too big.
  //
  if (candidates) {
    const uint64_t select = count % candidates;
    type = functions[select](solver);
  } else {
    type = rephase_best(solver);
  }

#ifndef QUIET
  char const *type_as_string = 0;
  switch (type) {
#define REPHASE(NAME, TYPE, INDEX) \
      case TYPE: \
        type_as_string = #NAME; \
	break;
    REPHASES
#undef REPHASE
  }
  assert(type_as_string);
  kissat_phase(solver, "rephase", GET(rephased),
        "%s phases in %s search mode",
        type_as_string, solver->stable ? "stable" : "focused");
#endif
  kissat_extremely_verbose(solver, "remembering last rephase type '%c' at "
        "%s conflicts", type, FORMAT_COUNT(CONFLICTS));
  solver->rephased.last = type;
  LOG("copying saved phases as target phases");
  memcpy(solver->phases.target, solver->phases.saved, VARS);
  UPDATE_CONFLICT_LIMIT(rephase, rephased, NLOGNLOGNLOGN, false);
  kissat_reset_target_assigned(solver);
  if (type == 'B') {
    kissat_reset_best_assigned(solver);
  }
  return type;
}

void kissat_rephase(kissat *solver) {
  kissat_backtrack_propagate_and_flush_trail(solver);
  assert(!solver->inconsistent);
  kissat_autarky(solver, 'a');
  if (TERMINATED(rephase_terminated_1)) {
    return;
  }
  START(rephase);
  INC(rephased);
#ifndef QUIET
  const char type =
#endif
        reset_phases(solver);
  REPORT(0, type);
  STOP(rephase);
  if (TERMINATED(rephase_terminated_2)) {
    return;
  }
  kissat_autarky(solver, 'z');
}

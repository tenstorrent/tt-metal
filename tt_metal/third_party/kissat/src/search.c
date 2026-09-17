#include "analyze.h"
#include "assume.h"
#include "bump.h"
#include "decide.h"
#include "eliminate.h"
#include "internal.h"
#include "logging.h"
#include "mode.h"
#include "print.h"
#include "probe.h"
#include "propsearch.h"
#include "search.h"
#include "reduce.h"
#include "reluctant.h"
#include "report.h"
#include "restart.h"
#include "restore.h"
#include "terminate.h"
#include "trail.h"
#include "walk.h"

#include <inttypes.h>
#include <string.h>

static void start_search(kissat *solver) {
  START(search);
  INC(searches);

#define SAVE(X) solver->search_start.X = solver->statistics.X;
  SAVE_AT_SEARCH_START
#undef SAVE

  solver->extended = false;
  if (solver->failed_early) {
    solver->failed_early = false;
  } else {
    CLEAR_STACK(solver->failed);
  }

  if (solver->restore) {
    REPORT(0, '+');
    kissat_restore_clauses(solver);
    REPORT(0, 'r');
    if (solver->inconsistent) {
      return;
    }
  } else {
    REPORT(0, '*');
  }

  bool stable = (GET_OPTION(stable) == 2);

  solver->stable = stable;
  kissat_phase(solver, "search", GET(searches),
        "initializing %s search after %" PRIu64 " conflicts",
        (stable ? "stable" : "focus"), CONFLICTS);

  solver->averages[0].initialized = false;
  solver->averages[1].initialized = false;

  kissat_init_averages(solver, &AVERAGES);

  memset(&solver->delays, 0, sizeof(delays));

  if (solver->stable) {
    kissat_init_reluctant(solver);
    kissat_update_scores(solver);
  }

  kissat_init_limits(solver);

  unsigned seed = GET_OPTION(seed);
  solver->random = seed;
  LOG("initialized random number generator with seed %u", seed);

  const unsigned eagersubsume = GET_OPTION(eagersubsume);
  if (eagersubsume && !solver->clueue.elements) {
    kissat_init_clueue(solver, &solver->clueue, eagersubsume);
  }
#ifndef QUIET
  limits *limits = &solver->limits;
  limited *limited = &solver->limited;
  if (!limited->conflicts && !limited->decisions) {
    kissat_very_verbose(solver, "starting unlimited search");
  } else if (limited->conflicts && !limited->decisions)
    kissat_very_verbose(solver,
          "starting search with conflicts limited to %" PRIu64,
          limits->conflicts);
  else if (!limited->conflicts && limited->decisions)
    kissat_very_verbose(solver,
          "starting search with decisions limited to %" PRIu64,
          limits->decisions);
  else
    kissat_very_verbose(solver,
          "starting search with decisions limited to %" PRIu64
          " and conflicts limited to %" PRIu64,
          limits->decisions, limits->conflicts);
  if (stable) {
    START(stable);
    REPORT(0, '[');
  } else {
    START(focused);
    REPORT(0, '{');
  }
#endif
}

static void stop_search(kissat *solver, int res) {
  if (solver->limited.conflicts) {
    LOG("reset conflict limit");
    solver->limited.conflicts = false;
  }

  if (solver->limited.decisions) {
    LOG("reset decision limit");
    solver->limited.decisions = false;
  }

  if (solver->termination.flagged) {
    kissat_very_verbose(solver, "termination forced externally");
    solver->termination.flagged = 0;
  }

  kissat_reset_assumptions(solver);

#ifndef QUIET
  LOG("search result %d", res);
  if (solver->stable) {
    REPORT(0, ']');
    STOP(stable);
    solver->stable = false;
  } else {
    REPORT(0, '}');
    STOP(focused);
  }
  char type = (res == 10 ? '1' : res == 20 ? '0' : '?');
  REPORT(0, type);
#else
  (void) res;
  solver->stable = false;
#endif

  STOP(search);
}

static void iterate(kissat *solver) {
  assert(solver->iterating);
  solver->iterating = false;
  REPORT(0, 'i');
}

static bool conflict_limit_hit(kissat *solver) {
  if (!solver->limited.conflicts) {
    return false;
  }
  if (solver->limits.conflicts > solver->statistics.conflicts) {
    return false;
  }
  kissat_very_verbose(solver, "conflict limit %" PRIu64
        " hit after %" PRIu64 " conflicts",
        solver->limits.conflicts,
        solver->statistics.conflicts);
  return true;
}

static bool decision_limit_hit(kissat *solver) {
  if (!solver->limited.decisions) {
    return false;
  }
  if (solver->limits.decisions > solver->statistics.decisions) {
    return false;
  }
  kissat_very_verbose(solver, "decision limit %" PRIu64
        " hit after %" PRIu64 " decisions",
        solver->limits.decisions,
        solver->statistics.decisions);
  return true;
}

int kissat_search(kissat *solver) {
  start_search(solver);
  int res = kissat_propagate_assumptions(solver);
  if (!res) {
    res = kissat_walk_initially(solver);
  }
  CLEAR_STACK(solver->assumption_implied);
  while (!res) {
    clause *conflict = kissat_search_propagate(solver);
    if (conflict) {
      res = kissat_analyze(solver, conflict);
    } else if (solver->iterating) {
      iterate(solver);
    } else if (!solver->unassigned) {
      if (kissat_assuming(solver)) {
        res = kissat_assign_assumption(solver);
      } else {
        res = 10;
      }
    } else if (TERMINATED(search_terminated_1)) {
      break;
    } else if (conflict_limit_hit(solver)) {
      break;
    } else if (kissat_reducing(solver)) {
      res = kissat_reduce(solver);
    } else if (kissat_switching_search_mode(solver)) {
      kissat_switch_search_mode(solver);
    } else if (kissat_restarting(solver)) {
      kissat_restart(solver);
    } else if (kissat_rephasing(solver)) {
      res = kissat_propagate_assumptions(solver);
      if (!res) {
        kissat_rephase(solver);
      }
      CLEAR_STACK(solver->assumption_implied);
    } else if (kissat_eliminating(solver)) {
      res = kissat_eliminate(solver);
    } else if (kissat_probing(solver)) {
      res = kissat_probe(solver);
    } else if (decision_limit_hit(solver)) {
      break;
    } else if (kissat_assuming(solver)) {
      res = kissat_assign_assumption(solver);
    } else {
      kissat_decide(solver);
    }
  }
  stop_search(solver, res);
  return res;
}

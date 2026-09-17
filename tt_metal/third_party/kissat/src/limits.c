#include "internal.h"
#include "logging.h"
#include "print.h"
#include "reduce.h"
#include "rephase.h"
#include "resources.h"
#include "restart.h"

#include <inttypes.h>
#include <math.h>

changes kissat_changes(kissat *solver) {
  changes res;
  res.variables.units = solver->statistics.units;
  res.variables.added = solver->statistics.variables_added;
  res.variables.removed = solver->statistics.variables_removed;
  res.eliminate.additional_clauses =
        solver->bounds.eliminate.additional_clauses;
  return res;
}

bool kissat_changed(changes b, changes a) {
  if (a.variables.added != b.variables.added) {
    return true;
  }
  if (a.variables.units != b.variables.units) {
    return true;
  }
  if (a.variables.removed != b.variables.removed) {
    return true;
  }
  if (a.eliminate.additional_clauses != b.eliminate.additional_clauses) {
    return true;
  }
  return false;
}

double kissat_logn(uint64_t count) {
  assert(count > 0);
  const double res = log10(count + 9);
  assert(res >= 1);
  return res;
}

static double kissat_lognlogn(uint64_t count) {
  assert(count > 0);
  const double tmp = log10(count + 9);
  const double res = tmp * tmp;
  assert(res >= 1);
  return res;
}

static double kissat_lognlognlogn(uint64_t count) {
  assert(count > 0);
  const double tmp = log10(count + 9);
  const double res = tmp * tmp * tmp;
  assert(res >= 1);
  return res;
}

double kissat_ndivlogn(uint64_t count) {
  assert(count > 0);
  const double div = kissat_logn(count);
  assert(div > 0);
  const double res = count / div;
  assert(res >= 1);
  return res;
}

double kissat_sqrt(uint64_t count) {
  assert(count > 0);
  const double res = sqrt(count);
  assert(res >= 1);
  return res;
}

double kissat_linear(uint64_t count) {
  assert(count > 0);
  const double res = count;
  assert(res >= 1);
  return res;
}

double kissat_nlogn(uint64_t count) {
  assert(count > 0);
  const double factor = kissat_logn(count);
  assert(factor >= 1);
  const double res = count * factor;
  assert(res >= 1);
  return res;
}

double kissat_nlognlogn(uint64_t count) {
  assert(count > 0);
  const double factor = kissat_lognlogn(count);
  const double res = count * factor;
  assert(res >= 1);
  return res;
}

double kissat_nlognlognlogn(uint64_t count) {
  assert(count > 0);
  const double factor = kissat_lognlognlogn(count);
  assert(factor >= 1);
  const double res = count * factor;
  assert(res >= 1);
  return res;
}

double kissat_quadratic(uint64_t count) {
  assert(count > 0);
  const double res = count * count;
  assert(res >= 1);
  return res;
}

uint64_t kissat_scale_delta(kissat *solver, const char *pretty,
      uint64_t delta) {
  const uint64_t C = IRREDUNDANT_CLAUSES;
  const double f = kissat_logn(C + 1);
  assert(f >= 1);
  const double ff = f * f;
  assert(ff >= 1);
  uint64_t scaled = ff * delta;
  assert(delta <= scaled);
// *INDENT-OFF*
  kissat_very_verbose (solver,
    "scaled %s delta %" PRIu64
    " = %g * %" PRIu64
    " = %g^2 * %" PRIu64
    " = log10^2(%" PRIu64 ") * %" PRIu64,
    pretty, scaled, ff, delta, f, delta, C, delta);
// *INDENT-ON*
  (void) pretty;
  return scaled;
}

uint64_t kissat_scale_limit(kissat *solver,
      const char *pretty, uint64_t count, int base) {
  assert(base >= 0);
  assert(count > 0);
  const double f = kissat_logn(count);
  assert(f >= 1);
  uint64_t scaled = f * base;
  kissat_very_verbose(solver,
        "scaled %s limit %" PRIu64 " = "
        "log10 (%" PRIu64 ") * %d = %g * %d",
        pretty, scaled, count, base, f, base);
  (void) solver;
  (void) pretty;
  return scaled;
}

static void init_enabled(kissat *solver) {
  bool probe;
  if (!GET_OPTION(simplify)) {
    probe = false;
  } else if (!GET_OPTION(probe)) {
    probe = false;
  } else if (GET_OPTION(substitute)) {
    probe = true;
  } else if (GET_OPTION(failed)) {
    probe = true;
  } else if (GET_OPTION(transitive)) {
    probe = true;
  } else if (GET_OPTION(sweep)) {
    probe = true;
  } else if (GET_OPTION(vivify)) {
    probe = true;
  } else {
    probe = false;
  }
  kissat_very_verbose(solver, "probing %sabled", probe ? "en" : "dis");
  solver->enabled.probe = probe;

  bool eliminate;
  if (!GET_OPTION(simplify)) {
    eliminate = false;
  } else if (!GET_OPTION(eliminate)) {
    eliminate = false;
  } else {
    eliminate = true;
  }
  kissat_very_verbose(solver, "eliminate %sabled", eliminate ? "en" : "dis");
  solver->enabled.eliminate = eliminate;

  bool autarky;
  if (!GET_OPTION(simplify)) {
    autarky = false;
  } else if (!GET_OPTION(autarky)) {
    autarky = false;
  } else {
    autarky = true;
  }
  kissat_very_verbose(solver, "autarky %sabled", autarky ? "en" : "dis");
  solver->enabled.autarky = autarky;

  bool rephase;
  if (!GET_OPTION(rephase)) {
    rephase = false;
  } else if (!GET_OPTION(phasesaving)) {
    rephase = false;
  } else if (
#define REPHASE(NAME,TYPE,INDEX) \
    GET_OPTION (rephase ## NAME) ||
        REPHASES 0)
#undef REPHASE
    rephase = true;
  else {
    rephase = false;
  }
  kissat_very_verbose(solver, "rephase %sabled", rephase ? "en" : "dis");
  solver->enabled.rephase = rephase;
}

static void init_mode_limit(kissat *solver) {
  limits *limits = &solver->limits;

  if (GET_OPTION(stable) == 1) {
    assert(!solver->stable);

    const uint64_t conflicts_delta = GET_OPTION(modeconflicts);
    const uint64_t conflicts_limit = CONFLICTS + conflicts_delta;
    limits->mode.conflicts = conflicts_limit;

    const uint64_t ticks_delta = GET_OPTION(modeticks);
    const uint64_t ticks_limit = solver->statistics.search_ticks + ticks_delta;
    limits->mode.ticks = ticks_limit;

    kissat_very_verbose(solver, "initial stable mode switching limit "
          "at %s conflicts and %s ticks",
          FORMAT_COUNT(conflicts_limit),
          FORMAT_COUNT(ticks_limit));

    solver->mode.ticks = solver->statistics.search_ticks;
#ifndef QUIET
    solver->mode.conflicts = CONFLICTS;
#ifdef METRICS
    solver->mode.propagations = solver->statistics.search_propagations;
#endif
// *INDENT-OFF*
      solver->mode.entered = kissat_process_time ();
      kissat_very_verbose (solver,
        "starting focused mode at %.2f seconds "
        "(%" PRIu64 " conflicts, %" PRIu64 " ticks"
#ifdef METRICS
	", %" PRIu64 " propagations, %" PRIu64 " visits"
#endif
	")",
        solver->mode.entered, solver->mode.conflicts, solver->mode.ticks
#ifdef METRICS
        , solver->mode.propagations, solver->mode.visits
#endif
	);
// *INDENT-ON*
#endif
  } else
    kissat_very_verbose(solver,
          "no need to set mode limit (only %s mode enabled)",
          GET_OPTION(stable) ? "stable" : "focused");
}

static void common_limits(kissat *solver) {
  init_mode_limit(solver);

  if (solver->enabled.eliminate) {
    solver->bounds.eliminate.max_bound_completed = 0;
    solver->bounds.eliminate.additional_clauses = 0;
    kissat_very_verbose(solver, "reset elimination bound to zero");
  }
}

void kissat_init_limits(kissat *solver) {
  init_enabled(solver);

  limits *limits = &solver->limits;

  if (GET_OPTION(reduce)) {
    INIT_CONFLICT_LIMIT(reduce, false);
  }

  if (solver->enabled.rephase) {
    INIT_CONFLICT_LIMIT(rephase, false);
  }

  if (!solver->stable) {
    kissat_new_focused_restart_limit(solver);
  }

  common_limits(solver);

  if (solver->enabled.eliminate) {
    INIT_CONFLICT_LIMIT(eliminate, true);
  }

  if (solver->enabled.probe) {
    INIT_CONFLICT_LIMIT(probe, true);
  }
}

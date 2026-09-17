#include "inline.h"
#include "inlineheap.h"
#include "inlinequeue.h"
#include "print.h"
#include "reluctant.h"
#include "rephase.h"
#include "report.h"
#include "restart.h"
#include "resources.h"

#include <inttypes.h>

static void new_mode_limit(kissat *solver) {
  kissat_init_averages(solver, &AVERAGES);

  limits *limits = &solver->limits;
  search_start *search_start = &solver->search_start;
  statistics *statistics = &solver->statistics;

  assert(GET_OPTION(stable) == 1);

  if (limits->mode.conflicts) {
    assert(solver->stable);
    limits->mode.interval =
          statistics->search_ticks - search_start->search_ticks;
    limits->mode.conflicts = 0;
  }

  const uint64_t interval = limits->mode.interval;
  assert(interval > 0);

  const uint64_t count =
        (statistics->switched_modes - search_start->switched_modes + 1) / 2;
  const uint64_t scaled = interval * kissat_quadratic(count);
  limits->mode.ticks = statistics->search_ticks + scaled;
#ifndef QUIET
  if (solver->stable)
    kissat_phase(solver, "stable", GET(stable_modes),
          "new focused mode switching limit of %s after %s ticks",
          FORMAT_COUNT(limits->mode.ticks), FORMAT_COUNT(scaled));
  else
    kissat_phase(solver, "focus", GET(focused_modes),
          "new stable mode switching limit of %s after %s ticks",
          FORMAT_COUNT(limits->mode.ticks), FORMAT_COUNT(scaled));

  solver->mode.conflicts = statistics->conflicts;
#ifdef METRICS
  solver->mode.propagations = statistics->search_propagations;
#endif
#endif
  solver->mode.ticks = statistics->search_ticks;
}

static void report_switching_from_mode(kissat *solver) {
#ifndef QUIET
  if (kissat_verbosity(solver) < 2) {
    return;
  }

  const double current_time = kissat_process_time();
  const double delta_time = current_time - solver->mode.entered;

  statistics *statistics = &solver->statistics;

  const uint64_t delta_conflicts =
        statistics->conflicts - solver->mode.conflicts;

  const uint64_t delta_ticks = statistics->search_ticks - solver->mode.ticks;
#ifdef METRICS
  const uint64_t delta_propagations =
        statistics->search_propagations - solver->mode.propagations;
#endif
  solver->mode.entered = current_time;

  // *INDENT-OFF*
  kissat_very_verbose (solver, "%s mode took %.2f seconds "
    "(%s conflicts, %s ticks"
#ifdef METRICS
    ", %s propagations"
#endif
    ")", solver->stable ? "stable" : "focused",
    delta_time, FORMAT_COUNT (delta_conflicts), FORMAT_COUNT (delta_ticks)
#ifdef METRICS
    , FORMAT_COUNT (delta_propagations)
#endif
    );
  // *INDENT-ON*
#else
  (void) solver;
#endif
}

static void switch_to_focused_mode(kissat *solver) {
  assert(solver->stable);
  report_switching_from_mode(solver);
  REPORT(0, ']');
  STOP(stable);
  INC(focused_modes);
  kissat_phase(solver, "focus", GET(focused_modes),
        "switching to focused mode after %s conflicts",
        FORMAT_COUNT(CONFLICTS));
  solver->stable = false;
  new_mode_limit(solver);
  START(focused);
  REPORT(0, '{');
  kissat_reset_queue(solver);
  kissat_new_focused_restart_limit(solver);
}

void kissat_update_scores(kissat *solver) {
  heap *scores = &solver->scores;
  for (all_variables(idx))
    if (ACTIVE(idx) && !kissat_heap_contains(scores, idx)) {
      kissat_push_heap(solver, scores, idx);
    }
}

static void switch_to_stable_mode(kissat *solver) {
  assert(!solver->stable);
  report_switching_from_mode(solver);
  REPORT(0, '}');
  STOP(focused);
  INC(stable_modes);
  solver->stable = true;
  kissat_phase(solver, "stable", GET(stable_modes),
        "switched to stable mode after %" PRIu64
        " conflicts", CONFLICTS);
  new_mode_limit(solver);
  START(stable);
  REPORT(0, '[');
  kissat_init_reluctant(solver);
  kissat_update_scores(solver);
}

bool kissat_switching_search_mode(kissat *solver) {
  assert(!solver->inconsistent);

  if (GET_OPTION(stable) != 1) {
    return false;
  }

  limits *limits = &solver->limits;
  statistics *statistics = &solver->statistics;

  if (limits->mode.conflicts) {
    assert(!solver->stable);
    assert(!statistics->switched_modes || GET(searches));
    if (statistics->conflicts >= limits->mode.conflicts) {
      return true;
    }
  }

  return statistics->search_ticks >= limits->mode.ticks;
}

void kissat_switch_search_mode(kissat *solver) {
  assert(kissat_switching_search_mode(solver));

  INC(switched_modes);

  if (solver->stable) {
    switch_to_focused_mode(solver);
  } else {
    switch_to_stable_mode(solver);
  }

  assert(!solver->limits.mode.conflicts);

  solver->averages[solver->stable].saved_decisions = DECISIONS;
}

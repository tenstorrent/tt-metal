#include "search.h"
#include "analyze.h"
#include "backtrack.h"
#include "decide.h"
#include "export.h"
#include "import.h"
#include "message.h"
#include "mode.h"
#include "probe.h"
#include "propagate.h"
#include "reduce.h"
#include "rephase.h"
#include "report.h"
#include "restart.h"
#include "ruler.h"
#include "simplify.h"
#include "walk.h"

#include <assert.h>
#include <inttypes.h>

static bool iterating (struct ring *ring) {
  struct ring_units *units = &ring->ring_units;
  return units->iterate < units->end;
}

void iterate (struct ring *ring) {
  struct ring_units *units = &ring->ring_units;
  if (units->iterate < units->end) {
#ifndef QUIET
    size_t new_units = units->end - units->iterate;
    very_verbose (ring, "iterating %zu units", new_units);
    int report_level = (ring->iterating <= 0);
    verbose_report (ring, 'i', report_level);
#endif
    export_units (ring);
    units->iterate = units->end;
  }
  ring->iterating = 0;
}

bool backtrack_propagate_iterate (struct ring *ring) {
  assert (!ring->inconsistent);
  if (ring->level)
    backtrack (ring, 0);
  ring->trail.propagate = ring->trail.begin;
  if (ring_propagate (ring, true, 0)) {
    set_inconsistent (ring,
                      "failed propagation after root-level backtracking");
    return false;
  }
  iterate (ring);
  assert (!ring->inconsistent);
  return true;
}

static void start_search (struct ring *ring) {
  ring->stable = !ring->options.focus_initially;
#ifndef QUIET
  double t = START (ring, search);
  ring->last.mode.time = t;
#endif
  if (ring->stable) {
    report (ring, '[');
    START (ring, stable);
  } else {
    report (ring, '{');
    START (ring, focus);
  }
}

static void stop_search (struct ring *ring, int res) {
  if (ring->stable) {
    report (ring, ']');
    STOP (ring, stable);
  } else {
    report (ring, '}');
    STOP (ring, focus);
  }
  if (res == 10)
    report (ring, '1');
  else if (res == 20)
    report (ring, '0');
  else
    report (ring, '?');
  STOP (ring, search);
}

static bool conflict_limit_hit (struct ring *ring) {
  long limit = ring->limits.conflicts;
  if (limit < 0)
    return false;
  uint64_t conflicts = SEARCH_CONFLICTS;
  if (conflicts < (unsigned long) limit)
    return false;
  verbose (ring, "conflict limit %ld hit at %" PRIu64 " conflicts", limit,
           conflicts);
  set_terminate (ring->ruler, ring);
  return true;
}

bool terminate_ring (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
#ifdef NFASTPATH
  if (pthread_mutex_lock (&ruler->locks.terminate))
    fatal_error ("failed to acquire terminate lock");
#endif
  bool res = ruler->terminate;
#ifdef NFASTPATH
  if (pthread_mutex_unlock (&ruler->locks.terminate))
    fatal_error ("failed to release terminate lock");
#endif
  return res;
}

static bool walk_initially (struct ring *ring) {
  return !ring->statistics.walked && ring->ruler->options.walk_initially;
}

int search (struct ring *ring) {
  start_search (ring);
  int res = ring->inconsistent ? 20 : 0;
  while (!res) {
    struct watch *conflict = ring_propagate (ring, true, 0);
    if (conflict) {
      if (!analyze (ring, conflict))
        res = 20;
    } else if (!ring->unassigned)
      set_satisfied (ring), res = 10;
    else if (iterating (ring))
      iterate (ring);
    else if (terminate_ring (ring))
      break;
    else if (walk_initially (ring))
      local_search (ring);
    else if (conflict_limit_hit (ring))
      break;
    else if (reducing (ring))
      reduce (ring);
    else if (restarting (ring))
      restart (ring);
    else if (switching_mode (ring))
      switch_mode (ring);
    else if (rephasing (ring))
      rephase (ring);
    else if (probing (ring))
      res = probe (ring);
    else if (simplifying (ring))
      res = simplify_ring (ring);
    else if (!import_shared (ring))
      decide (ring);
    else if (ring->inconsistent)
      res = 20;
  }
  stop_search (ring, res);
  assert (ring->ruler->terminate); // Might break due to races.
  return res;
}

#include "solve.h"
#include "message.h"
#include "ruler.h"
#include "scale.h"
#include "search.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

static void *solve_routine (void *ptr) {
  struct ring *ring = ptr;
  int res = search (ring);
  assert (ring->status == res);
  (void) res;
  return ring;
}

static void start_running_ring (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
  assert (ruler->threads);
  pthread_t *thread = ruler->threads + ring->id;
  if (pthread_create (thread, 0, solve_routine, ring))
    fatal_error ("failed to create solving thread %u", ring->id);
#ifndef __APPLE__
  int sched_getcpu (void);
  message (ring, "ring %u on CPU %08x", ring->id, sched_getcpu ());
#endif
}

static void stop_running_ring (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
  assert (ruler->threads);
  pthread_t *thread = ruler->threads + ring->id;
  if (pthread_join (*thread, 0))
    fatal_error ("failed to join solving thread %u", ring->id);
}

static void set_ring_limits (struct ring *ring, long long conflicts) {
  if (ring->inconsistent)
    return;
  assert (!ring->stable);
  assert (!SEARCH_CONFLICTS);
  struct ring_limits *limits = &ring->limits;
  if (ring->options.portfolio) {
    very_verbose (ring, "determining portfolio options");

    unsigned rootid = sqrt (ring->id);
    if (!ring->id || rootid * rootid != ring->id) {
      ring->options.focus_initially = true;
      ring->options.switch_mode = true;
    } else if (ring->id & 1) {
      ring->options.focus_initially = false;
      ring->options.switch_mode = false;
    } else {
      ring->options.focus_initially = true;
      ring->options.switch_mode = false;
    }

    if (ring->id % 4)
      ring->options.target_phases = 2;
    else
      ring->options.target_phases = 1;

    if (!ring->options.force_phase) {
      if ((ring->id % 6) < 4)
        ring->options.phase = 1;
      else
        ring->options.phase = 0;
    }

    if (ring->options.switch_mode) {
      if (ring->options.focus_initially)
        verbose (ring, "portfolio: starting in focused mode");
      else
        verbose (ring, "portfolio: starting in stable mode");
    } else {
      if (ring->options.focus_initially)
        verbose (ring, "portfolio: only running in focused mode");
      else
        verbose (ring, "portfolio: only running in stable mode");
    }

    if (ring->options.phase)
      verbose (ring, "portfolio: initial 'true' decision phase");
    else
      verbose (ring, "portfolio: initial 'false' decision phase");

    if (ring->options.target_phases == 0)
      verbose (ring, "portfolio: chasing target phases disabled");
    else if (ring->options.target_phases == 1)
      verbose (ring, "portfolio: "
                     "chasing target phases in stable mode only");
    else {
      assert (ring->options.target_phases == 2);
      verbose (ring,
               "portfolio: "
               "chasing target phases both in stable and focused mode");
    }
  } else
    very_verbose (ring, "keeping global options");

  limits->mode = ring->options.switch_interval;
  if (ring->options.switch_mode)
    verbose (ring,
             "initial mode switching interval of %" PRIu64 " conflicts",
             limits->mode);

  if (ring->options.random_decisions) {
    limits->randec = ring->options.random_decision_interval;
    verbose (ring, "random decision interval of %" PRIu64 " conflicts",
             limits->randec);
  }

  limits->reduce = ring->options.reduce_interval;
  limits->restart = FOCUSED_RESTART_INTERVAL;
  limits->rephase = ring->options.rephase_interval;

  verbose (ring, "reduce interval of %" PRIu64 " conflicts",
           limits->reduce);
  verbose (ring, "restart interval of %" PRIu64 " conflicts",
           limits->restart);
  verbose (ring, "rephase interval of %" PRIu64 " conflicts",
           limits->rephase);

  {
    uint64_t interval = ring->options.probe_interval;
    uint64_t scaled = scale_interval (ring, "probe", interval);
    verbose (ring, "probe limit of %" PRIu64 " conflicts", scaled);
    limits->probe.conflicts = scaled;
  }

  if (!ring->id) {
    uint64_t interval = ring->options.simplify_interval;
    uint64_t scaled = scale_interval (ring, "simplify", interval);
    verbose (ring, "simplify limit of %" PRIu64 " conflicts", scaled);
    limits->simplify = scaled;
  }

  if (conflicts >= 0) {
    limits->conflicts = conflicts;
    verbose (ring, "conflict limit set to %lld conflicts", conflicts);
  }
}

struct ring *solve_rings (struct ruler *ruler) {
  if (ruler->terminate)
    return ruler->winner;
#ifndef QUIET
  double start_solving = START (ruler, solve);
#endif
  assert (!ruler->solving);
  ruler->solving = true;
  size_t threads = SIZE (ruler->rings);
  long long conflicts = ruler->options.conflicts;
  if (verbosity >= 0) {
    printf ("c\n");
    if (conflicts >= 0)
      printf ("c conflict limit %lld\n", conflicts);
    fflush (stdout);
  }
  for (all_rings (ring))
    set_ring_limits (ring, conflicts);
  message (0, 0);
  if (threads > 1) {
    for (all_rings (ring))
      ring->probe = ring->id * (ruler->compact / threads);

    message (0, "starting and running %zu ring threads", threads);

#define BARRIER(NAME) init_barrier (&ruler->barriers.NAME, #NAME, threads);
    BARRIERS
#undef BARRIER
    // clang-format off

      for (all_rings (ring))
	start_running_ring (ring);

      for (all_rings (ring))
	stop_running_ring (ring);

    // clang-format on
  } else {
    message (0, "running single ring in main thread");
    struct ring *ring = first_ring (ruler);
    (void) solve_routine (ring);
  }
  assert (ruler->solving);
  ruler->solving = false;
#ifndef QUIET
  double end_solving = STOP (ruler, solve);
  verbose (0, "finished solving using %zu threads in %.2f seconds", threads,
           end_solving - start_solving);
#endif
  return (struct ring *) ruler->winner;
}

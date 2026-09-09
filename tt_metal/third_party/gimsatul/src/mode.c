#include "mode.h"
#include "bump.h"
#include "decide.h"
#include "message.h"
#include "options.h"
#include "report.h"
#include "ring.h"
#include "utilities.h"

#include <inttypes.h>

#ifndef QUIET

static void report_mode_duration (struct ring *ring, double t,
                                  const char *type) {
  struct ring_last *l = &ring->last;
  verbose (ring,
           "%s mode took %.2f seconds "
           "(%" PRIu64 " conflicts, %" PRIu64 " ticks)",
           type, t - l->mode.time, SEARCH_CONFLICTS - l->mode.conflicts,
           SEARCH_TICKS - l->mode.ticks);
  l->mode.time = t;
  l->mode.conflicts = SEARCH_CONFLICTS;
  l->mode.ticks = SEARCH_TICKS;
}

#endif

static void switch_to_focused_mode (struct ring *ring) {
  assert (ring->stable);
  report (ring, ']');
#ifndef QUIET
  double t = STOP (ring, stable);
  report_mode_duration (ring, t, "stable");
#endif
  ring->stable = false;
  START (ring, focus);
  report (ring, '{');
  struct ring_limits *limits = &ring->limits;
  limits->restart = SEARCH_CONFLICTS + FOCUSED_RESTART_INTERVAL;
}

static void switch_to_stable_mode (struct ring *ring) {
  assert (!ring->stable);
  report (ring, '}');
#ifndef QUIET
  double t = STOP (ring, focus);
  report_mode_duration (ring, t, "focused");
#endif
  ring->stable = true;
  START (ring, stable);
  report (ring, '[');
  struct ring_limits *limits = &ring->limits;
  limits->restart = SEARCH_CONFLICTS + STABLE_RESTART_INTERVAL;
  ring->reluctant.u = ring->reluctant.v = 1;
}

bool switching_mode (struct ring *ring) {
  if (!ring->options.switch_mode)
    return false;
  struct ring_limits *l = &ring->limits;
  if (ring->statistics.switched)
    return SEARCH_TICKS > l->mode;
  else
    return SEARCH_CONFLICTS > l->mode;
}

void switch_mode (struct ring *ring) {
  struct ring_statistics *s = &ring->statistics;
  struct intervals *i = &ring->intervals;
  struct ring_limits *l = &ring->limits;
  if (!s->switched++) {
    i->mode = SEARCH_TICKS;
    verbose (ring, "determined mode switching ticks interval %" PRIu64,
             i->mode);
  }
  if (ring->stable) {
    switch_to_focused_mode (ring);
    reset_queue_search (&ring->queue);
  } else {
    switch_to_stable_mode (ring);
    rebuild_heap (ring);
  }
  uint64_t base = i->mode;
  uint64_t interval = base * nlog4n (s->switched / 2 + 1);
  l->mode = SEARCH_TICKS + interval;
  very_verbose (ring,
                "new mode switching limit at %" PRIu64 " after %" PRIu64
                " ticks",
                l->mode, interval);
  ring->last.decisions = SEARCH_DECISIONS;

  start_random_decision_sequence (ring);
}

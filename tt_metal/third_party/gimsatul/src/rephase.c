#include "rephase.h"
#include "backtrack.h"
#include "decide.h"
#include "message.h"
#include "report.h"
#include "ring.h"
#include "search.h"
#include "utilities.h"
#include "walk.h"

#include <inttypes.h>

static char rephase_walk (struct ring *ring) {
  local_search (ring);
  for (all_phases (p))
    p->target = p->saved;
  return 'W';
}

static char rephase_best (struct ring *ring) {
  for (all_phases (p))
    p->target = p->saved = p->best;
  return 'B';
}

static char rephase_inverted (struct ring *ring) {
  for (all_phases (p))
    p->target = p->saved = -initial_phase (ring);
  return 'I';
}

static char rephase_original (struct ring *ring) {
  for (all_phases (p))
    p->target = p->saved = initial_phase (ring);
  return 'O';
}

bool rephasing (struct ring *ring) {
  if (!ring->options.rephase)
    return false;
  return ring->stable && SEARCH_CONFLICTS > ring->limits.rephase;
}

static char (*schedule[]) (struct ring *) = {
    rephase_original, rephase_best, rephase_walk,
    rephase_inverted, rephase_best, rephase_walk,
};

void rephase (struct ring *ring) {
  if (!backtrack_propagate_iterate (ring))
    return;
  struct ring_statistics *statistics = &ring->statistics;
  struct ring_limits *limits = &ring->limits;
  uint64_t rephased = ++statistics->rephased;
  size_t size_schedule = sizeof schedule / sizeof *schedule;
  char type = schedule[rephased % size_schedule](ring);
  verbose (ring, "resetting number of target assigned %u", ring->target);
  ring->target = 0;
  if (type == 'B') {
    verbose (ring, "resetting number of best assigned %u", ring->best);
    ring->best = 0;
  }
  uint64_t base = ring->options.rephase_interval;
  uint64_t interval = base * nlog3n (rephased);
  limits->rephase = SEARCH_CONFLICTS + interval;
  very_verbose (
      ring, "new rephase limit of %" PRIu64 " after %" PRIu64 " conflicts",
      limits->rephase, interval);
  report (ring, type);
}

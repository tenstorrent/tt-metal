#include "restart.h"
#include "backtrack.h"
#include "message.h"
#include "options.h"
#include "report.h"
#include "ring.h"
#include "utilities.h"

#include <inttypes.h>

bool restarting (struct ring *ring) {
  if (!ring->level)
    return false;
  struct ring_limits *l = &ring->limits;
  if (SEARCH_CONFLICTS <= l->restart)
    return false;
  if (ring->stable)
    return true;
  l->restart = SEARCH_CONFLICTS;
  struct averages *a = ring->averages;
  double fast = a->glue.fast.value;
  double slow = a->glue.slow.value;
  double margin = slow * RESTART_MARGIN;
  extremely_verbose (ring,
                     "restart glue limit %g = "
                     "%g * %g (slow glue) %c %g (fast glue)",
                     margin, RESTART_MARGIN, slow,
                     margin == fast  ? '='
                     : margin > fast ? '>'
                                     : '<',
                     fast);
  return margin <= fast;
}

void restart (struct ring *ring) {
  struct ring_statistics *statistics = &ring->statistics;
  statistics->restarts++;
  very_verbose (ring, "restart %" PRIu64 " at %" PRIu64 " conflicts",
                statistics->restarts, SEARCH_CONFLICTS);
  update_best_and_target_phases (ring);
  backtrack (ring, 0);
  struct ring_limits *limits = &ring->limits;
  uint64_t interval;
  if (ring->stable) {
    struct reluctant *reluctant = &ring->reluctant;
    uint64_t u = reluctant->u, v = reluctant->v;
    if ((u & -u) == v)
      u++, v = 1;
    else
      v *= 2;
    interval = STABLE_RESTART_INTERVAL * v;
    if (interval > MAX_STABLE_RESTART_INTERVAL)
      interval = MAX_STABLE_RESTART_INTERVAL;
    reluctant->u = u, reluctant->v = v;
  } else
    interval = FOCUSED_RESTART_INTERVAL + logn (statistics->restarts) - 1;
  limits->restart = SEARCH_CONFLICTS + interval;
  very_verbose (
      ring, "new restart limit at %" PRIu64 " after %" PRIu64 " conflicts",
      limits->restart, interval);
  verbose_report (ring, 'r', 1);
}

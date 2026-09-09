#include "probe.h"
#include "assign.h"
#include "backtrack.h"
#include "fail.h"
#include "message.h"
#include "propagate.h"
#include "ring.h"
#include "scale.h"
#include "search.h"
#include "utilities.h"
#include "vivify.h"

#include <inttypes.h>

bool probing (struct ring *ring) {
  if (!ring->options.probe)
    return false;
  if (ring->statistics.reductions < ring->limits.probe.reductions)
    return false;
  return SEARCH_CONFLICTS > ring->limits.probe.conflicts;
}

int probe (struct ring *ring) {
  if (!backtrack_propagate_iterate (ring))
    return 20;
  assert (ring->size);
  assert (ring->options.probe);
  STOP_SEARCH_AND_START (probe);
  assert (ring->context == SEARCH_CONTEXT);
  ring->context = PROBING_CONTEXT;
  ring->statistics.probings++;
  failed_literal_probing (ring);
  vivify_clauses (ring);
  ring->context = SEARCH_CONTEXT;
  ring->last.probing = SEARCH_TICKS;
  struct ring_statistics *statistics = &ring->statistics;
  struct ring_limits *limits = &ring->limits;
  uint64_t base = ring->options.probe_interval;
  uint64_t interval = base * nlogn (statistics->probings);
  uint64_t scaled = scale_interval (ring, "probe", interval);
  limits->probe.conflicts = SEARCH_CONFLICTS + scaled;
  limits->probe.reductions = statistics->reductions + 1;
  very_verbose (
      ring, "new probe limit at %" PRIu64 " after %" PRIu64 " conflicts",
      limits->probe.conflicts, scaled);
  STOP_AND_START_SEARCH (probe);
  return ring->inconsistent ? 20 : 0;
}

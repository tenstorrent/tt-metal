#include "tiers.h"
#include "message.h"
#include "ring.h"
#include "utilities.h"

#include <inttypes.h>

static void compute_tier_limits (struct ring *ring, unsigned stable,
                                 unsigned *tier1_ptr, unsigned *tier2_ptr,
                                 uint64_t *bumped1_ptr,
                                 uint64_t *bumped2_ptr) {
  uint64_t total = ring->statistics.usage[stable].bumped;
  uint64_t limit1 = total * 0.5, limit2 = total * 0.9;
  unsigned tier1 = UINT_MAX, tier2 = UINT_MAX;
  uint64_t bumped = 0, bumped1 = 0, bumped2 = 0;
  unsigned tier_glue = 1;
  for (unsigned j = 0; j <= MAX_GLUE; j++) {
    bumped += ring->statistics.usage[stable].glue[j];
    if (tier_glue == 1 && bumped >= limit1) {
      tier1 = j;
      bumped1 = bumped;
      tier_glue = 2;
    }
    if (tier_glue == 2 && bumped >= limit2) {
      tier2 = j;
      bumped2 = bumped;
      break;
    }
  }
#ifndef QUIET
  const char *mode = stable ? "stable" : "focused";
  extremely_verbose (ring, "%s tier1 limit %u %.2f%%", mode, tier1,
                     percent (bumped1, total));
  extremely_verbose (ring, "%s tier2 limit %u %.2f%%", mode, tier2,
                     percent (bumped2, total));
#endif
  *tier1_ptr = tier1, *tier2_ptr = tier2;
  if (bumped1_ptr)
    *bumped1_ptr = bumped1;
  if (bumped2_ptr)
    *bumped2_ptr = bumped2;
}

void recalculate_tier_limits (struct ring *ring) {
  if (!ring->options.calculate_tiers)
    return;
  unsigned stable = ring->stable, tier1, tier2;
  compute_tier_limits (ring, stable, &tier1, &tier2, 0, 0);
  ring->tier1_glue_limit[stable] = tier1;
  ring->tier2_glue_limit[stable] = tier2;
  very_verbose (ring,
                "recalculated %s tier1 limit %u and "
                "tier2 limit %u after %" PRIu64 " conflicts",
                stable ? "stable" : "focused", tier1, tier2,
                SEARCH_CONFLICTS);
}

void print_tiers_bumped_statistics (struct ring *ring) {
  struct ring_statistics *s = &ring->statistics;
  uint64_t total_focused = s->usage[0].bumped;
  uint64_t total_stable = s->usage[1].bumped;
  uint64_t total = s->bumped, bumped1, bumped2;
  assert (total == total_stable + total_focused);
  unsigned tier1, tier2;

  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% per bumped",
           "  bumped-focused:", total_focused,
           percent (total_focused, total));
  compute_tier_limits (ring, 0, &tier1, &tier2, &bumped1, &bumped2);
  PRINTLN ("%-22s %17u %13.2f %% per bumped",
           "  focused-tier1-limit:", tier1,
           percent (bumped1, total_focused));
  PRINTLN ("%-22s %17u %13.2f %% per bumped",
           "  focused-tier2-limit:", tier2,
           percent (bumped2, total_focused));

  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% per bumped",
           "  bumped-stable:", total_stable, percent (total_stable, total));
  compute_tier_limits (ring, 1, &tier1, &tier2, &bumped1, &bumped2);
  PRINTLN ("%-22s %17u %13.2f %% per bumped",
           "  stable-tier1-limit:", tier1, percent (bumped1, total_stable));
  PRINTLN ("%-22s %17u %13.2f %% per bumped",
           "  stable-tier2-limit:", tier2, percent (bumped2, total_stable));
}

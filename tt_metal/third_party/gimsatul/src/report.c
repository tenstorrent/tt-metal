#ifndef QUIET

#include "report.h"
#include "message.h"
#include "ruler.h"
#include "utilities.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// clang-format off

static _Atomic (uint64_t) reported;

// clang-format on

void reset_report () { reported = 0; }

void verbose_report (struct ring *ring, char type, int level) {
  if (verbosity < level)
    return;

  if (ring->options.report <= ring->id)
    return;

  struct ring_statistics *s = &ring->statistics;
  struct averages *a = ring->averages + ring->stable;

  acquire_message_lock ();

  double t = wall_clock_time ();
  double m = current_resident_set_size () / (double) (1 << 20);
  uint64_t conflicts = s->contexts[SEARCH_CONTEXT].conflicts;
  unsigned active = s->active;

  bool header = !(atomic_fetch_add (&reported, 1) % 20);

  if (header)
    fputs ("c\nc     seconds MB level reductions restarts "
           "conflicts redundant trail tier1 glue size irredundant "
           "variables\nc\n",
           stdout);

  printf ("c%u %c %7.2f %4.0f %5.0f %6" PRIu64 " %9" PRIu64 " %11" PRIu64
          " %9zu %3.0f%% %3u %6.1f %6.1f %9zu %9u %3.0f%%\n",
          ring->id, type, t, m, a->level.value, s->reductions, s->restarts,
          conflicts, s->redundant, a->trail.value,
          ring->tier1_glue_limit[ring->stable], a->glue.slow.value,
          a->size.value, s->irredundant, active,
          percent (active, ring->ruler->size));

  fflush (stdout);

  release_message_lock ();
}

void report (struct ring *ring, char type) {
  verbose_report (ring, type, 0);
}

#endif

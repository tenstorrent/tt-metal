#include "fail.h"
#include "assign.h"
#include "backtrack.h"
#include "export.h"
#include "import.h"
#include "message.h"
#include "probe.h"
#include "propagate.h"
#include "report.h"
#include "ring.h"
#include "ruler.h"
#include "search.h"
#include "utilities.h"

#include <inttypes.h>

void failed_literal_probing (struct ring *ring) {
  if (ring->inconsistent)
    return;
  if (!ring->options.fail)
    return;
  START (ring, fail);
  assert (SEARCH_TICKS >= ring->last.probing);
  uint64_t delta_search_ticks = SEARCH_TICKS - ring->last.probing;
  delta_search_ticks = MAX (MIN_ABSOLUTE_FFORT, delta_search_ticks);
  uint64_t delta_probing_ticks = FAILED_EFFORT * delta_search_ticks;
  verbose (ring,
           "failed literal probing effort of %" PRIu64 " = %g * %" PRIu64
           " search ticks",
           delta_probing_ticks, (double) FAILED_EFFORT, delta_search_ticks);
  uint64_t probing_ticks_before = PROBING_TICKS;
  uint64_t limit = probing_ticks_before + delta_probing_ticks;
  signed char *values = ring->values;
  bool *inactive = ring->inactive;
  unsigned start = INVALID;
  unsigned max_lit = 2 * ring->size;
  unsigned probe = ring->probe;
  if (probe >= max_lit)
    probe = 0;
  unsigned failed = 0, lifted = 0, probed = 0, last = INVALID;
  unsigned *stamps = allocate_and_clear_array (max_lit, sizeof *stamps);
  struct unsigneds lift;
  INIT (lift);
  while (PROBING_TICKS <= limit) {
    assert (!ring->inconsistent);
    while (probe != start &&
           (inactive[IDX (probe)] || stamps[probe] == failed + 1)) {
      if (start == INVALID)
        start = probe;
      if (++probe == max_lit)
        probe = 0;
    }
    if (!probed)
      very_verbose (ring, "failed literal probing starts at literal %d",
                    unmap_and_export_literal (ring->ruler->unmap, probe));
    if (terminate_ring (ring))
      break;
    if (probe == start)
      break;
    if (start == INVALID)
      start = probe;
    if (import_shared (ring)) {
      if (ring->inconsistent)
        break;
      if (ring_propagate (ring, false, 0)) {
        set_inconsistent (
            ring, "unit propagation fails after importing shared clauses "
                  "during failed literal probing");
        break;
      }
      assert (ring->iterating < 0);
      iterate (ring);
      if (values[probe])
        continue;
    }
    assert (!ring->values[probe]);
    ring->statistics.contexts[PROBING_CONTEXT].decisions++;
    assert (!ring->level);
    ring->level = 1;
    probed++;
    LOG ("probing literal %s", LOGLIT (probe));
    assign_decision (ring, probe);
    struct ring_trail *trail = &ring->trail;
    unsigned *saved = trail->propagate;
    assert (saved + 1 == trail->end);
    bool ok = !ring_propagate (ring, false, 0);
    unsigned unit = INVALID;
    if (ok) {
      unsigned not_probe = NOT (probe);
      if (last == not_probe) {
        assert (probe & 1);
        unsigned *q = lift.begin;
        for (unsigned *p = q; p != lift.end; p++)
          if (values[*p] > 0)
            *q++ = *p;
        lift.end = q;
      } else if (!EMPTY (lift))
        CLEAR (lift);
      if (EMPTY (lift)) {
        unsigned *end = trail->end;
        LOG ("stamping %zu literals not to be probed", end - saved);
        assert (failed < UINT_MAX);
        unsigned stamp = failed + 1;
        for (unsigned *p = saved; p != end; p++)
          stamps[*p] = stamp;
        if (!(probe & 1)) {
          CLEAR (lift);
          assert (saved < end);
          for (unsigned *p = saved + 1; p != end; p++)
            PUSH (lift, *p);
        }
      } else {
        assert (probe & 1);
        assert (last == NOT (probe));
        backtrack (ring, 0);
        assert (ring->trail.end == ring->trail.propagate);
        for (unsigned *p = lift.begin; p != lift.end; p++) {
          unit = *p;
          signed char value = values[unit];
          if (value > 0)
            continue;
          if (value < 0) {
            trace_add_empty (&ring->trace);
            set_inconsistent (
                ring, "falsified lifted literal yields empty clause "
                      "during failed literal probing");
            break;
          }
          LOG ("lifted literal %s", LOGLIT (unit));
          ring->statistics.lifted++;
          lifted++;
          trace_add_binary (&ring->trace, not_probe, unit);
          trace_add_binary (&ring->trace, probe, unit);
          trace_add_unit (&ring->trace, unit);
          trace_delete_binary (&ring->trace, not_probe, unit);
          trace_delete_binary (&ring->trace, probe, unit);
          assign_ring_unit (ring, unit);
          if (ring_propagate (ring, false, 0)) {
            trace_add_empty (&ring->trace);
            set_inconsistent (
                ring, "propagating of lifted literal yields empty clause "
                      "during failed literal probing");
            break;
          }
        }
        CLEAR (lift);
        assert (ok);
        if (ring->inconsistent)
          break;
        if (lifted)
          ring->iterating = -1;
      }
    }
    if (ring->level) {
      backtrack (ring, 0);
      assert (ring->trail.end == ring->trail.propagate);
    }
    assert (!ring->level);
    if (!ok) {
      LOG ("failed literal %s", LOGLIT (probe));
      ring->statistics.failed++;
      failed++;
      unit = NOT (probe);
      trace_add_unit (&ring->trace, unit);
      assign_ring_unit (ring, unit);
      if (ring_propagate (ring, false, 0)) {
        trace_add_empty (&ring->trace);
        set_inconsistent (
            ring, "propagation of failed literal yields empty clause");
        break;
      }
      ring->iterating = -1;
    }
    last = probe;
    if (++probe == max_lit)
      probe = 0;
    if (ring->iterating)
      iterate (ring);
  }
  RELEASE (lift);
  free (stamps);
  very_verbose (ring,
                "failed literal probing ends at literal %d after %" PRIu64
                " ticks (%s)",
                unmap_and_export_literal (ring->ruler->unmap, probe),
                PROBING_TICKS - probing_ticks_before,
                (PROBING_TICKS > limit ? "limit hit" : "completed"));
  ring->probe = probe;
  verbose (ring,
           "probed %u literals %.0f%% and "
           "found %u failed literals %.0f%% lifted %u",
           probed, percent (probed, max_lit), failed,
           percent (failed, probed), lifted);
  verbose_report (ring, 'f', !(failed || lifted));
  STOP (ring, fail);
}

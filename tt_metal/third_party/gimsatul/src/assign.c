#include "assign.h"
#include "macros.h"
#include "ruler.h"
#include "trace.h"

#define DECISION_REASON 0
#define UNIT_REASON 1
#define REAL_REASON 2

static void assign (struct ring *ring, unsigned lit, struct watch *reason,
                    int type) {
  const unsigned not_lit = NOT (lit);
  unsigned idx = IDX (lit);

  assert (idx < ring->size);
  assert (!ring->values[lit]);
  assert (!ring->values[not_lit]);
  assert (!ring->inactive[idx]);

  assert (ring->unassigned);
  ring->unassigned--;

  ring->values[lit] = 1;
  ring->values[not_lit] = -1;

  if (ring->context != PROBING_CONTEXT)
    ring->phases[idx].saved = SGN (lit) ? -1 : 1;

  struct variable *v = ring->variables + idx;
  unsigned level = ring->level;
  unsigned assignment_level;
  if (type == DECISION_REASON)
    assignment_level = level, reason = 0;
  else if (type == UNIT_REASON)
    assignment_level = 0, reason = 0;
  else if (!level)
    assignment_level = 0;
  else if (is_binary_pointer (reason)) {
    unsigned other = other_pointer (reason);
    unsigned other_idx = IDX (other);
    struct variable *u = ring->variables + other_idx;
    assignment_level = u->level;
    if (assignment_level && is_binary_pointer (u->reason)) {
      bool redundant =
          redundant_pointer (reason) || redundant_pointer (u->reason);
      reason = tag_binary (redundant, lit, other_pointer (u->reason));
#ifdef LOGGING
      v->level = assignment_level;
      LOGWATCH (reason, "jumping %s reason", LOGLIT (lit));
#endif
      ring->statistics.contexts[ring->context].jumped++;
    }
  } else {
    assignment_level = 0;
    struct watcher *watcher = get_watcher (ring, reason);
    for (all_watcher_literals (other, watcher)) {
      if (other == lit)
        continue;
      unsigned other_idx = IDX (other);
      struct variable *u = ring->variables + other_idx;
      unsigned other_level = u->level;
      if (other_level > assignment_level)
        assignment_level = other_level;
    }
  }

  assert (assignment_level <= level);
  v->level = assignment_level;

  if (!assignment_level) {
    if (reason)
      trace_add_unit (&ring->trace, lit);
    v->reason = 0;
    ring->statistics.fixed++;
    assert (ring->statistics.active);
    ring->statistics.active--;
    assert (!ring->inactive[idx]);
    ring->inactive[idx] = true;
    *ring->ring_units.end++ = lit;
  } else
    v->reason = reason;

  struct ring_trail *trail = &ring->trail;
  size_t pos = SIZE (*trail);
  assert (pos < ring->size);
  trail->pos[idx] = pos;
  assert (trail->end < trail->begin + ring->size);
  *trail->end++ = lit;

#ifdef LOGGING
  if (assignment_level < level) {
    if (reason)
      LOGWATCH (reason, "out-of-order assignment %s reason", LOGLIT (lit));
    else
      LOG ("out-of-order assignment %s", LOGLIT (lit));
  }
#endif
}

void assign_with_reason (struct ring *ring, unsigned lit,
                         struct watch *reason) {
  assert (reason);
  assign (ring, lit, reason, REAL_REASON);
  LOGWATCH (reason, "assign %s with reason", LOGLIT (lit));
}

void assign_ring_unit (struct ring *ring, unsigned unit) {
  assign (ring, unit, 0, UNIT_REASON);
  LOG ("assign %s unit", LOGLIT (unit));
}

void assign_decision (struct ring *ring, unsigned decision) {
  assert (ring->level);
  assign (ring, decision, 0, DECISION_REASON);
#ifdef LOGGING
  if (ring->context == WALK_CONTEXT)
    LOG ("assign %s decision warm-up", LOGLIT (decision));
  else if (ring->context == PROBING_CONTEXT)
    LOG ("assign %s decision probe", LOGLIT (decision));
  else {
    assert (ring->context == SEARCH_CONTEXT);
    if (ring->stable)
      LOG ("assign %s decision score %g", LOGLIT (decision),
           ring->heap.nodes[IDX (decision)].score);
    else
      LOG ("assign %s decision stamp %" PRIu64, LOGLIT (decision),
           ring->queue.links[IDX (decision)].stamp);
  }
#endif
}

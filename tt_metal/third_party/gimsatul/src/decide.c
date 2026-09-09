#include "decide.h"
#include "assign.h"
#include "macros.h"
#include "message.h"
#include "options.h"
#include "random.h"
#include "ring.h"
#include "utilities.h"

#include <inttypes.h>

signed char initial_phase (struct ring *ring) {
  return ring->options.phase ? 1 : -1;
}

signed char decide_phase (struct ring *ring, unsigned idx) {
  struct phases *p = ring->phases + idx;
  unsigned target = ring->options.target_phases;
  signed char res = 0;
  if (ring->options.force_phase)
    res = initial_phase (ring);
  if (!res)
    if ((target && ring->stable) || (target > 1 && !ring->stable))
      res = p->target;
  if (!res)
    res = p->saved;
  if (!res)
    res = initial_phase (ring);
  return res;
}

static unsigned random_decision (struct ring *ring) {
  assert (ring->unassigned);

  signed char *values = ring->values;
  bool *inactive = ring->inactive;
  unsigned size = ring->size;

  unsigned idx = random_modulo (&ring->random, size);
  unsigned lit = LIT (idx);

  if (inactive[idx] || values[lit]) {
    unsigned delta = random_modulo (&ring->random, size);
    while (gcd (delta, size) != 1)
      if (++delta == size)
        delta = 1;
    assert (delta < size);
    do {
      idx += delta;
      if (idx >= size)
        idx -= size;
      lit = LIT (idx);
    } while (inactive[idx] || values[lit]);
  }

  LOG ("random decision %s", LOGVAR (idx));

  if (ring->context == SEARCH_CONTEXT)
    ring->statistics.decisions.random++;

  return idx;
}

static unsigned best_decision_on_heap (struct ring *ring) {
  assert (ring->unassigned);

  signed char *values = ring->values;
  struct heap *heap = &ring->heap;
  struct node *nodes = heap->nodes;

  unsigned idx;
  for (;;) {
    struct node *root;
    root = heap->root;
    assert (root);
    assert (root - nodes < ring->size);
    idx = root - nodes;
    unsigned lit = LIT (idx);
    if (!values[lit])
      break;
    pop_heap (heap);
  }

  LOG ("best decision %s on heap with score %g", LOGVAR (idx),
       nodes[idx].score);

  if (ring->context == SEARCH_CONTEXT)
    ring->statistics.decisions.heap++;

  return idx;
}

static unsigned best_decision_on_queue (struct ring *ring) {
  assert (ring->unassigned);

  signed char *values = ring->values;
  struct queue *queue = &ring->queue;
  struct link *links = queue->links;
  struct link *search = queue->search;

  unsigned lit, idx;
  for (;;) {
    assert (search);
    idx = search - links;
    lit = LIT (idx);
    if (!values[lit])
      break;
    search = search->prev;
  }
  queue->search = search;

  LOG ("best decision %s on queue with stamp %" PRIu64, LOGVAR (idx),
       search->stamp);

  if (ring->context == SEARCH_CONTEXT)
    ring->statistics.decisions.queue++;

  return idx;
}

void start_random_decision_sequence (struct ring *ring) {
  if (!ring->options.random_decisions)
    return;

  if (ring->stable && !ring->options.random_stable_decisions)
    return;

  if (!ring->stable && !ring->options.random_focused_decisions)
    return;

  if (ring->randec)
    very_verbose (ring,
                  "continuing random decision sequence at %" PRIu64
                  " conflicts",
                  SEARCH_CONFLICTS);
  else {
    uint64_t sequences = ++ring->statistics.random_sequences;
    unsigned length = ring->options.random_decision_length;
    ring->randec = length;

    very_verbose (ring,
                  "starting random decision sequence %" PRIu64
                  " at %" PRIu64 " conflicts",
                  sequences, SEARCH_CONFLICTS);

    uint64_t base = ring->options.random_decision_interval;
    uint64_t interval = base * logn (sequences);
    ring->limits.randec = SEARCH_CONFLICTS + interval;
  }
}

static unsigned next_random_decision (struct ring *ring) {
  if (!ring->size)
    return INVALID_VAR;

  if (ring->context != SEARCH_CONTEXT)
    return INVALID_VAR;

  if (!ring->options.random_decisions)
    return INVALID_VAR;

  if (ring->stable && !ring->options.random_stable_decisions)
    return INVALID_VAR;

  if (!ring->stable && !ring->options.random_focused_decisions)
    return INVALID_VAR;

  if (!ring->randec) {
    assert (ring->level);
    if (ring->level > 1)
      return INVALID_VAR;

    uint64_t conflicts = SEARCH_CONFLICTS;
    if (conflicts < ring->limits.randec)
      return INVALID_VAR;

    start_random_decision_sequence (ring);
  }

  return random_decision (ring);
}

void decide (struct ring *ring) {
  ring->level++;

  unsigned idx = next_random_decision (ring);
  if (idx == INVALID_VAR) {
    if (ring->stable)
      idx = best_decision_on_heap (ring);
    else
      idx = best_decision_on_queue (ring);
  }

  signed char phase = decide_phase (ring, idx);
  unsigned lit = LIT (idx);

  if (phase < 0)
    lit = NOT (lit);

  struct context *context = ring->statistics.contexts;
  context += ring->context;
  context->decisions++;

  if (ring->context == SEARCH_CONTEXT) {
    if (phase < 0)
      ring->statistics.decisions.negative++;
    else
      ring->statistics.decisions.positive++;
  }

  assign_decision (ring, lit);
}

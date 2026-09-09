#include "backtrack.h"
#include "message.h"
#include "ring.h"

static void unassign (struct ring *ring, unsigned lit) {
#ifdef LOGGING
  ring->level = VAR (lit)->level;
  LOG ("unassign %s", LOGLIT (lit));
#endif
  unsigned not_lit = NOT (lit);
  signed char *values = ring->values;
  values[lit] = values[not_lit] = 0;
  assert (ring->unassigned < ring->size);
  ring->unassigned++;
  unsigned idx = IDX (lit);
  if (ring->stable) {
    struct heap *heap = &ring->heap;
    struct node *node = heap->nodes + idx;
    if (!heap_contains (heap, node))
      push_heap (heap, node);
  } else {
    struct queue *queue = &ring->queue;
    struct link *link = queue->links + idx;
    update_queue_search (queue, link);
  }
}

void backtrack (struct ring *ring, unsigned new_level) {
  assert (ring->level > new_level);
  LOG ("backtracking to decision level %u", new_level);
  struct ring_trail *trail = &ring->trail;
  unsigned *t = trail->end;
  assert (EMPTY (ring->outoforder));
  while (t != trail->begin) {
    unsigned lit = *--t;
    unsigned idx = IDX (lit);
    struct variable *v = ring->variables + idx;
    unsigned lit_level = v->level;
    if (lit_level <= new_level)
      PUSH (ring->outoforder, lit);
    else {
      unassign (ring, lit);
      if (!v->reason && new_level + 1 == lit_level)
        break;
    }
  }
  trail->end = trail->propagate = t;
  ring->level = new_level;
  LOG ("backtracked to decision level %u", new_level);
  size_t pos = SIZE (*trail);
  while (!EMPTY (ring->outoforder)) {
    unsigned lit = POP (ring->outoforder);
    LOG ("keeping out-of-order assigned %s", LOGLIT (lit));
    *trail->end++ = lit;
    unsigned idx = IDX (lit);
    trail->pos[idx] = pos++;
  }
  assert (pos == SIZE (*trail));
}

void update_best_and_target_phases (struct ring *ring) {
  if (!ring->stable)
    return;
  unsigned assigned = SIZE (ring->trail);
  if (ring->target < assigned) {
    very_verbose (ring,
                  "updating target assigned trail height from %u to %u",
                  ring->target, assigned);
    ring->target = assigned;
    signed char *q = ring->values;
    for (all_phases (p)) {
      signed char tmp = *q;
      q += 2;
      if (tmp)
        p->target = tmp;
    }
  }
  if (ring->best < assigned) {
    very_verbose (ring, "updating best assigned trail height from %u to %u",
                  ring->best, assigned);
    ring->best = assigned;
    signed char *q = ring->values;
    for (all_phases (p)) {
      signed char tmp = *q;
      q += 2;
      if (tmp)
        p->best = tmp;
    }
  }
}

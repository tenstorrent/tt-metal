#include "inline.h"
#include "inlinequeue.h"
#include "print.h"

void kissat_init_queue(kissat *solver) {
  queue *queue = &solver->queue;
  queue->first = queue->last = DISCONNECT;
  assert(!queue->stamp);

  queue->search.idx = DISCONNECT;
  assert(!queue->search.stamp);
}

void kissat_reset_queue(kissat *solver) {
  LOG("reset queue");
  queue *queue = &solver->queue;
  links *links = solver->links;
  const unsigned last = queue->last;
  assert(!DISCONNECTED(last));
  kissat_update_queue(solver, links, last);
}

void kissat_reassign_queue_stamps(kissat *solver) {
  kissat_very_verbose(solver, "need to reassign enqueue stamps on queue");

  queue *queue = &solver->queue;
  links *links = solver->links;
  queue->stamp = 0;

  struct links *l;
  for (unsigned idx = queue->first; !DISCONNECTED(idx); idx = l->next) {
    (l = links + idx)->stamp = ++queue->stamp;
  }

  if (!DISCONNECTED(queue->search.idx)) {
    queue->search.stamp = links[queue->search.idx].stamp;
  }
}

#if defined(CHECK_QUEUE) && !defined(NDEBUG)
void kissat_check_queue(kissat *solver) {
  links *links = solver->links;
  queue *queue = &solver->queue;
  bool passed_search_idx = false;
  const bool focused = !solver->stable;
  for (unsigned idx = queue->first, prev = DISCONNECT;
        !DISCONNECTED(idx); idx = links[idx].next) {
    if (!DISCONNECTED(prev)) {
      assert(links[prev].stamp < links[idx].stamp);
    }
    if (focused && passed_search_idx) {
      assert(VALUE(LIT(idx)));
    }
    if (idx == queue->search.idx) {
      passed_search_idx = true;
    }
  }
  if (!DISCONNECTED(queue->search.idx)) {
    assert(links[queue->search.idx].stamp == queue->search.stamp);
  }
}
#endif

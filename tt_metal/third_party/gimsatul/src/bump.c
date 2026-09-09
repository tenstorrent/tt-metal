#include "bump.h"
#include "logging.h"
#include "ring.h"

#include <string.h>

static void rescale_variable_scores (struct ring *ring) {
  struct heap *heap = &ring->heap;
  double max_score = heap->increment;
  struct node *nodes = heap->nodes;
  struct node *end = nodes + ring->size;
  for (struct node *node = nodes; node != end; node++)
    if (node->score > max_score)
      max_score = node->score;
  LOG ("rescaling by maximum score of %g", max_score);
  assert (max_score > 0);
  for (struct node *node = nodes; node != end; node++)
    node->score /= max_score;
  heap->increment /= max_score;
}

static void bump_variable_on_heap (struct ring *ring, unsigned idx) {
  struct heap *heap = &ring->heap;
  struct node *node = heap->nodes + idx;
  double old_score = node->score;
  double new_score = old_score + heap->increment;
  LOG ("bumping %s old score %g to new score %g", LOGVAR (idx), old_score,
       new_score);
  update_heap (heap, node, new_score);
  if (new_score > MAX_SCORE)
    rescale_variable_scores (ring);
}

static void bump_variable_on_queue (struct ring *ring, unsigned idx) {
  struct queue *queue = &ring->queue;
  struct link *link = queue->links + idx;
#ifdef LOGGING
  uint64_t old_stamp = link->stamp;
#endif
  dequeue (queue, link);
  unsigned lit = LIT (idx);
  bool unassigned = !ring->values[lit];
  enqueue (queue, link, unassigned);
#ifdef LOGGING
  uint64_t new_stamp = link->stamp;
  LOG ("bumping %s old stamp %" PRIu64 " new stamp %" PRIu64, LOGVAR (idx),
       old_stamp, new_stamp);
#endif
}

static struct node *first_active_node (struct ring *ring) {
  struct heap *heap = &ring->heap;
  struct node *nodes = heap->nodes;
  struct node *end = nodes + ring->size;
  struct node *res = nodes;
  bool *inactive = ring->inactive;
  while (res != end) {
    unsigned idx = res - nodes;
    if (!inactive[idx])
      return res;
    res++;
  }
  return res;
}

static struct node *next_active_node (struct ring *ring,
                                      struct node *node) {
  struct heap *heap = &ring->heap;
  struct node *nodes = heap->nodes;
  struct node *end = nodes + ring->size;
  assert (nodes <= node);
  assert (node < end);
  struct node *res = node + 1;
  bool *inactive = ring->inactive;
  while (res != end) {
    unsigned idx = res - nodes;
    if (!inactive[idx])
      return res;
    res++;
  }
  return res;
}

#define all_active_nodes(NODE) \
  struct node *NODE = first_active_node (ring), \
              *END_##NODE = ring->heap.nodes + ring->size; \
  NODE != END_##NODE; \
  NODE = next_active_node (ring, NODE)

void rebuild_heap (struct ring *ring) {
  struct heap *heap = &ring->heap;
  heap->root = 0;
  for (all_active_nodes (node)) {
    node->child = node->prev = node->next = 0;
    push_heap (heap, node);
  }
}

static void bump_score_increment (struct ring *ring) {
  if (!ring->stable)
    return;
  struct heap *heap = &ring->heap;
  double old_increment = heap->increment;
  double factor = 1.0 / DECAY;
  double new_increment = old_increment * factor;
  LOG ("new increment %g", new_increment);
  heap->increment = new_increment;
  if (heap->increment > MAX_SCORE)
    rescale_variable_scores (ring);
}

static void sort_analyzed_variable_according_to_stamp (struct ring *ring) {
  struct link *links = ring->queue.links;
  struct unsigneds *analyzed = &ring->analyzed;
  size_t size = SIZE (*analyzed), count[256];
  unsigned *begin = analyzed->begin;
  unsigned *tmp = 0, *a = begin, *b = 0, *c = a;
  uint64_t masked_lower = 0, masked_upper = 255;
  uint64_t upper = 0, lower = ~upper, shifted_mask = 255;
  bool bounded = false;
  for (size_t i = 0; i != 64; i += 8, shifted_mask <<= 8) {
    if (bounded && (lower & shifted_mask) == (upper & shifted_mask))
      continue;
    memset (count + masked_lower, 0,
            (masked_upper - masked_lower + 1) * sizeof *count);
    unsigned *end = c + size;
    bool sorted = true;
    uint64_t last = 0;
    for (unsigned *p = c; p != end; p++) {
      unsigned idx = *p;
      uint64_t r = links[idx].stamp;
      if (!bounded)
        lower &= r, upper |= r;
      uint64_t s = r >> i;
      uint64_t m = s & 255;
      if (sorted && last > m)
        sorted = false;
      else
        last = m;
      count[m]++;
    }
    masked_lower = (lower >> i) & 255;
    masked_upper = (upper >> i) & 255;
    if (!bounded) {
      bounded = true;
      if ((lower & shifted_mask) == (upper & shifted_mask))
        continue;
    }
    if (sorted)
      continue;
    size_t pos = 0;
    for (size_t j = masked_lower; j <= masked_upper; j++) {
      size_t delta = count[j];
      count[j] = pos;
      pos += delta;
    }
    if (!tmp) {
      assert (c == a);
      b = tmp = sorter_block (ring, size);
    }
    assert (b == tmp);
    unsigned *d = (c == a) ? b : a;
    for (unsigned *p = c; p != end; p++) {
      unsigned idx = *p;
      uint64_t r = links[idx].stamp;
      uint64_t s = r >> i;
      uint64_t m = s & 255;
      pos = count[m]++;
      d[pos] = idx;
    }
    c = d;
  }
  if (c == b) {
    size_t bytes = size * sizeof *begin;
    memcpy (a, b, bytes);
  }
#ifndef NDEBUG
  for (size_t i = 0; i + 1 < size; i++)
    assert (links[begin[i]].stamp < links[begin[i + 1]].stamp);
#endif
}

static void bump_analyze_variables_on_queue (struct ring *ring) {
  for (all_elements_on_stack (unsigned, idx, ring->analyzed))
    bump_variable_on_queue (ring, idx);
}

static void sort_and_bump_analyzed_variables_on_queue (struct ring *ring) {
  assert (!ring->stable);
  sort_analyzed_variable_according_to_stamp (ring);
  bump_analyze_variables_on_queue (ring);
}

static void bump_analyzed_variables_on_heap (struct ring *ring) {
  assert (ring->stable);
  for (all_elements_on_stack (unsigned, idx, ring->analyzed))
    bump_variable_on_heap (ring, idx);
  bump_score_increment (ring);
}

void bump_variables (struct ring *ring) {
  if (ring->stable)
    bump_analyzed_variables_on_heap (ring);
  else
    sort_and_bump_analyzed_variables_on_queue (ring);
}

#ifndef _inlineheap_h_INCLUDED
#define _inlineheap_h_INCLUDED

#include "allocate.h"
#include "internal.h"
#include "logging.h"

#define HEAP_CHILD(POS) \
  (assert ((POS) < (1u<<31)), (2*(POS) + 1))

#define HEAP_PARENT(POS) \
  (assert ((POS) > 0), (((POS) - 1)/2))

static inline void kissat_bubble_up(kissat *solver, heap *heap, unsigned idx) {
  unsigned *stack = BEGIN_STACK(heap->stack);
  unsigned *pos = heap->pos;
  unsigned idx_pos = pos[idx];
  const double *const score = heap->score;
  const double idx_score = score[idx];
  while (idx_pos) {
    const unsigned parent_pos = HEAP_PARENT(idx_pos);
    const unsigned parent = stack[parent_pos];
    if (score[parent] >= idx_score) {
      break;
    }
    LOG("heap bubble up: %u@%u = %g swapped with %u@%u = %g",
          parent, parent_pos, score[parent], idx, idx_pos, idx_score);
    stack[idx_pos] = parent;
    pos[parent] = idx_pos;
    idx_pos = parent_pos;
  }
  stack[idx_pos] = idx;
  pos[idx] = idx_pos;
#ifndef LOGGING
  (void) solver;
#endif
}

static inline void kissat_bubble_down(kissat *solver, heap *heap,
      unsigned idx) {
  unsigned *stack = BEGIN_STACK(heap->stack);
  const unsigned end = SIZE_STACK(heap->stack);
  unsigned *pos = heap->pos;
  unsigned idx_pos = pos[idx];
  const double *const score = heap->score;
  const double idx_score = score[idx];
  for (;;) {
    unsigned child_pos = HEAP_CHILD(idx_pos);
    if (child_pos >= end) {
      break;
    }
    unsigned child = stack[child_pos];
    double child_score = score[child];
    const unsigned sibling_pos = child_pos + 1;
    if (sibling_pos < end) {
      const unsigned sibling = stack[sibling_pos];
      const double sibling_score = score[sibling];
      if (sibling_score > child_score) {
        child = sibling;
        child_pos = sibling_pos;
        child_score = sibling_score;
      }
    }
    if (child_score <= idx_score) {
      break;
    }
    LOG("heap bubble down: %u@%u = %g swapped with %u@%u = %g",
          child, child_pos, score[child], idx, idx_pos, idx_score);
    stack[idx_pos] = child;
    pos[child] = idx_pos;
    idx_pos = child_pos;
  }
  stack[idx_pos] = idx;
  pos[idx] = idx_pos;
#ifndef LOGGING
  (void) solver;
#endif
}

#define HEAP_IMPORT(IDX) \
do { \
  assert ((IDX) < UINT_MAX-1); \
  if (heap->vars <= (IDX)) \
    kissat_enlarge_heap (solver, heap, (IDX) + 1); \
} while (0)

#define CHECK_HEAP_IMPORTED(IDX) \

static inline void kissat_push_heap(kissat *solver, heap *heap, unsigned idx) {
  LOG("push heap %u", idx);
  assert(!kissat_heap_contains(heap, idx));
  HEAP_IMPORT(idx);
  heap->pos[idx] = SIZE_STACK(heap->stack);
  PUSH_STACK(heap->stack, idx);
  kissat_bubble_up(solver, heap, idx);
}

static inline void kissat_pop_heap(kissat *solver, heap *heap, unsigned idx) {
  LOG("pop heap %u", idx);
  assert(kissat_heap_contains(heap, idx));
  const unsigned last = POP_STACK(heap->stack);
  heap->pos[last] = DISCONTAIN;
  if (last == idx) {
    return;
  }
  const unsigned idx_pos = heap->pos[idx];
  heap->pos[idx] = DISCONTAIN;
  POKE_STACK(heap->stack, idx_pos, last);
  heap->pos[last] = idx_pos;
  kissat_bubble_up(solver, heap, last);
  kissat_bubble_down(solver, heap, last);
#ifdef CHECK_HEAP
  kissat_check_heap(heap);
#endif
}

static inline unsigned kissat_pop_max_heap(kissat *solver, heap *heap) {
  assert(!EMPTY_STACK(heap->stack));
  unsigneds *stack = &heap->stack;
  unsigned *const begin = BEGIN_STACK(*stack);
  const unsigned idx = *begin;
  assert(!heap->pos[idx]);
  LOG("pop max heap %u", idx);
  const unsigned last = POP_STACK(*stack);
  unsigned *const pos = heap->pos;
  pos[last] = DISCONTAIN;
  if (last == idx) {
    return idx;
  }
  pos[idx] = DISCONTAIN;
  *begin = last;
  pos[last] = 0;
  kissat_bubble_down(solver, heap, last);
#ifdef CHECK_HEAP
  kissat_check_heap(heap);
#endif
  return idx;
}

static inline void kissat_update_heap(kissat *solver, heap *heap,
      unsigned idx, double new_score) {
  const double old_score = kissat_get_heap_score(heap, idx);
  if (old_score == new_score) {
    return;
  }
  HEAP_IMPORT(idx);
  LOG("update heap %u score from %g to %g", idx, old_score, new_score);
  heap->score[idx] = new_score;
  if (!heap->tainted) {
    heap->tainted = true;
    LOG("tainted heap");
  }
  if (!kissat_heap_contains(heap, idx)) {
    return;
  }
  if (new_score > old_score) {
    kissat_bubble_up(solver, heap, idx);
  } else {
    kissat_bubble_down(solver, heap, idx);
  }
#ifdef CHECK_HEAP
  kissat_check_heap(heap);
#endif
}

#endif

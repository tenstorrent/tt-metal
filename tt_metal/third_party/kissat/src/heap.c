#include "allocate.h"
#include "inlineheap.h"
#include "internal.h"
#include "logging.h"

#include <string.h>

void kissat_release_heap(kissat *solver, heap *heap) {
  RELEASE_STACK(heap->stack);
  DEALLOC(heap->pos, heap->size);
  DEALLOC(heap->score, heap->size);
  memset(heap, 0, sizeof * heap);
}

#ifndef NDEBUG

void kissat_check_heap(heap *heap) {
  const unsigned *const stack = BEGIN_STACK(heap->stack);
  const unsigned end = SIZE_STACK(heap->stack);
  const unsigned *const pos = heap->pos;
  const double *const score = heap->score;
  for (unsigned i = 0; i < end; i++) {
    const unsigned idx = stack[i];
    const unsigned idx_pos = pos[idx];
    assert(idx_pos == i);
    unsigned child_pos = HEAP_CHILD(idx_pos);
    unsigned parent_pos = HEAP_PARENT(child_pos);
    assert(parent_pos == idx_pos);
    if (child_pos < end) {
      unsigned child = stack[child_pos];
      assert(score[idx] >= score[child]);
      if (++child_pos < end) {
        parent_pos = HEAP_PARENT(child_pos);
        assert(parent_pos == idx_pos);
        child = stack[child_pos];
        assert(score[idx] >= score[child]);
      }
    }
  }
}

#endif

void kissat_resize_heap(kissat *solver, heap *heap, unsigned new_size) {
  const unsigned old_size = heap->size;
  if (old_size >= new_size) {
    return;
  }
  LOG("resizing %s heap from %u to %u",
        (heap->tainted ? "tainted" : "untainted"), old_size, new_size);

  heap->pos = kissat_nrealloc(solver, heap->pos,
              old_size, new_size, sizeof(unsigned));
  if (heap->tainted) {
    heap->score = kissat_nrealloc(solver, heap->score,
                old_size, new_size, sizeof(double));
  } else {
    if (old_size) {
      DEALLOC(heap->score, old_size);
    }
    heap->score = kissat_calloc(solver, new_size, sizeof(double));
  }
  heap->size = new_size;
#ifdef CHECK_HEAP
  kissat_check_heap(heap);
#endif
}

void kissat_rescale_heap(kissat *solver, heap *heap, double factor) {
  LOG("rescaling scores on heap with factor %g", factor);
  double *score = heap->score;
  for (unsigned i = 0; i < heap->vars; i++) {
    score[i] *= factor;
  }
#ifndef NDEBUG
  kissat_check_heap(heap);
#endif
#ifndef LOGGING
  (void) solver;
#endif
}

void kissat_enlarge_heap(kissat *solver, heap *heap, unsigned new_vars) {
  const unsigned old_vars = heap->vars;
  assert(old_vars < new_vars);
  assert(new_vars <= heap->size);
  const size_t delta = new_vars - heap->vars;
  memset(heap->pos + old_vars, 0xff, delta * sizeof(unsigned));
  heap->vars = new_vars;
  if (heap->tainted) {
    memset(heap->score + old_vars, 0, delta * sizeof(double));
  }
  LOG("enlarged heap from %u to %u", old_vars, new_vars);
#ifndef LOGGING
  (void) solver;
#endif
}

#ifndef NDEBUG

static void dump_heap(heap *heap) {
  for (unsigned i = 0; i < SIZE_STACK(heap->stack); i++) {
    printf("heap.stack[%u] = %u\n", i, PEEK_STACK(heap->stack, i));
  }
  for (unsigned i = 0; i < heap->vars; i++) {
    printf("heap.pos[%u] = %u\n", i, heap->pos[i]);
  }
  for (unsigned i = 0; i < heap->vars; i++) {
    printf("heap.score[%u] = %g\n", i, heap->score[i]);
  }
}

void kissat_dump_heap(heap *heap) {
  dump_heap(heap);
}

#endif

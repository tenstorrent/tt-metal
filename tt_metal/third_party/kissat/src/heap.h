#ifndef _heap_h_INCLUDED
#define _heap_h_INCLUDED

#include "stack.h"
#include "utilities.h"

#include <assert.h>
#include <limits.h>
#include <stdbool.h>

#define DISCONTAIN UINT_MAX
#define DISCONTAINED(IDX) ((int)(IDX) < 0)

typedef struct heap heap;

struct heap {
  bool tainted;
  unsigned vars;
  unsigned size;
  unsigneds stack;
  double *score;
  unsigned *pos;
};

struct kissat;

void kissat_resize_heap(struct kissat *, heap *, unsigned size);
void kissat_release_heap(struct kissat *, heap *);

static inline bool kissat_heap_contains(heap *heap, unsigned idx) {
  return idx < heap->vars && !DISCONTAINED(heap->pos[idx]);
}

static inline unsigned kissat_get_heap_pos(const heap *heap, unsigned idx) {
  return idx < heap->vars ? heap->pos[idx] : DISCONTAIN;
}

static inline double kissat_get_heap_score(const heap *heap, unsigned idx) {
  return idx < heap->vars ? heap->score[idx] : 0.0;
}

static inline bool kissat_empty_heap(heap *heap) {
  return EMPTY_STACK(heap->stack);
}

static inline size_t kissat_size_heap(heap *heap) {
  return SIZE_STACK(heap->stack);
}

static inline unsigned kissat_max_heap(heap *heap) {
  assert(!kissat_empty_heap(heap));
  return PEEK_STACK(heap->stack, 0);
}

void kissat_rescale_heap(struct kissat *, heap *heap, double factor);

void kissat_enlarge_heap(struct kissat *, heap *, unsigned new_vars);

static inline double kissat_max_score_on_heap(heap *heap) {
  if (!heap->tainted) {
    return 0;
  }
  assert(heap->vars);
  const double *const score = heap->score;
  const double *const end = score + heap->vars;
  double res = score[0];
  for (const double *p = score + 1; p != end; p++) {
    res = MAX(res, *p);
  }
  return res;
}

#ifndef NDEBUG
void kissat_dump_heap(heap *);
#endif

#ifndef NDEBUG
void kissat_check_heap(heap *);
#else
#define kissat_check_heap(...) do { } while (0)
#endif

#endif

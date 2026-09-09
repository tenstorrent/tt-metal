#ifndef _heap_h_INCLUDED
#define _heap_h_INCLUDED

#include <assert.h>
#include <stdbool.h>

struct node {
  double score;
  struct node *child, *prev, *next;
};

struct heap {
  double increment;
  struct node *nodes;
  struct node *root;
#ifndef NDEBUG
  double last;
#endif
};

/*------------------------------------------------------------------------*/

void pop_heap (struct heap *);
void push_heap (struct heap *, struct node *node);
void update_heap (struct heap *, struct node *, double new_score);

/*------------------------------------------------------------------------*/

static inline bool heap_contains (struct heap *heap, struct node *node) {
  return heap->root == node || node->prev;
}

#endif

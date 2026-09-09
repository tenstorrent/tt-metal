#include "heap.h"

static struct node *merge_nodes (struct node *a, struct node *b) {
  if (!a)
    return b;
  if (!b)
    return a;
  assert (a != b);
  struct node *parent, *child;
  if (b->score > a->score)
    parent = b, child = a;
  else
    parent = a, child = b;
  struct node *parent_child = parent->child;
  child->next = parent_child;
  if (parent_child)
    parent_child->prev = child;
  child->prev = parent;
  parent->child = child;
  parent->prev = parent->next = 0;

  return parent;
}

static struct node *collapse_node (struct node *node) {
  if (!node)
    return 0;

  struct node *next = node, *tail = 0;

  do {
    struct node *a = next;
    assert (a);
    struct node *b = a->next;
    if (b) {
      next = b->next;
      struct node *tmp = merge_nodes (a, b);
      assert (tmp);
      tmp->prev = tail;
      tail = tmp;
    } else {
      a->prev = tail;
      tail = a;
      break;
    }
  } while (next);

  struct node *res = 0;
  while (tail) {
    struct node *prev = tail->prev;
    res = merge_nodes (res, tail);
    tail = prev;
  }

  return res;
}

static void deheap_node (struct node *node) {
  assert (node);
  struct node *prev = node->prev;
  struct node *next = node->next;
  assert (prev);
  node->prev = 0;
  if (prev->child == node)
    prev->child = next;
  else
    prev->next = next;
  if (next)
    next->prev = prev;
}

void pop_heap (struct heap *heap) {
  struct node *root = heap->root;
  struct node *child = root->child;
  heap->root = collapse_node (child);
  assert (!heap_contains (heap, root));
#ifndef NDEBUG
  assert (heap->last >= root->score);
  heap->last = root->score;
#endif
}

void push_heap (struct heap *heap, struct node *node) {
  assert (!heap_contains (heap, node));
  node->child = 0;
  heap->root = merge_nodes (heap->root, node);
  assert (heap_contains (heap, node));
#ifndef NDEBUG
  if (heap->last < node->score)
    heap->last = node->score;
#endif
}

void update_heap (struct heap *heap, struct node *node, double new_score) {
  double old_score = node->score;
  assert (old_score <= new_score);
  if (old_score == new_score)
    return;
  node->score = new_score;
#ifndef NDEBUG
  if (heap->last < new_score)
    heap->last = new_score;
#endif
  struct node *root = heap->root;
  if (root == node)
    return;
  if (!node->prev)
    return;
  deheap_node (node);
  heap->root = merge_nodes (root, node);
}

#include "queue.h"

void enqueue (struct queue *queue, struct link *link, bool update) {
  if (queue->last)
    queue->last->next = link;
  else
    queue->first = link;
  link->prev = queue->last;
  queue->last = link;
  link->next = 0;
  link->stamp = ++queue->stamp;
  if (update || !queue->search)
    queue->search = link;
}

void dequeue (struct queue *queue, struct link *link) {
  assert (queue->search);
  if (link->prev) {
    assert (link->prev->next == link);
    link->prev->next = link->next;
  } else {
    assert (queue->first == link);
    queue->first = link->next;
  }
  if (link->next) {
    assert (link->next->prev == link);
    link->next->prev = link->prev;
  } else {
    assert (queue->last == link);
    queue->last = link->prev;
  }
  if (queue->search == link) {
    if (link->next)
      queue->search = link->next;
    else if (link->prev)
      queue->search = link->prev;
    else
      queue->search = 0;
  }
  link->prev = 0;
  link->next = 0;
}

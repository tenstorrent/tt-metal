#ifndef _barrier_h_INCLUDED
#define _barrier_h_INCLUDED

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

struct ring;
struct ruler;

struct barrier {
  const char *name;
  pthread_mutex_t mutex;
  pthread_cond_t condition;
  volatile bool disabled;
  unsigned waiting;
  unsigned left;
  unsigned size;
  uint64_t met;
};

void init_barrier (struct barrier *, const char *name, unsigned size);
bool rendezvous (struct barrier *, struct ring *, bool expected_enabled);
void abort_waiting_and_disable_barrier (struct barrier *);

#endif

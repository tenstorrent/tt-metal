#include "message.h"
#include "ruler.h"

#include <inttypes.h>
#include <string.h>

void init_barrier (struct barrier *barrier, const char *name,
                   unsigned size) {
  barrier->name = name;
  barrier->size = size;
  assert (!barrier->disabled);
  if (size < 2)
    return;
  pthread_mutex_init (&barrier->mutex, 0);
  pthread_cond_init (&barrier->condition, 0);
}

void abort_waiting_and_disable_barrier (struct barrier *barrier) {
  if (barrier->size < 2)
    return;
  if (pthread_mutex_lock (&barrier->mutex))
    fatal_error ("failed to acquire '%s[%" PRIu64 "]' barrier lock "
                 "to abort waiting",
                 barrier->name, barrier->met);
  uint64_t met = barrier->met;
  if (!barrier->disabled) {
    very_verbose (0, "disabling '%s[%" PRIu64 "]' barrier", barrier->name,
                  barrier->met);
    barrier->disabled = true;
    if (barrier->waiting) {
      very_verbose (
          0, "aborting %u waiting threads in '%s[%" PRIu64 "]' barrier",
          barrier->waiting, barrier->name, barrier->met);
      barrier->waiting = 0;
      barrier->met++;
      pthread_cond_broadcast (&barrier->condition);
    }
  }
  if (pthread_mutex_unlock (&barrier->mutex))
    fatal_error ("failed to release '%s[%" PRIu64 "]' barrier lock "
                 "to abort waiting",
                 barrier->name, met);
}

bool rendezvous (struct barrier *barrier, struct ring *ring,
                 bool expected_enabled) {
  if (barrier->size < 2)
    return true;
#ifndef NFASTPATH
  if (barrier->disabled)
    return false;
#endif
  if (pthread_mutex_lock (&barrier->mutex))
    fatal_error ("failed to acquire '%s[%" PRIu64 "]' barrier lock "
                 "during rendezvous of 'ring[%u}'",
                 barrier->name, barrier->met, ring->id);

  uint64_t met = barrier->met;
  bool res;

  if (barrier->disabled)
    res = false;
  else {
    assert (barrier->waiting < barrier->size);
    barrier->waiting++;

    very_verbose (ring, "entered '%s[%" PRIu64 "]' barrier (%u waiting)",
                  barrier->name, met, barrier->waiting);

    if (barrier->waiting == barrier->size) {
      barrier->met++;
      barrier->waiting = 0;
      pthread_cond_broadcast (&barrier->condition);
    } else
      while (!barrier->disabled && met == barrier->met)
        pthread_cond_wait (&barrier->condition, &barrier->mutex);

    barrier->left++;
    very_verbose (ring, "leaving '%s[%" PRIu64 "]' barrier (%u left)",
                  barrier->name, met, barrier->left);

    if (barrier->left == barrier->size)
      barrier->left = 0;

    res = !barrier->disabled;
  }

  if (pthread_mutex_unlock (&barrier->mutex))
    fatal_error ("failed to release '%s[%" PRIu64 "]' barrier lock "
                 "during rendezvous of 'ring[%u}'",
                 barrier->name, met, ring->id);

  if (expected_enabled && !res)
    fatal_error ("unexpected disabled '%s[%" PRIu64 "]' barrier "
                 "(%u waiting) in rendezvous of 'ring[%u]'",
                 barrier->name, met, barrier->waiting, ring->id);

  return res;
}

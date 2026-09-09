#include "detach.h"
#include "message.h"
#include "ruler.h"

#include <stdio.h>

static void *detach_and_delete_ring (void *ptr) {
  struct ring *ring = ptr;
  detach_ring (ring);
  delete_ring (ring);
  return ring;
}

static void start_detaching_and_deleting_ring (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
  assert (ruler->threads);
  pthread_t *thread = ruler->threads + ring->id;
  if (pthread_create (thread, 0, detach_and_delete_ring, ring))
    fatal_error ("failed to create deletion thread %u", ring->id);
}

static void stop_detaching_and_deleting_ring (struct ruler *ruler,
                                              unsigned id) {
  assert (ruler->threads);
  pthread_t *thread = ruler->threads + id;
  if (pthread_join (*thread, 0))
    fatal_error ("failed to join deletion thread %u", id);
}

void detach_and_delete_rings (struct ruler *ruler) {
  size_t threads = SIZE (ruler->rings);
  if (threads > 1) {
    if (verbosity > 0) {
      printf ("c deleting %zu rings in parallel\n", threads);
      fflush (stdout);
    }

    for (all_rings (ring))
      start_detaching_and_deleting_ring (ring);

    for (unsigned i = 0; i != threads; i++)
      stop_detaching_and_deleting_ring (ruler, i);
  } else if (threads) {
    if (verbosity > 0) {
      printf ("c deleting single ring in main thread\n");
      fflush (stdout);
    }

    struct ring *ring = first_ring (ruler);
    detach_and_delete_ring (ring);
  }
}

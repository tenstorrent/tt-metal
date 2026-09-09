#ifndef QUIET

#include "message.h"
#include "ruler.h"
#include "utilities.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

double start_profile (struct profile *profile, double time) {
  double volatile *p = &profile->start;
  assert (*p < 0);
  *p = time;
  return time;
}

double stop_profile (struct profile *profile, double time) {
  double volatile *p = &profile->start;
  double delta = time - *p;
  *p = -1;
  profile->time += delta;
  return time;
}

/*------------------------------------------------------------------------*/

static void flush_profile (double time, struct profile *profile) {
  double volatile *p = &profile->start;
  assert (*p >= 0);
  double delta = time - *p;
  *p = time;
  profile->time += delta;
}

static int cmp_profiles (struct profile *a, struct profile *b) {
  if (!a)
    return -1;
  if (!b)
    return -1;
  if (a->time < b->time)
    return -1;
  if (a->time > b->time)
    return 1;
  return strcmp (b->name, a->name);
}

/*------------------------------------------------------------------------*/

#define begin_ring_profiles ((struct profile *) (&ring->profiles))
#define end_ring_profiles (&ring->profiles.solve)

#define all_ring_profiles(PROFILE) \
  struct profile *PROFILE = begin_ring_profiles, \
                 *END_##PROFILE = end_ring_profiles; \
  PROFILE != END_##PROFILE; \
  ++PROFILE

static double flush_ring_profiles (struct ring *ring) {
  double time = current_time ();
  for (all_ring_profiles (profile))
    if (profile->start >= 0)
      flush_profile (time, profile);

  flush_profile (time, &ring->profiles.solve);
  return time;
}

void print_ring_profiles (struct ring *ring) {
  flush_ring_profiles (ring);
  double solving = ring->profiles.solve.time;
  struct profile *prev = 0;
  fputs ("c\n", stdout);
  for (;;) {
    struct profile *next = 0;
    for (all_ring_profiles (tmp))
      if (cmp_profiles (tmp, prev) < 0 && cmp_profiles (next, tmp) < 0)
        next = tmp;
    if (!next)
      break;
    PRINTLN ("%10.2f seconds  %5.1f %%  %s", next->time,
             percent (next->time, solving), next->name);
    prev = next;
  }
  PRINTLN ("-----------------------------------------");
  PRINTLN ("%10.2f seconds  100.0 %%  solving", solving);
  fputs ("c\n", stdout);
  fflush (stdout);
}

/*------------------------------------------------------------------------*/

#define begin_ruler_profiles ((struct profile *) (&ruler->profiles))
#define end_ruler_profiles (&ruler->profiles.total)

#define all_ruler_profiles(PROFILE) \
  struct profile *PROFILE = begin_ruler_profiles, \
                 *END_##PROFILE = end_ruler_profiles; \
  PROFILE != END_##PROFILE; \
  ++PROFILE

static double flush_ruler_profiles (struct ruler *ruler) {
  double time = current_time ();
  for (all_ruler_profiles (profile))
    if (profile->start >= 0)
      flush_profile (time, profile);

  flush_profile (time, &ruler->profiles.total);
  return time;
}

void print_ruler_profiles (struct ruler *ruler) {
  struct ring *ring = 0;
  flush_ruler_profiles (ruler);
  double total = ruler->profiles.total.time;
  struct profile *prev = 0;
  for (;;) {
    struct profile *next = 0;
    for (all_ruler_profiles (tmp))
      if (cmp_profiles (tmp, prev) < 0 && cmp_profiles (next, tmp) < 0)
        next = tmp;
    if (!next)
      break;
    PRINTLN ("%10.2f seconds  %5.1f %%  %s", next->time,
             percent (next->time, total), next->name);
    prev = next;
  }
  PRINTLN ("--------------------------------------------");
  PRINTLN ("%10.2f seconds  100.0 %%  total", total);
  fputs ("c\n", stdout);
  fflush (stdout);
}

#endif

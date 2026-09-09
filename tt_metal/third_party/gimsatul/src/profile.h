#ifndef _profile_h_INCLUDED
#define _profile_h_INCLUDED

#ifndef QUIET

#include "message.h"
#include "system.h"

struct profile {
  const char *name;
  volatile double start;
  volatile double time;
};

#define RING_PROFILES \
  RING_PROFILE (fail) \
  RING_PROFILE (focus) \
  RING_PROFILE (probe) \
  RING_PROFILE (reduce) \
  RING_PROFILE (search) \
  RING_PROFILE (stable) \
  RING_PROFILE (vivify) \
  RING_PROFILE (walk) \
\
  RING_PROFILE (solve)

struct ring_profiles {
#define RING_PROFILE(NAME) struct profile NAME;
  RING_PROFILES
#undef RING_PROFILE
};

#define RULER_PROFILES \
  RULER_PROFILE (clone) \
  RULER_PROFILE (eliminate) \
  RULER_PROFILE (deduplicate) \
  RULER_PROFILE (parse) \
  RULER_PROFILE (solve) \
  RULER_PROFILE (simplify) \
  RULER_PROFILE (substitute) \
  RULER_PROFILE (subsume) \
\
  RULER_PROFILE (total)

struct ruler_profiles {
#define RULER_PROFILE(NAME) struct profile NAME;
  RULER_PROFILES
#undef RULER_PROFILE
};

/*------------------------------------------------------------------------*/

#define profile_time current_time

#define START(OWNER, NAME) \
  (verbosity < 0 ? 0.0 \
                 : start_profile (&OWNER->profiles.NAME, profile_time ()))

#define STOP(OWNER, NAME) \
  (verbosity < 0 ? 0.0 \
                 : stop_profile (&OWNER->profiles.NAME, profile_time ()))

#define MODE_PROFILE \
  (ring->stable ? &ring->profiles.stable : &ring->profiles.focus)

#define STOP_SEARCH() \
  do { \
    if (verbosity < 0) \
      break; \
    double t = profile_time (); \
    stop_profile (MODE_PROFILE, t); \
    stop_profile (&ring->profiles.search, t); \
  } while (0)

#define START_SEARCH() \
  do { \
    if (verbosity < 0) \
      break; \
    double t = profile_time (); \
    start_profile (&ring->profiles.search, t); \
    start_profile (MODE_PROFILE, t); \
  } while (0)

#define STOP_SEARCH_AND_START(NAME) \
  do { \
    if (verbosity < 0) \
      break; \
    double t = profile_time (); \
    stop_profile (MODE_PROFILE, t); \
    stop_profile (&ring->profiles.search, t); \
    start_profile (&ring->profiles.NAME, t); \
  } while (0)

#define STOP_AND_START_SEARCH(NAME) \
  do { \
    if (verbosity < 0) \
      break; \
    double t = profile_time (); \
    stop_profile (&ring->profiles.NAME, t); \
    start_profile (&ring->profiles.search, t); \
    start_profile (MODE_PROFILE, t); \
  } while (0)

#define INIT_PROFILE(OWNER, NAME) \
  do { \
    if (verbosity < 0) \
      break; \
    struct profile *profile = &OWNER->profiles.NAME; \
    profile->start = -1; \
    profile->name = #NAME; \
  } while (0)

/*------------------------------------------------------------------------*/

double start_profile (struct profile *, double time);
double stop_profile (struct profile *, double time);

#else

struct ring_profiles {};
struct ruler_profiles {};

#define START(...) \
  do { \
  } while (0)
#define STOP(...) \
  do { \
  } while (0)
#define START_SEARCH(...) \
  do { \
  } while (0)
#define STOP_SEARCH(...) \
  do { \
  } while (0)
#define STOP_SEARCH_AND_START(...) \
  do { \
  } while (0)
#define STOP_AND_START_SEARCH(...) \
  do { \
  } while (0)
#define INIT_PROFILE(...) \
  do { \
  } while (0)

#endif

#endif

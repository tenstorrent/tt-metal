#ifndef _logging_h_INCLUDED
#define _logging_h_INCLUDED

#ifdef LOGGING

#include "message.h"

#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>

struct ring;
struct ruler;

extern bool ignore_values_and_levels_during_logging;

const char *loglit (struct ring *, unsigned lit);
const char *logvar (struct ring *, unsigned idx);
const char *roglit (struct ruler *, unsigned lit);
const char *rogvar (struct ruler *, unsigned idx);

#define LOGLIT(...) loglit (ring, __VA_ARGS__)
#define LOGVAR(...) logvar (ring, __VA_ARGS__)

#define ROGLIT(...) roglit (ruler, __VA_ARGS__)
#define ROGVAR(...) rogvar (ruler, __VA_ARGS__)

#define LOGPREFIX(...) \
  if (verbosity < INT_MAX) \
    break; \
  acquire_message_lock (); \
  printf (prefix_format, ring->id); \
  printf ("LOG %u ", ring->level); \
  printf (__VA_ARGS__)

#define LOGSUFFIX() \
  fputc ('\n', stdout); \
  fflush (stdout); \
  release_message_lock ()

#define LOG(...) \
  do { \
    LOGPREFIX (__VA_ARGS__); \
    LOGSUFFIX (); \
  } while (0)

#define LOGTMP(...) \
  do { \
    LOGPREFIX (__VA_ARGS__); \
    printf (" size %zu temporary clause", SIZE (ring->clause)); \
    for (all_elements_on_stack (unsigned, LIT, ring->clause)) \
      printf (" %s", LOGLIT (LIT)); \
    LOGSUFFIX (); \
  } while (0)

#define LOGBINARY(REDUNDANT, LIT, OTHER, ...) \
  do { \
    LOGPREFIX (__VA_ARGS__); \
    if ((REDUNDANT)) \
      printf (" redundant"); \
    else \
      printf (" irredundant"); \
    printf (" binary clause %s %s", LOGLIT (LIT), LOGLIT (OTHER)); \
    LOGSUFFIX (); \
  } while (0)

#define LOGCLAUSE(CLAUSE, ...) \
  do { \
    if (is_binary_pointer (CLAUSE)) { \
      bool REDUNDANT = redundant_pointer (CLAUSE); \
      unsigned LIT = lit_pointer (CLAUSE); \
      unsigned OTHER = other_pointer (CLAUSE); \
      LOGBINARY (REDUNDANT, LIT, OTHER, __VA_ARGS__); \
    } else { \
      LOGPREFIX (__VA_ARGS__); \
      if ((CLAUSE)->garbage) \
        printf (" garbage"); \
      if ((CLAUSE)->redundant) \
        printf (" redundant glue %u", (CLAUSE)->glue); \
      else \
        printf (" irredundant"); \
      printf (" size %u clause[%" PRIu64 "]", (CLAUSE)->size, \
              (uint64_t) (CLAUSE)->id); \
      for (all_literals_in_clause (LIT, (CLAUSE))) \
        printf (" %s", LOGLIT (LIT)); \
      LOGSUFFIX (); \
    } \
  } while (0)

#define LOGWATCH(WATCH, ...) \
  do { \
    if (is_binary_pointer (WATCH)) { \
      unsigned LIT = lit_pointer (WATCH); \
      unsigned OTHER = other_pointer (WATCH); \
      bool REDUNDANT = redundant_pointer (WATCH); \
      LOGBINARY (REDUNDANT, LIT, OTHER, __VA_ARGS__); \
    } else \
      LOGCLAUSE (get_clause (ring, WATCH), __VA_ARGS__); \
  } while (0)

#define ROGPREFIX(...) \
  if (verbosity < INT_MAX) \
    break; \
  acquire_message_lock (); \
  printf ("c LOG - "); \
  printf (__VA_ARGS__)

#define ROGSUFFIX LOGSUFFIX

#define ROG(...) \
  do { \
    ROGPREFIX (__VA_ARGS__); \
    ROGSUFFIX (); \
  } while (0)

#define LOG(...) \
  do { \
    LOGPREFIX (__VA_ARGS__); \
    LOGSUFFIX (); \
  } while (0)

#define ROGBINARY(LIT, OTHER, ...) \
  do { \
    ROGPREFIX (__VA_ARGS__); \
    printf (" irredundant"); \
    printf (" binary clause %s %s", ROGLIT (LIT), ROGLIT (OTHER)); \
    ROGSUFFIX (); \
  } while (0)

#define ROGCLAUSE(CLAUSE, ...) \
  do { \
    if (is_binary_pointer (CLAUSE)) { \
      assert (!redundant_pointer (CLAUSE)); \
      unsigned LIT = lit_pointer (CLAUSE); \
      unsigned OTHER = other_pointer (CLAUSE); \
      ROGBINARY (LIT, OTHER, __VA_ARGS__); \
    } else { \
      ROGPREFIX (__VA_ARGS__); \
      if ((CLAUSE)->garbage) \
        printf (" garbage"); \
      if ((CLAUSE)->redundant) \
        printf (" redundant glue %u", (CLAUSE)->glue); \
      else \
        printf (" irredundant"); \
      printf (" size %u clause[%" PRIu64 "]", (CLAUSE)->size, \
              (uint64_t) (CLAUSE)->id); \
      for (all_literals_in_clause (LIT, (CLAUSE))) \
        printf (" %s", ROGLIT (LIT)); \
      ROGSUFFIX (); \
    } \
  } while (0)

#else

#define LOG(...) \
  do { \
  } while (0)
#define LOGTMP(...) \
  do { \
  } while (0)
#define LOGBINARY(...) \
  do { \
  } while (0)
#define LOGCLAUSE(...) \
  do { \
  } while (0)
#define LOGWATCH(...) \
  do { \
  } while (0)

#define ROG(...) \
  do { \
  } while (0)
#define ROGBINARY(...) \
  do { \
  } while (0)
#define ROGCLAUSE(...) \
  do { \
  } while (0)

#endif

#endif

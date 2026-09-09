#ifndef _messages_h_INCLUDED
#define _messages_h_INCLUDED

#include <stdint.h>
#include <unistd.h>

void die (const char *, ...) __attribute__ ((format (printf, 1, 2)));
void fatal_error (const char *, ...)
    __attribute__ ((format (printf, 1, 2)));

#ifdef QUIET

static const int verbosity = -1;

#define PRINTLN(...) \
  do { \
  } while (0)

#define acquire_message_lock() \
  do { \
  } while (0)
#define release_message_lock() \
  do { \
  } while (0)
#define message(...) \
  do { \
  } while (0)
#define verbose(...) \
  do { \
  } while (0)
#define very_verbose(...) \
  do { \
  } while (0)
#define extremely_verbose(...) \
  do { \
  } while (0)

#else

extern int verbosity;
extern const char *prefix_format;

#ifdef LOGGING
#include <stdatomic.h>
// clang-format off
extern _Atomic (uint64_t) clause_ids;
// clang-format on
#endif

void acquire_message_lock (void);
void release_message_lock (void);

struct ring;

ssize_t print_line_without_acquiring_lock (struct ring *, const char *, ...)
    __attribute__ ((format (printf, 2, 3)));

void message (struct ring *ring, const char *, ...)
    __attribute__ ((format (printf, 2, 3)));

#define PRINTLN(...) print_line_without_acquiring_lock (ring, __VA_ARGS__)

#define verbose(...) \
  do { \
    if (verbosity > 1) \
      message (__VA_ARGS__); \
  } while (0)

#define very_verbose(...) \
  do { \
    if (verbosity > 2) \
      message (__VA_ARGS__); \
  } while (0)

#define extremely_verbose(...) \
  do { \
    if (verbosity > 3) \
      message (__VA_ARGS__); \
  } while (0)

#endif

#endif

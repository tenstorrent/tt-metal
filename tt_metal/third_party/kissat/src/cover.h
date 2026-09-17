#ifndef _cover_h_INCLUDED
#define _cover_h_INCLUDED

#include <stdio.h>
#include <stdlib.h>

#define COVER(COND) \
( \
  (COND) \
  ? \
  \
    ( \
      fflush (stdout), \
      fprintf (stderr, "%s:%ld: %s: Coverage goal `%s' reached.\n", \
	__FILE__, (long) __LINE__, __func__, #COND), \
      abort (), \
      (void) 0 \
    ) \
  : \
    (void) 0 \
)

#ifdef COVERAGE
#define FLUSH_COVERAGE() \
do { \
  void __gcov_flush (void); \
  __gcov_flush (); \
} while (0)
#else
#define FLUSH_COVERAGE() do { } while (0)
#endif

#endif

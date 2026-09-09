#include "types.h"
#include "clause.h"
#include "message.h"
#include "options.h"
#include "variable.h"
#include "walk.h"
#include "watches.h"

#include <stdatomic.h>
#include <stdio.h>

#define CHECK_TYPE(TYPE, BYTES) \
  do { \
    if (sizeof (TYPE) != (BYTES)) \
      fatal_error ("unsupported platform:\n" \
                   "'sizeof (" #TYPE ") == %zu' " \
                   "but expected 'sizeof (" #TYPE ") == %zu'", \
                   sizeof (TYPE), (size_t) (BYTES)); \
  } while (0)

void check_types (void) {
  CHECK_TYPE (signed char, 1);
  CHECK_TYPE (unsigned char, 1);
  CHECK_TYPE (atomic_bool, 1);

  CHECK_TYPE (unsigned short, 2);
  CHECK_TYPE (atomic_ushort, 2);

  CHECK_TYPE (unsigned, 4);
  CHECK_TYPE (int, 4);
  CHECK_TYPE (atomic_int, 4);

  CHECK_TYPE (size_t, 8);
  CHECK_TYPE (void *, 8);
  CHECK_TYPE (atomic_uintptr_t, 8);

  {
    if (MAX_THREADS & 7)
      fatal_error ("'MAX_THREADS = %u' not byte aligned", MAX_THREADS);
    size_t bytes_of_shared_field = sizeof ((struct clause *) 0)->shared;
    if ((MAX_THREADS >> 3) > (1u << (bytes_of_shared_field * 8 - 3)))
      fatal_error ("shared field of clauses with %zu bytes "
                   "does not fit 'MAX_THREADS = %u'",
                   bytes_of_shared_field, MAX_THREADS);
  }

  {
    size_t glue_in_clause_bytes = sizeof ((struct clause *) 0)->glue;
    if (1 << (glue_in_clause_bytes * 8) <= MAX_GLUE)
      fatal_error ("'MAX_GLUE = %u' exceeds 'sizeof (clause.glue) = %zu'",
                   MAX_GLUE, glue_in_clause_bytes);
  }

  {
    size_t glue_in_watcher_bytes = sizeof ((struct watcher *) 0)->glue;
    if (1 << (glue_in_watcher_bytes * 8) <= MAX_GLUE)
      fatal_error ("'MAX_GLUE = %u' exceeds 'sizeof (watcher.glue) = %zu'",
                   MAX_GLUE, glue_in_watcher_bytes);
  }

  if (verbosity > 0) {
    fputs ("c\n", stdout);
    printf ("c sizeof (struct clause) = %zu\n", sizeof (struct clause));
    printf ("c sizeof (struct counter) = %zu\n", sizeof (struct counter));
    printf ("c sizeof (struct phases) = %zu\n", sizeof (struct phases));
    printf ("c sizeof (struct variable) = %zu\n", sizeof (struct variable));
    printf ("c sizeof (struct watcher) = %zu\n", sizeof (struct watcher));
  }
}

#include "clause.h"
#include "logging.h"
#include "ring.h"
#include "tagging.h"
#include "trace.h"
#include "utilities.h"

#include <string.h>

struct clause *new_large_clause (size_t size, unsigned *literals,
                                 bool redundant, unsigned glue) {
  assert (2 <= size);
  size_t bytes = size * sizeof (unsigned);
  struct clause *clause = allocate_block (sizeof *clause + bytes);

#ifdef LOGGING
  clause->id = atomic_fetch_add (&clause_ids, 1);
#endif
  clause->shared = 0;
  clause->origin = -1;

  if (glue > MAX_GLUE)
    glue = MAX_GLUE;
  clause->glue = glue;

  clause->cleaned = false;
  clause->dirty = false;
  clause->garbage = false;
  clause->mapped = false;
  clause->padding = 0;
  clause->redundant = redundant;
  clause->subsume = false;
  clause->vivified = false;

  clause->size = size;

  memcpy (clause->literals, literals, bytes);

  return clause;
}

void mark_clause (signed char *marks, struct clause *clause,
                  unsigned except) {
  if (is_binary_pointer (clause))
    mark_literal (marks, other_pointer (clause));
  else
    for (all_literals_in_clause (other, clause))
      if (other != except)
        mark_literal (marks, other);
}

void unmark_clause (signed char *marks, struct clause *clause,
                    unsigned except) {
  if (is_binary_pointer (clause))
    unmark_literal (marks, other_pointer (clause));
  else
    for (all_literals_in_clause (other, clause))
      if (other != except)
        unmark_literal (marks, other);
}

void trace_add_clause (struct trace *trace, struct clause *clause) {
  assert (!is_binary_pointer (clause));
  trace_add_literals (trace, clause->size, clause->literals, INVALID);
}

void trace_delete_clause (struct trace *trace, struct clause *clause) {
  if (!clause->garbage)
    trace_delete_literals (trace, clause->size, clause->literals);
}

static void delete_clause (struct ring *ring, struct clause *clause) {
  assert (!is_binary_pointer (clause));
  LOGCLAUSE (clause, "delete");
  trace_delete_clause (&ring->trace, clause);
  free (clause);
}

void reference_clause (struct ring *ring, struct clause *clause,
                       unsigned inc) {
  assert (inc);
  assert (!is_binary_pointer (clause));
  unsigned shared = atomic_fetch_add (&clause->shared, inc);
  LOGCLAUSE (clause, "reference %u times (was shared %u)", inc, shared);
  assert (shared < MAX_THREADS - inc), (void) shared;
}

bool dereference_clause (struct ring *ring, struct clause *clause) {
  assert (!is_binary_pointer (clause));
  unsigned shared = atomic_fetch_sub (&clause->shared, 1);
  LOGCLAUSE (clause, "dereference once (was shared %u)", shared);
  assert (shared + 1);
  if (shared)
    return false;
  delete_clause (ring, clause);
  return true;
}

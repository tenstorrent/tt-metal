#include "deduplicate.h"
#include "message.h"
#include "ruler.h"
#include "simplify.h"
#include "trace.h"
#include "utilities.h"

static size_t
remove_duplicated_binaries_of_literal (struct simplifier *simplifier,
                                       unsigned lit) {
  struct ruler *ruler = simplifier->ruler;
  ruler->statistics.ticks.subsumption++;
  struct clauses *clauses = &OCCURRENCES (lit);
  struct clause **begin = clauses->begin, **q = begin;
  struct clause **end = clauses->end, **p = q;
  signed char *values = (signed char *) ruler->values;
  assert (!values[lit]);
  signed char *marks = simplifier->marks;
  size_t removed = 0;
  ruler->statistics.ticks.subsumption += cache_lines (end, begin);
  while (p != end) {
    struct clause *clause = *q++ = *p++;
    if (!is_binary_pointer (clause))
      continue;
    unsigned other = other_pointer (clause);
    if (values[other])
      continue;
    signed char mark = marked_literal (marks, other);
    if (!mark)
      mark_literal (marks, other);
    else if (mark > 0) {
      q--;
      ROGBINARY (lit, other, "removed duplicated");
      assert (ruler->statistics.binaries);
      ruler->statistics.binaries--;
      trace_delete_binary (&ruler->trace, lit, other);
      struct clause *other_clause = tag_binary (false, other, lit);
      disconnect_literal (ruler, other, other_clause);
      mark_eliminate_literal (simplifier, other);
      ruler->statistics.deduplicated++;
      ruler->statistics.subsumed++;
      removed++;
    } else {
      assert (mark < 0);
      ROG ("binary clause %s %s and %s %s yield unit %s", ROGLIT (lit),
           ROGLIT (NOT (other)), ROGLIT (lit), ROGLIT (other),
           ROGLIT (lit));
      trace_add_unit (&ruler->trace, lit);
      assign_ruler_unit (ruler, lit);
      while (p != end)
        *q++ = *p++;
      break;
    }
  }
  clauses->end = q;
  for (all_clauses (clause, *clauses))
    if (is_binary_pointer (clause))
      marks[IDX (other_pointer (clause))] = 0;
  if (removed)
    mark_eliminate_literal (simplifier, lit);
  return removed;
}

bool remove_duplicated_binaries (struct simplifier *simplifier,
                                 unsigned round) {
  struct ruler *ruler = simplifier->ruler;
  if (!ruler->options.deduplicate)
    return false;
#ifndef QUIET
  double start_deduplication = START (ruler, deduplicate);
#endif
  bool *eliminated = simplifier->eliminated;
  signed char *values = (signed char *) ruler->values;
  unsigned units_before = ruler->statistics.fixed.total;
  size_t removed = 0;
  for (all_ruler_literals (lit)) {
    if (ruler->terminate)
      break;
    if (values[lit])
      continue;
    if (eliminated[IDX (lit)])
      continue;
    removed += remove_duplicated_binaries_of_literal (simplifier, lit);
    if (ruler->inconsistent)
      break;
  }
  unsigned units_after = ruler->statistics.fixed.total;
  if (units_after > units_before)
    verbose (0, "[%u] deduplicating found %u units", round,
             units_after - units_before);
#ifndef QUIET
  double stop_deduplication = STOP (ruler, deduplicate);
  message (0,
           "[%u] removed %zu duplicated binary clauses %.0f%% "
           "in %.2f seconds",
           round, removed, percent (removed, ruler->statistics.original),
           stop_deduplication - start_deduplication);
#endif
  return removed;
}

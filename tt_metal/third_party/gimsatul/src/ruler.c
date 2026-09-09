#include "ruler.h"
#include "message.h"
#include "pthread.h"
#include "simplify.h"
#include "trace.h"
#include "utilities.h"

#include <string.h>

/*------------------------------------------------------------------------*/

#ifndef QUIET

void init_ruler_profiles (struct ruler *ruler) {
#define RULER_PROFILE(NAME) INIT_PROFILE (ruler, NAME);
  RULER_PROFILES
#undef RULER_PROFILE
  START (ruler, total);
}

#endif

static void init_locks (struct ruler *ruler) {
#define LOCK(NAME) pthread_mutex_init (&ruler->locks.NAME, 0);
  LOCKS
#undef LOCK
}

struct ruler *new_ruler (size_t size, struct options *opts) {
  assert (0 < opts->threads);
  assert (opts->threads <= MAX_THREADS);
  struct ruler *ruler = allocate_and_clear_block (sizeof *ruler);
  ruler->size = ruler->compact = size;

  ruler->eliminate = allocate_block (size);
  ruler->subsume = allocate_block (size);
  memset (ruler->eliminate, 1, size);
  memset (ruler->subsume, 1, size);

  init_locks (ruler);

  ruler->occurrences =
      allocate_and_clear_array (2 * size, sizeof *ruler->occurrences);
  ruler->values = allocate_and_clear_block (2 * size);

#ifndef NDEBUG
  ruler->original = allocate_and_clear_block (sizeof *ruler->original);
#endif
  ruler->units.begin = allocate_array (size, sizeof (unsigned));
  ruler->units.propagate = ruler->units.end = ruler->units.begin;

  ruler->trace.binary = opts->binary;
  ruler->trace.file = opts->proof.file ? &opts->proof : 0;

  memcpy (&ruler->options, opts, sizeof *opts);
#ifndef QUIET
  init_ruler_profiles (ruler);
#endif
  ruler->statistics.active = size;

  return ruler;
}

/*------------------------------------------------------------------------*/

static void release_occurrences (struct ruler *ruler) {
  if (!ruler->occurrences)
    return;
  for (all_ruler_literals (lit))
    RELEASE (OCCURRENCES (lit));
  free (ruler->occurrences);
}

static void release_clauses (struct ruler *ruler) {
  for (all_clauses (clause, ruler->clauses))
    if (!is_binary_pointer (clause))
      free (clause);
  RELEASE (ruler->clauses);
}

#ifndef NDEBUG

static void release_original (struct ruler *ruler) {
  RELEASE (*ruler->original);
  free (ruler->original);
}

#endif

void delete_ruler (struct ruler *ruler) {
  free (ruler->eliminate);
  free (ruler->subsume);

  release_occurrences (ruler);
  free (ruler->threads);
  free (ruler->unmap);
  free ((void *) ruler->values);

  release_clauses (ruler);
  RELEASE (ruler->extension[0]);
  RELEASE (ruler->extension[1]);
#ifndef NDEBUG
  release_original (ruler);
#endif
  RELEASE (ruler->rings);
  free (ruler->units.begin);

  RELEASE (ruler->trace.buffer);

  free (ruler);
}

/*------------------------------------------------------------------------*/

void flush_large_clause_occurrences (struct ruler *ruler) {
  ROG ("flushing large clauses occurrences");
#ifndef QUIET
  size_t flushed = 0;
#endif
  for (all_ruler_literals (lit)) {
    struct clauses *clauses = &OCCURRENCES (lit);
    struct clause **begin = clauses->begin, **q = begin;
    struct clause **end = clauses->end, **p = q;
    while (p != end) {
      struct clause *clause = *p++;
      if (is_binary_pointer (clause))
        *q++ = clause;
      else {
#ifndef QUIET
        flushed++;
#endif
      }
    }
    clauses->end = q;
  }
  very_verbose (0, "flushed %zu large clause occurrences", flushed);
}

static void connect_ruler_binary (struct ruler *ruler, unsigned lit,
                                  unsigned other) {
  struct clauses *clauses = &OCCURRENCES (lit);
  struct clause *watch_lit = tag_binary (false, lit, other);
  PUSH (*clauses, watch_lit);
}

void new_ruler_binary_clause (struct ruler *ruler, unsigned lit,
                              unsigned other) {
  ROGBINARY (lit, other, "new");
  connect_ruler_binary (ruler, lit, other);
  connect_ruler_binary (ruler, other, lit);
  ruler->statistics.binaries++;
}

void disconnect_literal (struct ruler *ruler, unsigned lit,
                         struct clause *clause) {
  ROGCLAUSE (clause, "disconnecting %s from", ROGLIT (lit));
  struct clauses *clauses = &OCCURRENCES (lit);
  struct clause **begin = clauses->begin, **q = begin;
  struct clause **end = clauses->end, **p = q;
  uint64_t ticks = 1 + cache_lines (end, begin);
  if (ruler->eliminating)
    ruler->statistics.ticks.elimination += ticks;
  if (ruler->subsuming)
    ruler->statistics.ticks.subsumption += ticks;
  while (p != end) {
    struct clause *other_clause = *q++ = *p++;
    if (other_clause == clause) {
      q--;
      break;
    }
  }
  while (p != end)
    *q++ = *p++;
  assert (q + 1 == p);
  clauses->end = q;
  if (EMPTY (*clauses))
    RELEASE (*clauses);
}

void connect_large_clause (struct ruler *ruler, struct clause *clause) {
  assert (!is_binary_pointer (clause));
  for (all_literals_in_clause (lit, clause))
    connect_literal (ruler, lit, clause);
}

void assign_ruler_unit (struct ruler *ruler, unsigned unit) {
  signed char *values = (signed char *) ruler->values;
  unsigned not_unit = NOT (unit);
  assert (!values[unit]);
  assert (!values[not_unit]);
  values[unit] = 1;
  values[not_unit] = -1;
  assert (ruler->units.end < ruler->units.begin + ruler->size);
  *ruler->units.end++ = unit;
  ROG ("assign %s unit", ROGLIT (unit));
  if (ruler->simplifying)
    ruler->statistics.fixed.simplifying++;
  if (ruler->solving)
    ruler->statistics.fixed.solving++;
  ruler->statistics.fixed.total++;
  assert (ruler->statistics.active);
  ruler->statistics.active--;
}

void recycle_clause (struct simplifier *simplifier, struct clause *clause,
                     unsigned lit) {
  struct ruler *ruler = simplifier->ruler;
  if (is_binary_pointer (clause)) {
    assert (lit == lit_pointer (clause));
    assert (!redundant_pointer (clause));
    unsigned other = other_pointer (clause);
    struct clause *other_clause = tag_binary (false, other, lit);
    disconnect_literal (ruler, other, other_clause);
    ROGBINARY (lit, other, "disconnected and deleted");
    assert (ruler->statistics.binaries);
    ruler->statistics.binaries--;
    trace_delete_binary (&ruler->trace, lit, other);
    mark_eliminate_literal (simplifier, other);
  } else {
    ROGCLAUSE (clause, "disconnecting and marking garbage");
    trace_delete_clause (&ruler->trace, clause);
    ruler->statistics.garbage++;
    clause->garbage = true;
    for (all_literals_in_clause (other, clause))
      if (other != lit)
        mark_eliminate_literal (simplifier, other);
  }
}

void recycle_clauses (struct simplifier *simplifier,
                      struct clauses *clauses, unsigned except) {
#ifdef LOGGING
  struct ruler *ruler = simplifier->ruler;
  ROG ("disconnecting and deleting clauses with %s", ROGLIT (except));
#endif
  for (all_clauses (clause, *clauses))
    recycle_clause (simplifier, clause, except);
  RELEASE (*clauses);
}

/*------------------------------------------------------------------------*/

void push_ring (struct ruler *ruler, struct ring *ring) {
  if (pthread_mutex_lock (&ruler->locks.rings))
    fatal_error ("failed to acquire rings lock while pushing ring");
  size_t id = SIZE (ruler->rings);
  PUSH (ruler->rings, ring);
  if (pthread_mutex_unlock (&ruler->locks.rings))
    fatal_error ("failed to release rings lock while pushing ring");
  assert (id < MAX_THREADS);
  ring->id = id;
  ring->random = ring->id;
  ring->ruler = ruler;
  ring->ruler_units = ruler->units.end;
  ring->trace.unmap = ruler->unmap;
}

void detach_ring (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
  if (pthread_mutex_lock (&ruler->locks.rings))
    fatal_error ("failed to acquire rings lock while detaching ring");
  assert (ring->id < SIZE (ruler->rings));
  assert (ruler->rings.begin[ring->id] == ring);
  ruler->rings.begin[ring->id] = 0;
  if (pthread_mutex_unlock (&ruler->locks.rings))
    fatal_error ("failed to release rings lock while detaching ring");
}

/*------------------------------------------------------------------------*/

void set_terminate (struct ruler *ruler, struct ring *ring) {
  bool terminated;
  if (pthread_mutex_lock (&ruler->locks.terminate))
    fatal_error ("failed to acquire terminate lock");
  terminated = ruler->terminate;
  if (!terminated)
    ruler->terminate = true;
  if (pthread_mutex_unlock (&ruler->locks.terminate))
    fatal_error ("failed to release terminate lock");

  // Always force to disable barriers to avoid races.

  abort_waiting_and_disable_barrier (&ruler->barriers.start);
  abort_waiting_and_disable_barrier (&ruler->barriers.import);
  abort_waiting_and_disable_barrier (&ruler->barriers.unclone);

  if (terminated)
    return;

  if (ring)
    verbose (ring, "termination forced by ring %u", ring->id);
  else
    verbose (0, "termination forced globally");
}

void set_winner (struct ring *ring) {
  volatile struct ring *winner;
  struct ruler *ruler = ring->ruler;
  bool winning;
  if (pthread_mutex_lock (&ruler->locks.winner))
    fatal_error ("failed to acquire winner lock");
  winner = ruler->winner;
  assert (winner != ring);
  winning = !winner;
  if (winning)
    ruler->winner = ring;
  if (pthread_mutex_unlock (&ruler->locks.winner))
    fatal_error ("failed to release winner lock");
  set_terminate (ruler, ring); // Always force termination to avoid race.
  if (!winning) {
    assert (winner);
    assert (winner->status == ring->status);
  } else
    verbose (ring, "winning ring[%u] status %d", ring->id, ring->status);
}

/*------------------------------------------------------------------------*/

struct ring *first_ring (struct ruler *ruler) {
  if (pthread_mutex_lock (&ruler->locks.rings))
    fatal_error ("failed to acquire rings lock while getting first");
  assert (!EMPTY (ruler->rings));
  struct ring *first = ruler->rings.begin[0];
  if (pthread_mutex_unlock (&ruler->locks.rings))
    fatal_error ("failed to release rings lock while getting first");
  return first;
}

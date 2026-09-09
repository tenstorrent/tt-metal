#include "simplify.h"
#include "backtrack.h"
#include "clone.h"
#include "compact.h"
#include "deduplicate.h"
#include "eliminate.h"
#include "export.h"
#include "import.h"
#include "message.h"
#include "propagate.h"
#include "report.h"
#include "scale.h"
#include "search.h"
#include "substitute.h"
#include "subsume.h"
#include "trace.h"
#include "unclone.h"
#include "utilities.h"

#include <inttypes.h>
#include <string.h>

#include "cover.h"

struct simplifier *new_simplifier (struct ruler *ruler) {
  size_t size = ruler->compact;
  struct simplifier *simplifier =
      allocate_and_clear_block (sizeof *simplifier);
  simplifier->ruler = ruler;
  simplifier->marks = allocate_and_clear_block (2 * size);
  simplifier->eliminated = allocate_and_clear_block (size);
  return simplifier;
}

void delete_simplifier (struct simplifier *simplifier) {
  free (simplifier->marks);
  free (simplifier->eliminated);
  free (simplifier);
}

void add_resolvent (struct simplifier *simplifier) {
  struct ruler *ruler = simplifier->ruler;
  assert (!ruler->inconsistent);
  struct unsigneds *resolvent = &simplifier->resolvent;
  unsigned *literals = resolvent->begin;
  size_t size = SIZE (*resolvent);
  trace_add_literals (&ruler->trace, size, literals, INVALID);
  if (!size) {
    very_verbose (0, "%s", "empty resolvent");
    ruler->inconsistent = true;
  } else if (size == 1) {
    const unsigned unit = literals[0];
    ROG ("unit resolvent %s", ROGLIT (unit));
    assign_ruler_unit (ruler, unit);
  } else if (size == 2) {
    unsigned lit = literals[0];
    unsigned other = literals[1];
    new_ruler_binary_clause (ruler, lit, other);
    mark_subsume_literal (simplifier, other);
    mark_subsume_literal (simplifier, lit);
  } else {
    assert (size > 2);
    if (ruler->eliminating)
      ruler->statistics.ticks.elimination += size;
    struct clause *clause = new_large_clause (size, literals, false, 0);
    connect_large_clause (ruler, clause);
    mark_subsume_clause (simplifier, clause);
    PUSH (ruler->clauses, clause);
    ROGCLAUSE (clause, "new");
  }
}

/*------------------------------------------------------------------------*/

static bool ruler_propagate (struct simplifier *simplifier) {
  struct ruler *ruler = simplifier->ruler;
  signed char *values = (signed char *) ruler->values;
  struct ruler_trail *units = &ruler->units;
#ifndef QUIET
  size_t garbage = 0;
#endif
  while (!ruler->inconsistent && units->propagate != units->end) {
    unsigned lit = *units->propagate++;
    ROG ("propagating unit %s", ROGLIT (lit));
    unsigned not_lit = NOT (lit);
    struct clauses *clauses = &OCCURRENCES (not_lit);
    for (all_clauses (clause, *clauses)) {
      bool satisfied = false;
      unsigned unit = INVALID;
      unsigned non_false = 0;
      if (is_binary_pointer (clause)) {
        assert (lit_pointer (clause) == not_lit);
        unsigned other = other_pointer (clause);
        signed char value = values[other];
        if (value > 0)
          continue;
        if (value < 0) {
          ROGBINARY (not_lit, other, "conflict");
          goto CONFLICT;
        }
        ROGBINARY (not_lit, other, "unit %s forcing", ROGLIT (other));
        trace_add_unit (&ruler->trace, other);
        assign_ruler_unit (ruler, other);
        continue;
      }
      if (clause->garbage)
        continue;
      for (all_literals_in_clause (other, clause)) {
        signed char value = values[other];
        if (value > 0) {
          satisfied = true;
          break;
        }
        if (value < 0)
          continue;
        if (non_false++)
          break;
        unit = other;
      }
      if (!satisfied && !non_false) {
        ROGCLAUSE (clause, "conflict");
      CONFLICT:
        assert (!ruler->inconsistent);
        verbose (0, "%s", "propagation yields inconsistency");
        ruler->inconsistent = true;
        trace_add_empty (&ruler->trace);
        break;
      }
      if (!satisfied && non_false == 1) {
        ROGCLAUSE (clause, "unit %s forcing", ROGLIT (unit));
        assert (unit != INVALID);
        trace_add_unit (&ruler->trace, unit);
        assign_ruler_unit (ruler, unit);
        satisfied = true;
      }
      if (satisfied) {
        ROGCLAUSE (clause, "marking satisfied garbage");
        trace_delete_clause (&ruler->trace, clause);
        mark_eliminate_clause (simplifier, clause);
        ruler->statistics.garbage++;
        clause->garbage = true;
#ifndef QUIET
        garbage++;
#endif
      }
    }
  }
  very_verbose (0, "marked %zu garbage clause during propagation", garbage);
  return !ruler->inconsistent;
}

static void mark_satisfied_ruler_clauses (struct simplifier *simplifier) {
  struct ruler *ruler = simplifier->ruler;
  signed char *values = (signed char *) ruler->values;
#ifndef QUIET
  size_t marked_satisfied = 0, marked_dirty = 0;
#endif
  for (all_clauses (clause, ruler->clauses)) {
    if (clause->garbage)
      continue;
    bool satisfied = false, dirty = false;
    for (all_literals_in_clause (lit, clause)) {
      signed char value = values[lit];
      if (value > 0) {
        satisfied = true;
        break;
      }
      if (!dirty && value < 0)
        dirty = true;
    }
    if (satisfied) {
      ROGCLAUSE (clause, "marking satisfied garbage");
      trace_delete_clause (&ruler->trace, clause);
      mark_eliminate_clause (simplifier, clause);
      ruler->statistics.garbage++;
      clause->garbage = true;
#ifndef QUIET
      marked_satisfied++;
#endif
    } else if (dirty) {
      ROGCLAUSE (clause, "marking dirty");
      assert (!clause->dirty);
      clause->dirty = true;
#ifndef QUIET
      marked_dirty++;
#endif
    }
  }
  very_verbose (0,
                "found %zu additional large satisfied clauses and "
                "marked %zu dirty",
                marked_satisfied, marked_dirty);
}

static void
flush_garbage_and_satisfied_occurrences (struct simplifier *simplifier) {
  struct ruler *ruler = simplifier->ruler;
  signed char *values = (signed char *) ruler->values;
#ifndef QUIET
  size_t flushed = 0;
#endif
  size_t deleted = 0;
  for (all_ruler_literals (lit)) {
    signed char lit_value = values[lit];
    struct clauses *clauses = &OCCURRENCES (lit);
    struct clause **begin = clauses->begin, **q = begin;
    struct clause **end = clauses->end, **p = q;
    while (p != end) {
      struct clause *clause = *q++ = *p++;
      if (is_binary_pointer (clause)) {
        assert (lit_pointer (clause) == lit);
        unsigned other = other_pointer (clause);
        signed char other_value = values[other];
        if (other_value > 0 || lit_value > 0) {
          if (other < lit) {
            ROGBINARY (lit, other, "deleting satisfied");
            trace_delete_binary (&ruler->trace, lit, other);
            if (!lit_value)
              mark_eliminate_literal (simplifier, lit);
            if (!other_value)
              mark_eliminate_literal (simplifier, other);
            deleted++;
          }
#ifndef QUIET
          flushed++;
#endif
          q--;
        } else {
          assert (!lit_value);
          assert (!other_value);
        }
      } else if (clause->garbage) {
#ifndef QUIET
        flushed++;
#endif
        q--;
      }
    }
    if (lit_value) {
#ifndef QUIET
      flushed += q - begin;
#endif
      RELEASE (*clauses);
    } else
      clauses->end = q;
  }
  very_verbose (0, "flushed %zu garbage watches", flushed);
  very_verbose (0, "deleted %zu satisfied binary clauses", deleted);
  assert (deleted <= ruler->statistics.binaries);
  ruler->statistics.binaries -= deleted;
}

static void
delete_large_garbage_ruler_clauses (struct simplifier *simplifier) {
  struct ruler *ruler = simplifier->ruler;
  struct clauses *clauses = &ruler->clauses;
  struct clause **begin_clauses = clauses->begin, **q = begin_clauses;
  struct clause **end_clauses = clauses->end, **p = q;
#ifndef QUIET
  size_t deleted = 0;
  size_t shrunken = 0;
#endif
  signed char *values = (signed char *) ruler->values;
  bool trace = ruler->options.proof.file;
  struct unsigneds remove;
  INIT (remove);
  while (p != end_clauses) {
    struct clause *clause = *q++ = *p++;
    if (clause->garbage) {
      ROGCLAUSE (clause, "finally deleting");
      free (clause);
#ifndef QUIET
      deleted++;
#endif
      q--;
    } else if (clause->dirty) {
      assert (EMPTY (remove));
#ifndef QUIET
      shrunken++;
#endif
      ROGCLAUSE (clause, "shrinking dirty");
      unsigned *literals = clause->literals;
      unsigned old_size = clause->size;
      assert (old_size > 2);
      unsigned *end_literals = literals + old_size;
      unsigned *l = literals, *k = l;
      while (l != end_literals) {
        unsigned lit = *k++ = *l++;
        signed char value = values[lit];
        assert (value <= 0);
        if (trace)
          PUSH (remove, lit);
        if (value < 0)
          k--;
      }
      size_t new_size = k - literals;
      assert (1 < new_size);
      assert (new_size < old_size);
      clause->size = new_size;
      clause->dirty = false;
      ROGCLAUSE (clause, "shrunken dirty");
      if (trace) {
        assert (old_size == SIZE (remove));
        trace_add_clause (&ruler->trace, clause);
        trace_delete_literals (&ruler->trace, old_size, remove.begin);
        CLEAR (remove);
      }
      if (2 < new_size)
        mark_subsume_clause (simplifier, clause);
      else {
        unsigned lit = literals[0];
        unsigned other = literals[1];
        disconnect_literal (ruler, lit, clause);
        disconnect_literal (ruler, other, clause);
        ROGCLAUSE (clause, "deleting shrunken dirty");
        new_ruler_binary_clause (ruler, lit, other);
        mark_subsume_literal (simplifier, other);
        mark_subsume_literal (simplifier, lit);
        free (clause);
        q--;
      }
    }
  }
  clauses->end = q;
  if (trace)
    RELEASE (remove);
  very_verbose (0, "finally deleted %zu large garbage clauses", deleted);
  very_verbose (0, "shrunken %zu dirty clauses", shrunken);
}

static bool
propagate_and_flush_ruler_units (struct simplifier *simplifier) {
  if (!ruler_propagate (simplifier))
    return false;
  struct ruler *ruler = simplifier->ruler;
  struct ruler_last *last = &ruler->last;
  if (last->fixed != ruler->statistics.fixed.total)
    mark_satisfied_ruler_clauses (simplifier);
  if (last->fixed != ruler->statistics.fixed.total ||
      last->garbage != ruler->statistics.garbage) {
    flush_garbage_and_satisfied_occurrences (simplifier);
    delete_large_garbage_ruler_clauses (simplifier);
  }
  last->fixed = ruler->statistics.fixed.total;
  last->garbage = ruler->statistics.garbage;
  assert (!ruler->inconsistent);
  return true;
}

static void connect_all_large_clauses (struct ruler *ruler) {
  ROG ("connecting all large clauses");
  for (all_clauses (clause, ruler->clauses))
    connect_large_clause (ruler, clause);
}

static uint64_t add_saturated (uint64_t a, uint64_t b, uint64_t limit) {
  return (limit - a < b) ? limit : a + b;
}

static uint64_t multiply_saturated (uint64_t a, uint64_t b,
                                    uint64_t limit) {
  if (!a)
    return 0;
  return (limit / a < b) ? limit : a * b;
}

static void set_ruler_limits (struct ruler *ruler) {
  unsigned level = ruler->options.optimize;
  verbose (0, "simplification optimization level %u", level);

  uint64_t scale10 = 1, scale4 = 1, scale2 = 1;
  for (unsigned i = 0; i != level; i++) {
    scale10 = multiply_saturated (scale10, 10, UINT64_MAX);
    scale4 = multiply_saturated (scale4, 4, UINT64_MAX);
    scale2 = multiply_saturated (scale2, 2, UINT64_MAX);
  }

  if (level)
    verbose (0, "scaling all simplification ticks limits by %" PRIu64,
             scale10);
  else
    verbose (0, "keeping simplification ticks limits at their default");

  struct ruler_limits *limits = &ruler->limits;
  struct ruler_statistics *statistics = &ruler->statistics;

  bool initial = !limits->initialized;
  limits->initialized = true;

  {
    unsigned boost = 1;
    if (initial && ruler->options.simplify_boost) {
      boost = ruler->options.simplify_boost_ticks;
      verbose (0, "boosting ticks limits initially by%s factor of %u",
               (level ? " another" : ""), boost);
    }

    uint64_t search;
    if (EMPTY (ruler->rings)) {
      search = 0;
      assert (!ruler->last.search);
    } else {
      struct ring *first = first_ring (ruler);
      search = first->statistics.contexts[SEARCH_CONTEXT].ticks;
      search -= ruler->last.search;
    }

    {
      uint64_t effort = ELIMINATE_EFFORT * search;
      uint64_t ticks = MAX (effort, MIN_ABSOLUTE_FFORT);
      uint64_t delta = multiply_saturated (scale10, ticks, UINT64_MAX);
      uint64_t boosted = multiply_saturated (boost, delta, UINT64_MAX);
      uint64_t current = statistics->ticks.elimination;
      uint64_t limit = add_saturated (current, boosted, UINT64_MAX);
      limits->elimination = limit;
      verbose (0,
               "setting elimination limit to %" PRIu64
               " ticks after %" PRIu64,
               limit, boosted);
    }

    {
      uint64_t effort = SUBSUME_EFFORT * search;
      uint64_t base = 1e6 * ruler->options.subsume_ticks;
      uint64_t ticks = MAX (effort, base);
      uint64_t delta = multiply_saturated (scale10, ticks, UINT64_MAX);
      uint64_t boosted = multiply_saturated (boost, delta, UINT64_MAX);
      uint64_t current = statistics->ticks.subsumption;
      uint64_t limit = add_saturated (current, boosted, UINT64_MAX);
      limits->subsumption = limit;
      verbose (0,
               "setting subsumption limit to %" PRIu64
               " ticks after %" PRIu64,
               limit, boosted);
    }
  }

  {
    unsigned boost = 1;
    if (initial && ruler->options.simplify_boost) {
      boost = ruler->options.simplify_boost_rounds;
      verbose (0, "boosting round limits initially by%s factor of %u",
               (level ? " another" : ""), boost);
    }

    unsigned max_rounds = ruler->options.simplify_rounds;
    if (level || boost > 1) {
      unsigned scale = multiply_saturated (boost, scale4, UINT_MAX);
      max_rounds = multiply_saturated (max_rounds, scale, UINT_MAX);
      verbose (0, "running at most %u simplification rounds (scaled %u)",
               max_rounds, scale);
    } else
      verbose (0, "running at most %u simplification rounds (default)",
               max_rounds);
    limits->max_rounds = max_rounds;
  }

  if (initial) {
    {
      unsigned max_bound = ruler->options.eliminate_bound;
      if (level) {
        max_bound = multiply_saturated (max_bound, scale2, UINT_MAX);
        verbose (0, "maximum elimination bound %u (scaled %" PRIu64 ")",
                 max_bound, scale2);
      } else
        verbose (0, "maximum elimination bound %u (default)", max_bound);
      limits->max_bound = max_bound;
    }

    {
      unsigned clause_size_limit = ruler->options.clause_size_limit;
      if (level) {
        clause_size_limit =
            multiply_saturated (clause_size_limit, scale10, UINT_MAX);
        verbose (0, "clause size limit %u (scaled %" PRIu64 ")",
                 clause_size_limit, scale10);
      } else
        verbose (0, "clause size limit %u (default)", clause_size_limit);
      limits->clause_size_limit = clause_size_limit;
    }

    {
      unsigned occurrence_limit = ruler->options.occurrence_limit;
      if (level) {
        occurrence_limit =
            multiply_saturated (occurrence_limit, scale10, UINT_MAX);
        verbose (0, "occurrence limit %u (scaled %" PRIu64 ")",
                 occurrence_limit, scale10);
      } else
        verbose (0, "occurrence limit %u (default)", occurrence_limit);

      limits->occurrence_limit = occurrence_limit;
    }
  }

  verbose (0, "current elimination bound %u", ruler->limits.current_bound);
}

#ifndef QUIET

static size_t current_ruler_clauses (struct ruler *ruler) {
  return SIZE (ruler->clauses) + ruler->statistics.binaries;
}

#endif

static void push_ruler_units_to_extension_stack (struct ruler *ruler) {
  struct unsigneds *extension = &ruler->extension[1];
  unsigned *unmap = ruler->unmap;
#ifndef QUIET
  size_t pushed = 0;
#endif
  for (all_elements_on_stack (unsigned, lit, ruler->units)) {
    unsigned unmapped = unmap_literal (unmap, lit);
    PUSH (*extension, unmapped);
#ifndef QUIET
    pushed++;
#endif
  }
  verbose (0, "pushed %zu units on extension stack", pushed);
  ruler->units.end = ruler->units.propagate = ruler->units.begin;
}

static void
run_only_root_level_propagation (struct simplifier *simplifier) {
  message (0, "simplification #%" PRIu64 " by root-level propagation only",
           simplifier->ruler->statistics.simplifications);
  connect_all_large_clauses (simplifier->ruler);
  propagate_and_flush_ruler_units (simplifier);
}

static void run_full_blown_simplification (struct simplifier *simplifier) {
  struct ruler *ruler = simplifier->ruler;
#ifndef QUIET
  struct ruler_statistics *statistics = &ruler->statistics;
  uint64_t simplifications = statistics->simplifications;
  message (0, "starting full simplification #%" PRIu64, simplifications);
#endif
  connect_all_large_clauses (ruler);

  set_ruler_limits (ruler);

#ifndef QUIET
  struct {
    size_t clauses, variables;
    struct {
      uint64_t elimination, subsumption;
    } ticks;
  } before, after, delta;

  before.clauses = current_ruler_clauses (ruler);
  before.variables = statistics->active;
  before.ticks.elimination = statistics->ticks.elimination;
  before.ticks.subsumption = statistics->ticks.subsumption;
#endif

  unsigned max_rounds = ruler->limits.max_rounds;

  bool complete = false;

  for (unsigned round = 1; !complete && round <= max_rounds; round++) {
    if (ruler->terminate)
      break;

    complete = true;
    if (!propagate_and_flush_ruler_units (simplifier))
      break;

    if (equivalent_literal_substitution (simplifier, round))
      complete = false;
    if (!propagate_and_flush_ruler_units (simplifier))
      break;
    if (ruler->terminate)
      break;

    if (remove_duplicated_binaries (simplifier, round))
      complete = false;
    if (!propagate_and_flush_ruler_units (simplifier))
      break;
    if (ruler->terminate)
      break;

    if (subsume_clauses (simplifier, round))
      complete = false;
    if (!propagate_and_flush_ruler_units (simplifier))
      break;
    if (ruler->terminate)
      break;

    if (eliminate_variables (simplifier, round))
      complete = false;
    if (!propagate_and_flush_ruler_units (simplifier))
      break;
    if (elimination_ticks_limit_hit (simplifier))
      break;
    if (ruler->terminate)
      break;
  }

#ifndef QUIET
  message (0, 0);
  after.variables = statistics->active;
  assert (after.variables <= before.variables);
  delta.variables = before.variables - after.variables;

  message (0, "removed %zu variables %.0f%% with %zu remaining %.0f%%",
           delta.variables, percent (delta.variables, before.variables),
           after.variables, percent (after.variables, ruler->size));

  after.clauses = current_ruler_clauses (ruler);
  size_t original = statistics->original;

  if (after.clauses <= before.clauses) {
    delta.clauses = before.clauses - after.clauses;
    message (0, "removed %zu clauses %.0f%% with %zu remaining %.0f%%",
             delta.clauses, percent (delta.clauses, before.clauses),
             after.clauses, percent (after.clauses, original));
  } else {
    delta.clauses = after.clauses - before.clauses;
    message (0,
             "simplification ADDED %zu clauses %.0f%% "
             "with %zu remaining %.0f%%",
             delta.clauses, percent (delta.clauses, before.clauses),
             after.clauses, percent (after.clauses, original));
  }

  if (ruler->inconsistent)
    verbose (0, "simplification produced empty clause");

  after.ticks.elimination = statistics->ticks.elimination;
  after.ticks.subsumption = statistics->ticks.subsumption;

  delta.ticks.elimination =
      after.ticks.elimination - before.ticks.elimination;
  delta.ticks.subsumption =
      after.ticks.subsumption - before.ticks.subsumption;

  verbose (0, "elimination at %" PRIu64 " ticks used %" PRIu64 " ticks%s",
           after.ticks.elimination, delta.ticks.elimination,
           elimination_ticks_limit_hit (simplifier) ? " (limit hit)" : "");
  verbose (0, "subsumption at %" PRIu64 " ticks used %" PRIu64 " ticks%s",
           after.ticks.subsumption, delta.ticks.subsumption,
           subsumption_ticks_limit_hit (simplifier) ? " (limit hit)" : "");
#endif

  if (complete)
    try_to_increase_elimination_bound (ruler);
}

void simplify_ruler (struct ruler *ruler) {
  if (ruler->inconsistent)
    return;

#ifndef QUIET
  double start_simplification = START (ruler, simplify);
#endif
  assert (!ruler->simplifying);
  ruler->simplifying = true;

  struct simplifier *simplifier = new_simplifier (ruler);

  bool initially = !ruler->statistics.simplifications++;
  bool full_simplification = ruler->options.simplify;

  if (full_simplification) {
    if (initially && !ruler->options.simplify_initially)
      full_simplification = false;
    if (!initially && !ruler->options.simplify_regularly)
      full_simplification = false;
  }

  message (0, 0);

  if (full_simplification)
    run_full_blown_simplification (simplifier);
  else
    run_only_root_level_propagation (simplifier);

  push_ruler_units_to_extension_stack (ruler);
  compact_ruler (simplifier, initially);
  delete_simplifier (simplifier);

  assert (ruler->simplifying);
  ruler->simplifying = false;

#ifndef QUIET
  double end_simplification = STOP (ruler, simplify);
  if (full_simplification)
    message (0, 0);
  message (0, "simplification #%" PRIu64 " took %.2f seconds",
           ruler->statistics.simplifications,
           end_simplification - start_simplification);
  reset_report ();
#endif
}

static void trigger_synchronization (struct ring *ring) {
  if (!ring->id) {
    struct ruler *ruler = ring->ruler;
    if (pthread_mutex_lock (&ruler->locks.simplify))
      fatal_error ("failed to acquire simplify lock during starting");
    assert (!ruler->simplify);
    ruler->simplify = true;
    if (pthread_mutex_unlock (&ruler->locks.simplify))
      fatal_error ("failed to release simplify lock during starting");
  }
}

static bool wait_to_actually_start_synchronization (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
  bool res = rendezvous (&ruler->barriers.start, ring, false);
  if (!ring->id) {
    if (pthread_mutex_lock (&ruler->locks.simplify))
      fatal_error ("failed to acquire simplify lock during preparation");
    assert (ruler->simplify);
    ruler->simplify = false;
    if (pthread_mutex_unlock (&ruler->locks.simplify))
      fatal_error ("failed to release simplify lock during preparation");
  }
  return res;
}

static bool continue_importing_and_propagating_units (struct ring *ring) {
  if (!ring->pool)
    return false;
  if (ring->inconsistent)
    return false;
  struct ruler *ruler = ring->ruler;
  if (ruler->terminate)
    return false;
  if (ruler->winner)
    return false;
  if (pthread_mutex_lock (&ruler->locks.units))
    fatal_error ("failed to acquire units lock during "
                 "simplification preparation");
  bool done = true;
  unsigned *ruler_units_end = ruler->units.end;
  for (all_rings (ring))
    if (ring->ruler_units != ruler_units_end) {
      done = false;
      break;
    }
  if (pthread_mutex_unlock (&ruler->locks.units))
    fatal_error ("failed to release units lock during "
                 "simplification preparation");
  return !done;
}

static bool synchronize_exported_and_imported_units (struct ring *ring) {
  flush_pool (ring);
  struct ruler *ruler = ring->ruler;

  if (!rendezvous (&ruler->barriers.import, ring, false))
    return false;

  assert (!ring->level);
  while (continue_importing_and_propagating_units (ring))
    if (import_shared (ring))
      if (!ring->inconsistent)
        if (ring_propagate (ring, false, 0))
          set_inconsistent (ring, "propagation after importing failed");

  assert (ring->inconsistent || ring->trail.propagate == ring->trail.end);

  return !ring->inconsistent;
}

static bool unclone_before_running_simplification (struct ring *ring) {
  if (!rendezvous (&ring->ruler->barriers.unclone, ring, false))
    return false;
  unclone_ring (ring);
  return true;
}

static void clone_first_ring_after_simplification (struct ring *ring) {
  assert (!ring->id);
  assert (ring->ruler->inconsistent || ring->references ||
          ring->ruler->terminate);
  copy_ruler (ring);
}

static void run_ring_simplification (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
  (void) rendezvous (&ruler->barriers.run, ring, true);
  if (ring->id)
    return;
  STOP (ruler, solve);
  simplify_ruler (ruler);
  START (ruler, solve);
  clone_first_ring_after_simplification (ring);
}

static void copy_other_ring_after_simplification (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
  (void) rendezvous (&ruler->barriers.copy, ring, true);
  if (!ring->id)
    return;
  if (ruler->inconsistent)
    return;
  if (ruler->terminate)
    return;
  assert (ring->references);
  copy_ring (ring);
}

static void finish_ring_simplification (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
  (void) rendezvous (&ruler->barriers.end, ring, true);
  if (ring->id)
    return;
  RELEASE (ruler->clauses);
  struct ring_limits *limits = &ring->limits;
  struct ring_statistics *statistics = &ring->statistics;
  uint64_t base = ring->options.simplify_interval;
  uint64_t interval = base * nlog2n (statistics->simplifications);
  uint64_t scaled = scale_interval (ring, "simplify", interval);
  limits->simplify = SEARCH_CONFLICTS + scaled;
  ruler->last.search = statistics->contexts[SEARCH_CONTEXT].ticks;
  very_verbose (
      ring, "new simplify limit at %" PRIu64 " after %" PRIu64 " conflicts",
      limits->simplify, scaled);
}

#ifndef NDEBUG
void check_clause_statistics (struct ring *);
void check_redundant_offset (struct ring *);
#endif

int simplify_ring (struct ring *ring) {
  if (ring->level)
    backtrack_propagate_iterate (ring);
  trigger_synchronization (ring);
  if (!wait_to_actually_start_synchronization (ring))
    return ring->status;
  if (!synchronize_exported_and_imported_units (ring))
    return ring->status;
  ring->trail.propagate = ring->trail.begin;
  if (!unclone_before_running_simplification (ring))
    return ring->status;
  ring->statistics.simplifications++;
  STOP_SEARCH ();
  run_ring_simplification (ring);
  copy_other_ring_after_simplification (ring);
  finish_ring_simplification (ring);
#ifndef NDEBUG
  if (!ring->ruler->inconsistent && !ring->ruler->terminate) {
    check_clause_statistics (ring);
    check_redundant_offset (ring);
  }
#endif
  report (ring, 's');
  START_SEARCH ();
  return ring->status;
}

bool simplifying (struct ring *ring) {
  if (!ring->options.simplify)
    return false;
  if (!ring->options.simplify_regularly)
    return false;
  if (!ring->id)
    return ring->limits.simplify <= SEARCH_CONFLICTS;
  struct ruler *ruler = ring->ruler;
#ifndef NFASTPATH
  if (!ruler->simplify)
    return false;
#endif
  if (pthread_mutex_lock (&ruler->locks.simplify))
    fatal_error ("failed to acquire simplify lock during checking");
  bool res = ruler->simplify;
  if (pthread_mutex_unlock (&ruler->locks.simplify))
    fatal_error ("failed to release simplify lock during checking");
  return res;
}

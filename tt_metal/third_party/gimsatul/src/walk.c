#include "walk.h"
#include "backtrack.h"
#include "clause.h"
#include "decide.h"
#include "logging.h"
#include "message.h"
#include "propagate.h"
#include "random.h"
#include "ruler.h"
#include "search.h"
#include "set.h"
#include "tagging.h"
#include "utilities.h"
#include "warm.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

struct doubles {
  double *begin, *end, *allocated;
};

struct counters {
  struct counter **begin, **end, **allocated;
  unsigned *binaries;
};

struct walker {
  struct ring *ring;
  struct counters *occurrences;
  struct counter *counters;
  struct set unsatisfied;
  struct unsigneds literals;
  struct unsigneds trail;
  struct watches saved;
  struct doubles scores;
  struct doubles breaks;
  unsigned maxbreak;
  double epsilon;
  size_t minimum;
  size_t initial;
  unsigned best;
  uint64_t limit;
  uint64_t extra;
  uint64_t flips;
};

#define COUNTERS(LIT) (walker->occurrences[LIT])

#define all_counters(ELEM, COUNTERS) \
  all_pointers_on_stack (struct counter, ELEM, COUNTERS)

#ifdef LOGGING

#define WOG(...) \
  do { \
    struct ring *ring = walker->ring; \
    LOG (__VA_ARGS__); \
  } while (0)

#else
#define WOG(...) \
  do { \
  } while (0)
#endif

static size_t
count_irredundant_non_garbage_clauses (struct ring *ring,
                                       struct clause **last_ptr) {
  size_t res = 0;
  struct clause *last = 0;
  for (all_watchers (watcher)) {
    if (watcher->garbage)
      continue;
    if (watcher->redundant)
      continue;
    struct clause *clause = watcher->clause;
    last = clause;
    res++;
  }
  *last_ptr = last;
  return res;
}

static double base_values[][2] = {{0.0, 2.00}, {3.0, 2.50}, {4.0, 2.85},
                                  {5.0, 3.70}, {6.0, 5.10}, {7.0, 7.40}};

static double interpolate_base (double size) {
  const size_t num_base_values = sizeof base_values / sizeof *base_values;
  size_t i = 0;
  while (i + 2 < num_base_values &&
         (base_values[i][0] > size || base_values[i + 1][0] < size))
    i++;
  double x2 = base_values[i + 1][0], x1 = base_values[i][0];
  double y2 = base_values[i + 1][1], y1 = base_values[i][1];
  double dx = x2 - x1, dy = y2 - y1;
  assert (dx);
  double res = dy * (size - x1) / dx + y1;
  assert (res > 0);
  if (res < 1.01)
    res = 1.01;
  return res;
}

static void initialize_break_table (struct walker *walker, double length) {
  struct ring *ring = walker->ring;
  verbose (ring, "average clause length %.2f", length);
  double epsilon = 1;
  unsigned maxbreak = 0;
  uint64_t walked = ring->statistics.walked;
  const double base = (walked & 1) ? 2.0 : interpolate_base (length);
  verbose (ring, "propability exponential sample base %.2f", base);
  assert (base > 1);
  for (;;) {
    double next = epsilon / base;
    if (!next)
      break;
    maxbreak++;
    PUSH (walker->breaks, epsilon);
    epsilon = next;
  }
  walker->epsilon = epsilon;
  walker->maxbreak = maxbreak;
  WOG ("epsilon score %g of %u break count and more", epsilon, maxbreak);
}

static void *min_max_tag_binary (bool redundant, unsigned first,
                                 unsigned second) {
  assert (first != second);
  unsigned min = first < second ? first : second;
  unsigned max = first < second ? second : first;
  return tag_binary (redundant, min, max);
}

static double connect_counters (struct walker *walker,
                                struct clause *last) {
  struct ring *ring = walker->ring;
  signed char *values = ring->values;
  struct counter *p = walker->counters;
  double sum_lengths = 0;
  size_t clauses = 0;
  uint64_t ticks = 1;
  volatile bool *terminate = &ring->ruler->terminate;
  for (all_watchers (watcher)) {
    ticks++;
    if (watcher->garbage)
      continue;
    if (watcher->redundant)
      continue;
    struct clause *clause = watcher->clause;
    unsigned length = 0;
    ticks++;
    for (all_literals_in_clause (lit, clause))
      if (values[lit])
        length++;
    if (!length) {
      LOGCLAUSE (clause, "WARNING: fully assigned");
      continue;
    }
    unsigned count = 0;
    for (all_literals_in_clause (lit, clause)) {
      signed char value = values[lit];
      if (!value)
        continue;
      count += (value > 0);
      PUSH (walker->occurrences[lit], p);
      ticks++;
    }
    sum_lengths += length;
    p->count = count;
    p->clause = clause;
    if (!count) {
      set_insert (&walker->unsatisfied, p);
      LOGCLAUSE (clause, "initially broken");
      ticks++;
    }
    clauses++;
    p++;
    if (clause == last)
      break;
    if (*terminate)
      break;
  }
  for (all_ring_literals (lit)) {
    signed char lit_value = values[lit];
    if (!lit_value)
      continue;
    struct counters *counters = &COUNTERS (lit);
    ticks++;
    unsigned *binaries = counters->binaries;
    if (!binaries)
      continue;
    unsigned *p, other;
    for (p = binaries; (other = *p) != INVALID; p++) {
      if (lit > other)
        continue;
      signed char other_value = values[other];
      if (!other_value)
        continue;
      sum_lengths += 2;
      clauses++;
      if (lit_value > 0)
        continue;
      if (other_value > 0)
        continue;
      LOGBINARY (false, lit, other, "initially broken");
      void *ptr = tag_binary (false, lit, other);
      assert (ptr == min_max_tag_binary (false, lit, other));
      set_insert (&walker->unsatisfied, ptr);
      ticks++;
    }
    ticks += cache_lines (p, binaries);
    if (*terminate)
      break;
  }

  very_verbose (ring, "connecting counters took %" PRIu64 " extra ticks",
                ticks);

  walker->extra += ticks;

  double average_length = average (sum_lengths, clauses);
  return average_length;
}

static void import_decisions (struct walker *walker) {
  struct ring *ring = walker->ring;
  assert (ring->context == WALK_CONTEXT);
  uint64_t saved = ring->statistics.contexts[WALK_CONTEXT].ticks;
  warming_up_saved_phases (ring);
  uint64_t extra = ring->statistics.contexts[WALK_CONTEXT].ticks - saved;
  walker->extra += extra;
  very_verbose (ring, "warming up needed %" PRIu64 " extra ticks", extra);
  signed char *values = ring->values;
#ifndef QUIET
  unsigned pos = 0, neg = 0, ignored = 0;
#endif
  struct variable *v = ring->variables;
  signed char *q = values;
  assert (!ring->level);
  for (all_ring_indices (idx)) {
    assert (q == values + LIT (idx));
    signed phase = decide_phase (ring, idx);
    assert (phase);
    if (*q) {
      phase = 0;
#ifndef QUIET
      ignored++;
#endif
    } else {
#ifndef QUIET
      pos += (phase > 0);
      neg += (phase < 0);
#endif
      v->level = INVALID_VAR;
    }
    *q++ = phase;
    *q++ = -phase;
    v++;
  }
  assert (q == values + 2 * ring->size);
  verbose (ring, "imported %u positive %u negative decisions (%u ignored)",
           pos, neg, ignored);
}

static void fix_values_after_local_search (struct ring *ring) {
  signed char *values = ring->values;
  memset (values, 0, 2 * ring->size);
  for (all_elements_on_stack (unsigned, lit, ring->trail)) {
    values[lit] = 1;
    values[NOT (lit)] = -1;
    VAR (lit)->level = 0;
  }
}

static void set_walking_limits (struct walker *walker) {
  struct ring *ring = walker->ring;
  struct ring_statistics *statistics = &ring->statistics;
  struct ring_last *last = &ring->last;
  uint64_t search = statistics->contexts[SEARCH_CONTEXT].ticks;
  uint64_t walk = statistics->contexts[WALK_CONTEXT].ticks;
  uint64_t ticks = search - last->walk;
  ticks = MAX (MIN_ABSOLUTE_FFORT, ticks);
  uint64_t extra = walker->extra;
  uint64_t effort = extra + WALK_EFFORT * ticks;
  walker->limit = walk + effort;
  very_verbose (ring,
                "walking effort %" PRIu64 " ticks = "
                "%" PRIu64 " + %g * %" PRIu64 " = %" PRIu64
                " + %g * (%" PRIu64 " - %" PRIu64 ")",
                effort, extra, (double) WALK_EFFORT, ticks, extra,
                (double) WALK_EFFORT, search, last->walk);
}

static size_t hash_counter_or_binary (void *state, void *ptr) {
  if (is_binary_pointer (ptr))
    return (size_t) ptr;
  struct counter *counters = state;
  struct counter *counter = ptr;
  assert (counters <= counter);
  size_t res = counter - counters;
  return res;
}

static struct walker *new_walker (struct ring *ring) {
  struct clause *last = 0;
  size_t clauses = count_irredundant_non_garbage_clauses (ring, &last);

  verbose (ring, "local search over %zu clauses %.0f%%", clauses,
           percent (clauses, ring->statistics.irredundant));

  struct walker *walker = allocate_and_clear_block (sizeof *walker);
  walker->ring = ring;

  import_decisions (walker);
  disconnect_references (ring, &walker->saved);

  walker->counters =
      allocate_and_clear_array (clauses, sizeof *walker->counters);
  walker->occurrences = (struct counters *) ring->references;

  walker->unsatisfied.hash.function = hash_counter_or_binary;
  walker->unsatisfied.hash.state = walker->counters;

  double length = connect_counters (walker, last);
  set_walking_limits (walker);
  initialize_break_table (walker, length);

  walker->initial = walker->minimum = walker->unsatisfied.size;
  verbose (ring, "initially %zu clauses unsatisfied", walker->minimum);

  return walker;
}

static void delete_walker (struct walker *walker) {
  struct ring *ring = walker->ring;
  free (walker->counters);
  free (walker->unsatisfied.table);
  RELEASE (walker->literals);
  RELEASE (walker->trail);
  RELEASE (walker->scores);
  RELEASE (walker->breaks);
  release_references (ring);
  reconnect_watches (ring, &walker->saved);
  RELEASE (walker->saved);
  free (walker);
}

static unsigned break_count (struct walker *walker, unsigned lit) {
  struct ring *ring = walker->ring;
  signed char *values = ring->values;
  unsigned not_lit = NOT (lit);
  assert (values[not_lit] > 0);
  unsigned res = 0;
  struct counters *counters = &COUNTERS (not_lit);
  unsigned *binaries = counters->binaries;
  uint64_t ticks = 1;
  if (binaries) {
    unsigned *p, other;
    for (p = binaries; (other = *p) != INVALID; p++)
      if (values[other] <= 0)
        res++;
    ticks += cache_lines (p, binaries);
  }
  for (all_counters (counter, *counters)) {
    ticks++;
    assert (!is_binary_pointer (counter));
    if (counter->count == 1)
      res++;
  }
  ring->statistics.contexts[WALK_CONTEXT].ticks += ticks;
  return res;
}

static double break_score (struct walker *walker, unsigned lit) {
  unsigned count = break_count (walker, lit);
  assert (SIZE (walker->breaks) == walker->maxbreak);
  double res;
  if (count >= walker->maxbreak)
    res = walker->epsilon;
  else
    res = walker->breaks.begin[count];
  WOG ("break count of %s is %u and score %g", LOGLIT (lit), count, res);
  return res;
}

static void save_all_values (struct walker *walker) {
  assert (walker->best == INVALID);
  struct ring *ring = walker->ring;
  signed char *q = ring->values;
  for (all_phases (p)) {
    signed char value = *q;
    q += 2;
    if (value)
      p->saved = value;
  }
  walker->best = 0;
}

static void save_walker_trail (struct walker *walker, bool keep) {
  assert (walker->best != INVALID);
  unsigned *begin = walker->trail.begin;
  unsigned *best = begin + walker->best;
  unsigned *end = walker->trail.end;
  assert (best <= end);
  struct ring *ring = walker->ring;
  struct phases *phases = ring->phases;
  for (unsigned *q = begin; q != best; q++) {
    unsigned lit = *q;
    signed phase = SGN (lit) ? -1 : 1;
    unsigned idx = IDX (lit);
    struct phases *p = phases + idx;
    p->saved = phase;
  }
  if (!keep)
    return;
  unsigned *q = begin;
  for (unsigned *p = best; p != end; p++)
    *q++ = *p;
  walker->trail.end = q;
  walker->best = 0;
}

static void save_final_minimum (struct walker *walker) {
#ifndef QUIET
  struct ring *ring = walker->ring;
#endif
  if (walker->minimum == walker->initial) {
    verbose (ring, "minimum number of unsatisfied clauses %zu unchanged",
             walker->minimum);
    return;
  }

  verbose (ring, "saving improved assignment of %zu unsatisfied clauses",
           walker->minimum);

  if (walker->best && walker->best != INVALID)
    save_walker_trail (walker, false);
}

static void push_flipped (struct walker *walker, unsigned flipped) {
  if (walker->best == INVALID)
    return;
  struct ring *ring = walker->ring;
  struct unsigneds *trail = &walker->trail;
  size_t size = SIZE (*trail);
  unsigned limit = ring->size / 4 + 1;
  if (size < limit)
    PUSH (*trail, flipped);
  else if (walker->best) {
    save_walker_trail (walker, true);
    PUSH (*trail, flipped);
  } else {
    CLEAR (*trail);
    walker->best = INVALID;
  }
}

static void new_minimium (struct walker *walker, unsigned unsatisfied) {
  very_verbose (walker->ring,
                "new minimum %u of unsatisfied clauses after %" PRIu64
                " flips",
                unsatisfied, walker->flips);
  walker->minimum = unsatisfied;
  if (walker->best == INVALID)
    save_all_values (walker);
  else
    walker->best = SIZE (walker->trail);
}

static void update_minimum (struct walker *walker, unsigned lit) {
  (void) lit;

  unsigned unsatisfied = walker->unsatisfied.size;
  WOG ("making literal %s gives %u unsatisfied clauses", LOGLIT (lit),
       unsatisfied);

  if (unsatisfied < walker->minimum)
    new_minimium (walker, unsatisfied);
}

static void make_literal (struct walker *walker, unsigned lit) {
  struct ring *ring = walker->ring;
  signed char *values = ring->values;
  assert (values[lit] > 0);
  uint64_t ticks = 1;
  struct counters *counters = &COUNTERS (lit);
  for (all_counters (counter, *counters)) {
    ticks++;
    assert (!is_binary_pointer (counter));
    if (counter->count++)
      continue;
    LOGCLAUSE (counter->clause, "literal %s makes", LOGLIT (lit));
    set_remove (&walker->unsatisfied, counter);
    ticks++;
  }
  unsigned *binaries = counters->binaries;
  if (binaries) {
    unsigned *p, other;
    for (p = binaries; (other = *p) != INVALID; p++)
      if (values[other] < 0) {
        LOGBINARY (false, lit, other, "literal %s makes", LOGLIT (lit));
        void *ptr = min_max_tag_binary (false, lit, other);
        set_remove (&walker->unsatisfied, ptr);
        ticks++;
      }
    ticks += cache_lines (p, binaries);
  }
  ring->statistics.contexts[WALK_CONTEXT].ticks += ticks;
}

static void break_literal (struct walker *walker, unsigned lit) {
  struct ring *ring = walker->ring;
  signed char *values = ring->values;
  assert (values[lit] < 0);
  uint64_t ticks = 1;
  struct counters *counters = &COUNTERS (lit);
  for (all_counters (counter, *counters)) {
    ticks++;
    assert (!is_binary_pointer (counter));
    assert (counter->count);
    if (--counter->count)
      continue;
    LOGCLAUSE (counter->clause, "literal %s breaks", LOGLIT (lit));
    set_insert (&walker->unsatisfied, counter);
    ticks++;
  }
  unsigned *binaries = counters->binaries;
  if (binaries) {
    ticks++;
    unsigned *p, other;
    for (p = binaries; (other = *p) != INVALID; p++)
      if (values[other] < 0) {
        LOGBINARY (false, lit, other, "literal %s breaks", LOGLIT (lit));
        void *ptr = min_max_tag_binary (false, lit, other);
        set_insert (&walker->unsatisfied, ptr);
        ticks++;
      }
    ticks += cache_lines (p, binaries);
  }
  ring->statistics.contexts[WALK_CONTEXT].ticks += ticks;
}

static void flip_literal (struct walker *walker, unsigned lit) {
  struct ring *ring = walker->ring;
  signed char *values = ring->values;
  assert (values[lit] < 0);
  ring->statistics.flips++;
  walker->flips++;
  unsigned not_lit = NOT (lit);
  values[lit] = 1, values[not_lit] = -1;
  break_literal (walker, not_lit);
  make_literal (walker, lit);
}

static unsigned pick_literal_to_flip (struct walker *walker, size_t size,
                                      unsigned *literals) {
  assert (EMPTY (walker->literals));
  assert (EMPTY (walker->scores));

  struct ring *ring = walker->ring;
  signed char *values = ring->values;

  unsigned res = INVALID;
  double total = 0, score = -1;

  unsigned *end = literals + size;

  for (unsigned *p = literals; p != end; p++) {
    unsigned lit = *p;
    if (!values[lit])
      continue;
    PUSH (walker->literals, lit);
    score = break_score (walker, lit);
    PUSH (walker->scores, score);
    total += score;
    res = lit;
  }

  double random = random_double (&ring->random);
  assert (0 <= random), assert (random < 1);
  double threshold = random * total;

  double sum = 0, *scores = walker->scores.begin;

  literals = walker->literals.begin;
  end = walker->literals.end;

  unsigned *p = literals;
  while (p != end) {
    unsigned other = *p++;
    if (!values[other])
      continue;
    double tmp = *scores++;
    sum += tmp;
    if (p != end && threshold >= sum)
      continue;
    res = other;
    score = tmp;
    break;
  }

  assert (res != INVALID);
  assert (score >= 0);

  CLEAR (walker->literals);
  CLEAR (walker->scores);

  LOG ("flipping literal %s with score %g", LOGLIT (res), score);

  return res;
}

static void walking_step (struct walker *walker) {
  struct set *unsatisfied = &walker->unsatisfied;
  struct ring *ring = walker->ring;
  struct counter *counter = random_set (&ring->random, unsatisfied);
  unsigned lit;
  if (is_binary_pointer (counter)) {
    unsigned first = lit_pointer (counter);
    unsigned second = other_pointer (counter);
    assert (!redundant_pointer (counter));
    unsigned literals[2] = {first, second};
    LOGBINARY (false, first, second, "picked broken");
    lit = pick_literal_to_flip (walker, 2, literals);
  } else {
    assert (!counter->count);
    struct clause *clause = counter->clause;
    LOGCLAUSE (clause, "picked broken");
    lit = pick_literal_to_flip (walker, clause->size, clause->literals);
  }
  flip_literal (walker, lit);
  push_flipped (walker, lit);
  update_minimum (walker, lit);
}

static void walking_loop (struct walker *walker) {
  struct ring *ring = walker->ring;
  uint64_t *ticks = &ring->statistics.contexts[WALK_CONTEXT].ticks;
  uint64_t limit = walker->limit;
  volatile bool *terminate = &ring->ruler->terminate;
#ifndef QUIET
  uint64_t ticks_before = *ticks;
#endif
  while (walker->minimum && *ticks <= limit && !*terminate)
    walking_step (walker);
#ifndef QUIET
  uint64_t ticks_after = *ticks;
  uint64_t ticks_delta = ticks_after - ticks_before;
  very_verbose (ring, "walking used %" PRIu64 " ticks", ticks_delta);
#endif
}

void local_search (struct ring *ring) {
  if (!backtrack_propagate_iterate (ring))
    return;
  STOP_SEARCH_AND_START (walk);
  assert (ring->context == SEARCH_CONTEXT);
  ring->context = WALK_CONTEXT;
  ring->statistics.walked++;
  if (ring->last.fixed != ring->statistics.fixed)
    mark_satisfied_watchers_as_garbage (ring);
  {
    struct walker *walker = new_walker (ring);
    walking_loop (walker);
    save_final_minimum (walker);
    verbose (ring, "walker flipped %" PRIu64 " literals", walker->flips);
    delete_walker (walker);
    fix_values_after_local_search (ring);
  }
  ring->last.walk = SEARCH_TICKS;
  assert (ring->context == WALK_CONTEXT);
  ring->context = SEARCH_CONTEXT;
  STOP_AND_START_SEARCH (walk);
}

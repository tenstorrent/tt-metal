#include "allocate.h"
#include "backtrack.h"
#include "decide.h"
#include "dense.h"
#include "inline.h"
#include "phases.h"
#include "print.h"
#include "rephase.h"
#include "report.h"
#include "propsearch.h"
#include "terminate.h"
#include "walk.h"

#include <string.h>

typedef struct tagged tagged;
typedef struct counter counter;
typedef struct walker walker;

#define LD_MAX_WALK_REF 31
#define MAX_WALK_REF ((1u << LD_MAX_WALK_REF) - 1)

struct tagged {
unsigned ref:
  LD_MAX_WALK_REF;
  bool binary: 1;
};

static inline tagged make_tagged(bool binary, unsigned ref) {
  assert(ref <= MAX_WALK_REF);
  tagged res = {.binary = binary, .ref = ref };
  return res;
}

struct counter {
  unsigned count;
  unsigned weight;
  unsigned pos;
};

// *INDENT-OFF*
typedef STACK (double) doubles;
// *INDENT-ON*

#define INVALID_BEST UINT_MAX

struct walker {
  kissat *solver;

  unsigned best;
  unsigned clauses;
  unsigned current;
  unsigned exponents;
  unsigned initial;
  unsigned minimum;
  unsigned offset;

  int weighted;

  generator random;

  counter *counters;
  litpairs *binaries;
  value *saved;
  tagged *refs;
  double *table;

  doubles scores;
  unsigneds unsat;
  unsigneds trail;

  double size;
  double epsilon;

  uint64_t limit;
  uint64_t flipped;
#ifndef QUIET
  uint64_t start;
  struct {
    uint64_t flipped;
    unsigned minimum;
  } report;
#endif
};

static const unsigned *dereference_literals(kissat *solver, walker *walker,
      unsigned counter_ref, unsigned *size_ptr) {
  assert(counter_ref < walker->clauses);
  tagged tagged = walker->refs[counter_ref];
  unsigned const *lits;
  if (tagged.binary) {
    const unsigned binary_ref = tagged.ref;
    lits = PEEK_STACK(*walker->binaries, binary_ref).lits;
    *size_ptr = 2;
  } else {
    const reference clause_ref = tagged.ref;
    clause *c = kissat_dereference_clause(solver, clause_ref);
    *size_ptr = c->size;
    lits = c->lits;
  }
  return lits;
}

static void push_unsat(kissat *solver, walker *walker,
      counter *counters, unsigned counter_ref) {
  assert(counter_ref < walker->clauses);
  counter *counter = counters + counter_ref;
  assert(SIZE_STACK(walker->unsat) <= UINT_MAX);
  counter->pos = SIZE_STACK(walker->unsat);
  PUSH_STACK(walker->unsat, counter_ref);
#ifdef LOGGING
  unsigned size;
  const unsigned *const lits = dereference_literals(solver, walker,
              counter_ref, &size);
  LOGLITS(size, lits, "pushed unsatisfied[%u]", counter->pos);
#endif
}

static bool pop_unsat(kissat *solver, walker *walker,
      counter *counters, unsigned counter_ref, unsigned pos) {
  assert(walker->current);
  assert(counter_ref < walker->clauses);
  assert(counters[counter_ref].pos == pos);
  assert(walker->current == SIZE_STACK(walker->unsat));
  const unsigned other_counter_ref = POP_STACK(walker->unsat);
  walker->current--;
  bool res = false;
  if (counter_ref != other_counter_ref) {
    assert(other_counter_ref < walker->clauses);
    counter *other_counter = counters + other_counter_ref;
    assert(other_counter->pos == walker->current);
    assert(pos < other_counter->pos);
    other_counter->pos = pos;
    POKE_STACK(walker->unsat, pos, other_counter_ref);
    res = true;
  }
#ifdef LOGGING
  unsigned size;
  const unsigned *const lits = dereference_literals(solver, walker,
              counter_ref, &size);
  LOGLITS(size, lits, "popped unsatisfied[%u]", pos);
#else
  (void) solver;
#endif
  return res;
}

static double cbvals[][2] = {
  {0.0, 2.00},
  {3.0, 2.50},
  {4.0, 2.85},
  {5.0, 3.70},
  {6.0, 5.10},
  {7.0, 7.40}
};

static double fit_cbval(double size) {
  const size_t num_cbvals = sizeof cbvals / sizeof * cbvals;
  size_t i = 0;
  while (i + 2 < num_cbvals
        && (cbvals[i][0] > size || cbvals[i + 1][0] < size)) {
    i++;
  }
  const double x2 = cbvals[i + 1][0], x1 = cbvals[i][0];
  const double y2 = cbvals[i + 1][1], y1 = cbvals[i][1];
  const double dx = x2 - x1, dy = y2 - y1;
  assert(dx);
  const double res = dy * (size - x1) / dx + y1;
  assert(res > 0);
  return res;
}

static void init_score_table(walker *walker) {
  kissat *solver = walker->solver;

  const double cb =
        (GET_OPTION(walkfit) && (solver->statistics.walks & 1)) ?
        fit_cbval(walker->size) : 2.0;
  const double base = 1 / cb;

  double next;
  unsigned exponents = 0;
  for (next = 1; next; next *= base) {
    exponents++;
  }

  walker->table = kissat_malloc(solver, exponents * sizeof(double));

  unsigned i = 0;
  double epsilon;
  for (epsilon = next = 1; next; next = epsilon * base) {
    walker->table[i++] = epsilon = next;
  }

  assert(i == exponents);
  walker->exponents = exponents;
  walker->epsilon = epsilon;

  kissat_phase(solver, "walk", GET(walks),
        "CB %.2f with inverse %.2f as base", cb, base);
  kissat_phase(solver, "walk", GET(walks),
        "table size %u and epsilon %g", exponents, epsilon);
}

static unsigned currently_unsatified(walker *walker) {
  return SIZE_STACK(walker->unsat);
}

static void import_decision_phases(walker *walker) {
  kissat *solver = walker->solver;
  INC(walk_decisions);
  value *const saved = solver->phases.saved;
  const value *const target = solver->phases.target;
  const value initial_phase = INITIAL_PHASE;
  const flags *const flags = solver->flags;
  const bool stable = solver->stable;
  value *values = solver->values;
#ifndef QUIET
  unsigned imported = 0;
  unsigned overwritten = 0;
#endif
  for (all_variables(idx)) {
    if (!flags[idx].active) {
      continue;
    }
    value value = 0;
    if (stable) {
      value = target[idx];
    }
    if (!value) {
      value = saved[idx];
    }
    if (!value) {
      value = initial_phase;
    }
    assert(value);
    if (saved[idx] != value) {
      saved[idx] = value;
#ifndef QUIET
      overwritten++;
#endif
    }
    const unsigned lit = LIT(idx);
    const unsigned not_lit = NOT(lit);
    values[lit] = value;
    values[not_lit] = -value;
#ifndef QUIET
    imported++;
#endif
    LOG("copied %s decision phase %d", LOGVAR(idx), (int) value);
    saved[idx] = value;
  }
  kissat_phase(solver, "walk", GET(walks),
        "imported %u decision phases %.0f%% (saved %u phases %.0f%%)",
        imported, kissat_percent(imported, solver->active),
        overwritten, kissat_percent(overwritten, solver->active));
}

static void import_previous_phases(walker *walker, const bits *bits) {
  kissat *solver = walker->solver;
  INC(walk_previous);
  value *const saved = solver->phases.saved;
  const value *const target = solver->phases.target;
  const value initial_phase = INITIAL_PHASE;
  const flags *const flags = solver->flags;
  const bool stable = solver->stable;
  value *values = solver->values;
  const unsigned vars = VARS;
#ifndef QUIET
  unsigned imported = 0;
  unsigned overwritten = 0;
#endif
  for (all_variables(idx)) {
    if (!flags[idx].active) {
      continue;
    }
    value value;
    if (bits) {
      value = kissat_get_bit(bits, vars, idx) ? 1 : -1;
    } else {
      value = 0;
    }
    if (!value && stable) {
      value = target[idx];
    }
    if (!value) {
      value = saved[idx];
    }
    if (!value) {
      value = initial_phase;
    }
    assert(value);
    if (saved[idx] != value) {
      saved[idx] = value;
#ifndef QUIET
      overwritten++;
#endif
    }
    const unsigned lit = LIT(idx);
    const unsigned not_lit = NOT(lit);
    values[lit] = value;
    values[not_lit] = -value;
#ifndef QUIET
    imported++;
#endif
    LOG("copied %s previous phase %d", LOGVAR(idx), (int) value);
    saved[idx] = value;
  }
  kissat_phase(solver, "walk", GET(walks),
        "imported %u previous phases %.0f%% (saved %u phases %.0f%%)",
        imported, kissat_percent(imported, solver->active),
        overwritten, kissat_percent(overwritten, solver->active));
}

static unsigned clause_weight(walker *walker, unsigned size) {
  const int weighted = walker->weighted;
  unsigned weight;
  if (weighted > 0) {
    weight = MIN(size, 5);
  } else if (weighted < 0 && size < 6) {
    weight = 7 - size;
  } else {
    weight = 1;
  }
  return weight;
}

static unsigned connect_binary_counters(walker *walker) {
  kissat *solver = walker->solver;
  value *values = solver->values;
  tagged *refs = walker->refs;
  watches *all_watches = solver->watches;
  counter *counters = walker->counters;

  assert(SIZE_STACK(*walker->binaries) <= UINT_MAX);
  const unsigned size = SIZE_STACK(*walker->binaries);
  litpair *binaries = BEGIN_STACK(*walker->binaries);
  unsigned unsat = 0, counter_ref = 0;

  const unsigned weight = clause_weight(walker, 2);

  for (unsigned binary_ref = 0; binary_ref < size; binary_ref++) {
    const litpair *const litpair = binaries + binary_ref;
    const unsigned first = litpair->lits[0];
    const unsigned second = litpair->lits[1];
    assert(first < LITS), assert(second < LITS);
    const value first_value = values[first];
    const value second_value = values[second];
    if (!first_value || !second_value) {
      continue;
    }
    assert(counter_ref < walker->clauses);
    refs[counter_ref] = make_tagged(true, binary_ref);
    watches *first_watches = all_watches + first;
    watches *second_watches = all_watches + second;
    kissat_push_large_watch(solver, first_watches, counter_ref);
    kissat_push_large_watch(solver, second_watches, counter_ref);
    const unsigned count = (first_value > 0) + (second_value > 0);
    counter *counter = counters + counter_ref;
    counter->count = count;
    counter->weight = weight;
    if (!count) {
      push_unsat(solver, walker, counters, counter_ref);
      unsat++;
    }
    counter_ref++;
  }
  kissat_phase(solver, "walk", GET(walks),
        "initially %u unsatisfied binary clauses %.0f%% out of %u",
        unsat, kissat_percent(unsat, counter_ref), counter_ref);
#ifdef QUIET
  (void) unsat;
#endif
  walker->size += 2.0 * counter_ref;
  return counter_ref;
}

static void connect_large_counters(walker *walker, unsigned counter_ref) {
  kissat *solver = walker->solver;
  assert(!solver->level);
  const value *const saved = walker->saved;
  const value *const values = solver->values;
  ward *const arena = BEGIN_STACK(solver->arena);
  counter *counters = walker->counters;
  tagged *refs = walker->refs;

  unsigned unsat = 0;
  unsigned large = 0;

  clause *last_irredundant = kissat_last_irredundant_clause(solver);

  for (all_clauses(c)) {
    if (last_irredundant && c > last_irredundant) {
      break;
    }
    if (c->garbage) {
      continue;
    }
    if (c->redundant) {
      continue;
    }
    bool continue_with_next_clause = false;
    for (all_literals_in_clause(lit, c)) {
      const value value = saved[lit];
      if (value <= 0) {
        continue;
      }
      LOGCLS(c, "%s satisfied", LOGLIT(lit));
      if (EMPTY_STACK(solver->assumption_implied)) {
        kissat_mark_clause_as_garbage(solver, false, c);
        assert(c->garbage);
      }
      continue_with_next_clause = true;
      break;
    }
    if (continue_with_next_clause) {
      continue;
    }
    large++;
    assert(kissat_clause_in_arena(solver, c));
    reference clause_ref = (ward *) c - arena;
    assert(clause_ref <= MAX_WALK_REF);
    assert(counter_ref < walker->clauses);
    refs[counter_ref] = make_tagged(false, clause_ref);
    unsigned count = 0, size = 0;
    for (all_literals_in_clause(lit, c)) {
      const value value = values[lit];
      if (!value) {
        assert(saved[lit] < 0);
        continue;
      }
      watches *watches = &WATCHES(lit);
      kissat_push_large_watch(solver, watches, counter_ref);
      size++;
      if (value > 0) {
        count++;
      }
    }
    counter *counter = walker->counters + counter_ref;
    counter->count = count;
    counter->weight = clause_weight(walker, size);

    if (!count) {
      push_unsat(solver, walker, counters, counter_ref);
      unsat++;
    }
    counter_ref++;
    walker->size += size;
  }
  kissat_phase(solver, "walk", GET(walks),
        "initially %u unsatisfied large clauses %.0f%% out of %u",
        unsat, kissat_percent(unsat, large), large);
#ifdef QUIET
  (void) large;
  (void) unsat;
#endif
}

#ifndef QUIET

static void report_initial_minimum(kissat *solver, walker *walker) {
  walker->report.minimum = walker->minimum;
  kissat_very_verbose(solver, "initial minimum of %u unsatisfied clauses",
        walker->minimum);
}

static void report_minimum(const char *type, kissat *solver, walker *walker) {
  assert(walker->minimum <= walker->report.minimum);
  kissat_very_verbose(solver,
        "%s minimum of %u unsatisfied clauses after %"
        PRIu64 " flipped literals", type,
        walker->minimum, walker->flipped);
  walker->report.minimum = walker->minimum;
}
#else
#define report_initial_minimum(...) do { } while (0)
#define report_minimum(...) do { } while (0)
#endif

static void init_walker(kissat *solver, walker *walker,
      litpairs *binaries, bool use_previous_phase) {
  assert(IRREDUNDANT_CLAUSES <= MAX_WALK_REF);
  const unsigned clauses = IRREDUNDANT_CLAUSES;

  memset(walker, 0, sizeof * walker);

  walker->solver = solver;
  walker->clauses = clauses;
  walker->binaries = binaries;
  walker->random = solver->random ^ solver->statistics.walks;

  walker->saved = solver->values;
  solver->values = kissat_calloc(solver, LITS, 1);

  for (all_stack(unsigned, lit, solver->assumption_implied)) {
    assert(!walker->saved[lit]);
    walker->saved[lit] = 1;
    walker->saved[NOT(lit)] = -1;
  }

  const bits *bits = 0;
  if (use_previous_phase && (bits = kissat_lookup_cache(solver))) {
    import_previous_phases(walker, bits);
  } else {
    import_decision_phases(walker);
  }

  if (GET_OPTION(walkweighted))
    switch (GET(walks) % 3) {
    case 1:
    UNIFORM:
      kissat_very_verbose(solver, "uniform clause weight one");
      walker->weighted = 0;
      break;
    case 2:
      kissat_very_verbose(solver, "monotone increasing clause weights");
      walker->weighted = 1;
      break;
    default:
      kissat_very_verbose(solver, "monotone decreasing clause weights");
      walker->weighted = -1;
      break;
    } else {
    goto UNIFORM;
  }

  walker->counters = kissat_malloc(solver, clauses * sizeof(counter));
  walker->refs = kissat_malloc(solver, clauses * sizeof(tagged));

  assert(!walker->size);
  const unsigned counter_ref = connect_binary_counters(walker);
  connect_large_counters(walker, counter_ref);

  walker->current = walker->initial = currently_unsatified(walker);

  kissat_phase(solver, "walk", GET(walks),
        "initially %u unsatisfied irredundant clauses %.0f%% "
        "out of %" PRIu64, walker->initial,
        kissat_percent(walker->initial, IRREDUNDANT_CLAUSES),
        IRREDUNDANT_CLAUSES);

  walker->size = kissat_average(walker->size, clauses);
  kissat_phase(solver, "walk", GET(walks),
        "average clause size %.2f", walker->size);

  walker->minimum = walker->current;
  init_score_table(walker);

  report_initial_minimum(solver, walker);

  if (bits) {
    kissat_update_cache(solver, walker->minimum);
  }
}

static void init_walker_limit(kissat *solver, walker *walker) {
  SET_EFFORT_LIMIT(limit, walk, walk_steps, 2 * CLAUSES);
  walker->limit = limit;
  walker->flipped = 0;
#ifndef QUIET
  walker->start = solver->statistics.walk_steps;
  walker->report.minimum = UINT_MAX;
  walker->report.flipped = 0;
#endif
}

static void release_walker(walker *walker) {
  kissat *solver = walker->solver;
  kissat_dealloc(solver, walker->table, walker->exponents, sizeof(double));
  unsigned clauses = walker->clauses;
  kissat_dealloc(solver, walker->refs, clauses, sizeof(tagged));
  kissat_dealloc(solver, walker->counters, clauses, sizeof(counter));
  RELEASE_STACK(walker->unsat);
  RELEASE_STACK(walker->scores);
  RELEASE_STACK(walker->trail);
  kissat_free(solver, solver->values, LITS);
  RELEASE_STACK(walker->unsat);
  for (all_stack(unsigned, lit, solver->assumption_implied)) {
    walker->saved[lit] = 0;
    walker->saved[NOT(lit)] = 0;
  }
  solver->values = walker->saved;
}

static unsigned break_value(kissat *solver, walker *walker, value *values,
      unsigned lit) {
  assert(values[lit] < 0);
  const unsigned not_lit = NOT(lit);
  watches *watches = &WATCHES(not_lit);
  unsigned steps = 1;
  unsigned res = 0;
  for (all_binary_large_watches(watch, *watches)) {
    steps++;
    assert(!watch.type.binary);
    reference counter_ref = watch.large.ref;
    assert(counter_ref < walker->clauses);
    counter *counter = walker->counters + counter_ref;
    if (counter->count == 1) {
      res += counter->weight;
    }
  }
  ADD(walk_steps, steps);
#ifdef NDEBUG
  (void) values;
#endif
  return res;
}

static double scale_score(walker *walker, unsigned breaks) {
  if (breaks < walker->exponents) {
    return walker->table[breaks];
  } else {
    return walker->epsilon;
  }
}

static unsigned pick_literal(kissat *solver, walker *walker) {
  assert(walker->current == SIZE_STACK(walker->unsat));
  const unsigned pos = walker->flipped++ % walker->current;
  const unsigned counter_ref = PEEK_STACK(walker->unsat, pos);
  unsigned size;
  const unsigned *const lits =
        dereference_literals(solver, walker, counter_ref, &size);

  LOGLITS(size, lits, "picked unsatisfied[%u]", pos);
  assert(EMPTY_STACK(walker->scores));

  value *values = solver->values;

  double sum = 0;
  unsigned picked_lit = INVALID_LIT;

  const unsigned *const end_of_lits = lits + size;
  for (const unsigned *p = lits; p != end_of_lits; p++) {
    const unsigned lit = *p;
    if (!values[lit]) {
      continue;
    }
    picked_lit = lit;
    const unsigned breaks = break_value(solver, walker, values, lit);
    const double score = scale_score(walker, breaks);
    assert(score > 0);
    LOG("literal %s breaks %u score %g", LOGLIT(lit), breaks, score);
    PUSH_STACK(walker->scores, score);
    sum += score;
  }
  assert(picked_lit != INVALID_LIT);
  assert(0 < sum);

  const double random = kissat_pick_double(&walker->random);
  assert(0 <= random), assert(random < 1);

  const double threshold = sum * random;
  LOG("score sum %g and random threshold %g", sum, threshold);

  // assert (threshold < sum); // NOT TRUE!!!!

  double *scores = BEGIN_STACK(walker->scores);
#ifdef LOGGING
  double picked_score = 0;
#endif

  sum = 0;

  for (const unsigned *p = lits; p != end_of_lits; p++) {
    const unsigned lit = *p;
    if (!values[lit]) {
      continue;
    }
    const double score = *scores++;
    sum += score;
    if (threshold < sum) {
      picked_lit = lit;
#ifdef LOGGING
      picked_score = score;
#endif
      break;
    }
  }
  assert(picked_lit != INVALID_LIT);
  LOG("picked literal %s with score %g", LOGLIT(picked_lit), picked_score);

  CLEAR_STACK(walker->scores);

  return picked_lit;
}

static void break_clauses(kissat *solver, walker *walker,
      const value *const values, unsigned flipped) {
#ifdef LOGGING
  unsigned broken = 0;
#endif
  const unsigned not_flipped = NOT(flipped);
  assert(values[not_flipped] < 0);
  LOG("breaking one-satisfied clauses containing negated flipped literal %s",
        LOGLIT(not_flipped));
  watches *watches = &WATCHES(not_flipped);
  counter *counters = walker->counters;
  unsigned steps = 1;
  for (all_binary_large_watches(watch, *watches)) {
    steps++;
    assert(!watch.type.binary);
    const unsigned counter_ref = watch.large.ref;
    assert(counter_ref < walker->clauses);
    counter *counter = counters + counter_ref;
    assert(counter->count);
    if (--counter->count) {
      continue;
    }
    push_unsat(solver, walker, counters, counter_ref);
#ifdef LOGGING
    broken++;
#endif
  }
  LOG("broken %u one-satisfied clauses containing "
        "negated flipped literal %s", broken, LOGLIT(not_flipped));
  ADD(walk_steps, steps);
#ifdef NDEBUG
  (void) values;
#endif
}

static void make_clauses(kissat *solver, walker *walker,
      const value *const values, unsigned flipped) {
  assert(values[flipped] > 0);
  LOG("making unsatisfied clauses containing flipped literal %s",
        LOGLIT(flipped));
  watches *watches = &WATCHES(flipped);
  counter *counters = walker->counters;
  unsigned steps = 1;
#ifdef LOGGING
  unsigned made = 0;
#endif
  for (all_binary_large_watches(watch, *watches)) {
    steps++;
    assert(!watch.type.binary);
    const unsigned counter_ref = watch.large.ref;
    assert(counter_ref < walker->clauses);
    counter *counter = counters + counter_ref;
    assert(counter->count < UINT_MAX);
    if (counter->count++) {
      continue;
    }
    if (pop_unsat(solver, walker, counters, counter_ref, counter->pos)) {
      steps++;
    }
#ifdef LOGGING
    made++;
#endif
  }
  LOG("made %u unsatisfied clauses containing flipped literal %s",
        made, LOGLIT(flipped));
  ADD(walk_steps, steps);
#ifdef NDEBUG
  (void) values;
#endif
}

static void save_all_values(kissat *solver, walker *walker) {
  assert(EMPTY_STACK(walker->trail));
  assert(walker->best == INVALID_BEST);
  LOG("copying all values as saved phases since trail is invalid");
  const value *const values = solver->values;
  value *saved = solver->phases.saved;
  for (all_variables(idx)) {
    const unsigned lit = LIT(idx);
    const value value = values[lit];
    if (value) {
      saved[idx] = value;
    }
  }
  LOG("reset best trail position to 0");
  walker->best = 0;
}

static void save_walker_trail(kissat *solver, walker *walker, bool keep) {
#if defined(LOGGING) || !defined(NDEBUG)
  assert(walker->best != INVALID_BEST);
  assert(SIZE_STACK(walker->trail) <= UINT_MAX);
  const unsigned size_trail = SIZE_STACK(walker->trail);
  assert(walker->best <= size_trail);
  const unsigned kept = size_trail - walker->best;
  LOG("saving %u values of flipped literals on trail of size %u",
        walker->best, size_trail);
#endif
  unsigned *begin = BEGIN_STACK(walker->trail);
  const unsigned *const best = begin + walker->best;
  value *saved = solver->phases.saved;
  for (const unsigned *p = begin; p != best; p++) {
    const unsigned lit = *p;
    const value value = NEGATED(lit) ? -1 : 1;
    const unsigned idx = IDX(lit);
    saved[idx] = value;
  }
  if (!keep) {
    LOG("no need to shift and keep remaining %u literals", kept);
    return;
  }
  LOG("flushed %u literals %.0f%% from trail",
        walker->best, kissat_percent(walker->best, size_trail));
  const unsigned *const end = END_STACK(walker->trail);
  unsigned *q = begin;
  for (const unsigned *p = best; p != end; p++) {
    *q++ = *p;
  }
  assert((size_t)(end - q) == walker->best);
  assert((size_t)(q - begin) == kept);
  SET_END_OF_STACK(walker->trail, q);
  LOG("keeping %u literals %.0f%% on trail",
        kept, kissat_percent(kept, size_trail));
  LOG("reset best trail position to 0");
  walker->best = 0;
}

static void push_flipped(kissat *solver, walker *walker, unsigned flipped) {
  if (walker->best == INVALID_BEST) {
    LOG("not pushing flipped %s to already invalid trail",
          LOGLIT(flipped));
    assert(EMPTY_STACK(walker->trail));
  } else {
    assert(SIZE_STACK(walker->trail) <= UINT_MAX);
    const unsigned size_trail = SIZE_STACK(walker->trail);
    assert(walker->best <= size_trail);
    const unsigned limit = VARS / 4 + 1;
    assert(limit < INVALID_BEST);
    if (size_trail < limit) {
      PUSH_STACK(walker->trail, flipped);
      LOG("pushed flipped %s to trail which now has size %u",
            LOGLIT(flipped), size_trail + 1);
    } else if (walker->best) {
      LOG("trail reached limit %u but has best position %u",
            limit, walker->best);
      save_walker_trail(solver, walker, true);
      PUSH_STACK(walker->trail, flipped);
      assert(SIZE_STACK(walker->trail) <= UINT_MAX);
      LOG("pushed flipped %s to trail which now has size %zu",
            LOGLIT(flipped), SIZE_STACK(walker->trail));
    } else {
      LOG("trail reached limit %u without best position", limit);
      CLEAR_STACK(walker->trail);
      LOG("not pushing %s to invalidated trail", LOGLIT(flipped));
      walker->best = INVALID_BEST;
      LOG("best trail position becomes invalid");
    }
  }
}

static void flip_literal(kissat *solver, walker *walker, unsigned flip) {
  LOG("flipping literal %s", LOGLIT(flip));
  value *values = solver->values;
  const value value = values[flip];
  assert(value < 0);
  values[flip] = -value;
  values[NOT(flip)] = value;
  make_clauses(solver, walker, values, flip);
  break_clauses(solver, walker, values, flip);
  walker->current = currently_unsatified(walker);
}

static void update_best(kissat *solver, walker *walker) {
  assert(walker->current < walker->minimum);
  walker->minimum = walker->current;
#ifndef QUIET
  int verbosity = kissat_verbosity(solver);
  bool report = (verbosity > 2);
  if (verbosity == 2) {
    if (walker->flipped / 2 >= walker->report.flipped) {
      report = true;
    } else if (walker->minimum < 5 ||
          walker->report.minimum == UINT_MAX ||
          walker->minimum <= walker->report.minimum / 2) {
      report = true;
    }
    if (report) {
      walker->report.minimum = walker->minimum;
      walker->report.flipped = walker->flipped;
    }
  }
  if (report) {
    report_minimum("new", solver, walker);
  }
#endif
  if (walker->best == INVALID_BEST) {
    save_all_values(solver, walker);
  } else {
    assert(SIZE_STACK(walker->trail) < INVALID_BEST);
    walker->best = SIZE_STACK(walker->trail);
    LOG("new best trail position %u", walker->best);
  }
}

static void local_search_step(kissat *solver, walker *walker) {
  assert(walker->current);
  INC(flipped);
  assert(walker->flipped < UINT64_MAX);
  walker->flipped++;
  LOG("starting local search flip %" PRIu64 " with %u unsatisfied clauses",
        GET(flipped), walker->current);
  unsigned lit = pick_literal(solver, walker);
  flip_literal(solver, walker, lit);
  push_flipped(solver, walker, lit);
  if (walker->current < walker->minimum) {
    update_best(solver, walker);
  }
  LOG("ending local search step %" PRIu64 " with %u unsatisfied clauses",
        GET(flipped), walker->current);
}

static void local_search_round(walker *walker) {
  kissat *solver = walker->solver;
#ifndef QUIET
  const unsigned before = walker->minimum;
#endif
  statistics *statistics = &solver->statistics;
  while (walker->minimum && walker->limit > statistics->walk_steps) {
    if (TERMINATED(walk_terminated_1)) {
      break;
    }
    local_search_step(solver, walker);
  }
#ifndef QUIET
  report_minimum("last", solver, walker);
  assert(statistics->walk_steps >= walker->start);
  const uint64_t steps = statistics->walk_steps - walker->start;
  // *INDENT-OFF*
  kissat_very_verbose (solver,
    "walking ends with %u unsatisfied clauses", walker->current);
  kissat_very_verbose (solver,
    "flipping %" PRIu64 " literals took %" PRIu64 " steps (%.2f per flipped)",
    walker->flipped, steps, kissat_average (steps, walker->flipped));
  // *INDENT-ON*
  const unsigned after = walker->minimum;
  kissat_phase(solver, "walk", GET(walks),
        "%s minimum %u after %" PRIu64 " flips",
        after < before ? "new" : "unchanged", after, walker->flipped);
#endif
}

static void save_final_minimum(walker *walker, bool first_time) {
  kissat *solver = walker->solver;

  assert(walker->minimum <= walker->initial);
  if (walker->minimum == walker->initial) {
    if (first_time)
      kissat_phase(solver, "walk", GET(walks),
            "saving first assignment falsifying %u clauses",
            walker->minimum);
    else if (!walker->minimum)
      kissat_phase(solver, "walk", GET(walks),
            "saving assignment satisfying all clauses");
    else {
      kissat_phase(solver, "walk", GET(walks),
            "no improvement thus keeping saved phases");
      return;
    }
  } else
    kissat_phase(solver, "walk", GET(walks),
          "saving improved assignment of %u unsatisfied clauses",
          walker->minimum);

  if (!walker->best || walker->best == INVALID_BEST) {
    LOG("minimum already saved");
  } else {
    save_walker_trail(solver, walker, false);
  }

  INC(walk_improved);

  if (GET_OPTION(walkreuse)) {
    kissat_insert_cache(solver, walker->minimum);
  }
}

#ifdef CHECK_WALK

static void check_walk(kissat *solver, unsigned expected) {
  unsigned unsatisfied = 0;
  watches *all_watches = solver->watches;
  for (all_literals(lit)) {
    assert(lit < LITS);
    watches *watches = all_watches + lit;
    if (kissat_empty_vector(watches)) {
      continue;
    }
    value value = solver->values[lit];
    if (!value) {
      value = solver->phases.saved[IDX(lit)];
      assert(value);
      if (NEGATED(lit)) {
        value = -value;
      }
    }
    if (value > 0) {
      continue;
    }
    for (all_binary_blocking_watches(watch, *watches))
      if (watch.type.binary) {
        if (watch.binary.redundant) {
          continue;
        }
        const unsigned other = watch.binary.lit;
        if (other < lit) {
          continue;
        }
        value = solver->values[other];
        if (!value) {
          value = solver->phases.saved[IDX(other)];
          assert(value);
          if (NEGATED(other)) {
            value = -value;
          }
        }
        if (value > 0) {
          continue;
        }
        unsatisfied++;
        LOGBINARY(lit, other, "unsat");
      }
  }
  for (all_clauses(c)) {
    if (c->redundant) {
      continue;
    }
    if (c->garbage) {
      continue;
    }
    bool satisfied = false;
    for (all_literals_in_clause(lit, c)) {
      value value = solver->values[lit];
      if (!value) {
        value = solver->phases.saved[IDX(lit)];
        assert(value);
        if (NEGATED(lit)) {
          value = -value;
        }
      }
      if (value > 0) {
        satisfied = true;
      }

    }
    if (satisfied) {
      continue;
    }
    LOGCLS(c, "unsatisfied");
    unsatisfied++;
  }
  LOG("expected %u unsatisfied", expected);
  LOG("actually %u unsatisfied", unsatisfied);
  assert(expected == unsatisfied);
}

#endif

static void walking_phase(kissat *solver, bool first_time,
      bool use_previous_phase) {
  INC(walks);
  litpairs irredundant;
  litwatches redundant;
  INIT_STACK(irredundant);
  INIT_STACK(redundant);
  kissat_enter_dense_mode(solver, &irredundant, &redundant);
  walker walker;
  init_walker(solver, &walker, &irredundant, use_previous_phase);
  init_walker_limit(solver, &walker);
  local_search_round(&walker);
  save_final_minimum(&walker, first_time);
#ifdef CHECK_WALK
  unsigned expected = walker.minimum;
#endif
  release_walker(&walker);
  kissat_resume_sparse_mode(solver, false, &irredundant, &redundant);
  RELEASE_STACK(irredundant);
  RELEASE_STACK(redundant);
#if CHECK_WALK
  check_walk(solver, expected);
#endif
}

bool kissat_walking(kissat *solver) {
  reference last_irredundant = solver->last_irredundant;
  if (last_irredundant == INVALID_REF) {
    last_irredundant = SIZE_STACK(solver->arena);
  }

  if (last_irredundant > MAX_WALK_REF) {
    kissat_extremely_verbose(solver, "can not walk since last "
          "irredundant clause reference %u too large",
          last_irredundant);
    return false;
  }

  if (IRREDUNDANT_CLAUSES > MAX_WALK_REF) {
    kissat_extremely_verbose(solver, "can not walk due to "
          "way too many irredundant clauses %" PRIu64,
          IRREDUNDANT_CLAUSES);
    return false;
  }

  return true;
}

static void walk(kissat *solver, bool first_time, bool use_previous_phase) {
  assert(!solver->level);
  assert(!solver->inconsistent);
  assert(kissat_propagated(solver));

  reference last_irredundant = solver->last_irredundant;
  if (last_irredundant == INVALID_REF) {
    last_irredundant = SIZE_STACK(solver->arena);
  }

  if (last_irredundant > MAX_WALK_REF) {
    kissat_phase(solver, "walk", GET(walks),
          "last irredundant clause reference %u too large",
          last_irredundant);
    return;
  }

  if (IRREDUNDANT_CLAUSES > MAX_WALK_REF) {
    kissat_phase(solver, "walk", GET(walks),
          "way too many irredundant clauses %" PRIu64,
          IRREDUNDANT_CLAUSES);
    return;
  }

  STOP_SEARCH_AND_START_SIMPLIFIER(walking);
  walking_phase(solver, first_time, use_previous_phase);
  STOP_SIMPLIFIER_AND_RESUME_SEARCH(walking);
}

void kissat_walk(kissat *solver) {
  assert(kissat_walking(solver));
  const char last = solver->rephased.last;
  assert(last);

  unsigned bit = 0;
#define REPHASE(NAME,TYPE,INDEX) \
  if (last == TYPE) \
    bit = INDEX; \
  else
  REPHASES
#undef REPHASE
  bit = 32;
  assert(bit < 32);
  const unsigned mask = (1u << bit);

  const uint64_t walks = GET(walks);
  const bool first_time = !walks;
  const int reuse = GET_OPTION(walkreuse);
  const bool decisions_used = ! !(solver->walked & mask);

  bool use_previous_phases;

  if (first_time || !reuse) {
    use_previous_phases = false;
  } else if (reuse > 1) {
    use_previous_phases = true;
  } else if (!decisions_used) {
    use_previous_phases = false;
  } else {
    use_previous_phases = walks & 1;
  }

  kissat_extremely_verbose(solver,
        "walking uses %s phases after last '%c' rephase",
        use_previous_phases ? "previous" : "decision",
        last);

  walk(solver, first_time, use_previous_phases);

  solver->walked |= mask;
}

int kissat_walk_initially(kissat *solver) {
  if (solver->inconsistent || !EMPTY_STACK(solver->failed)) {
    return 20;
  }
  if (TERMINATED(walk_terminated_2)) {
    return 0;
  }
  if (!GET_OPTION(walkinitially)) {
    return 0;
  }
  if (!kissat_walking(solver)) {
    return 0;
  }
  kissat_backtrack_without_updating_phases(solver, 0);
  assert(kissat_propagated(solver));
  walk(solver, true, false);
  REPORT(0, 'W');
  return 0;
}

#include "dense.h"
#include "inline.h"
#include "logging.h"
#include "kitten.h"
#include "print.h"
#include "propdense.h"
#include "report.h"
#include "sweep.h"
#include "terminate.h"

#include <inttypes.h>

typedef struct sweeper sweeper;

struct sweeper {
  kissat *solver;
  unsigned encoded;
  unsigned *depths;
  unsigned *reprs;
  unsigneds vars;
  references refs;
  unsigneds clause;
  unsigneds schedule;
  unsigneds backbone;
  unsigneds partition;
  unsigneds core;
  uint64_t limit;
  uint64_t solved;
};

static int sweep_solve(kissat *solver, sweeper *sweeper) {
  kitten *kitten = solver->kitten;
  if ((sweeper->solved++) % 3) {
    kitten_flip_phases(kitten);
  } else {
    kitten_randomize_phases(kitten);
  }
  INC(sweep_solved);
  int res = kitten_solve(kitten);
  if (res == 10) {
    INC(sweep_sat);
  }
  if (res == 20) {
    INC(sweep_unsat);
  }
  return res;
}

static void set_kitten_ticks_limit(kissat *solver, sweeper *sweeper) {
  uint64_t remaining = 0;
  if (solver->statistics.kitten_ticks < sweeper->limit) {
    remaining = sweeper->limit - solver->statistics.kitten_ticks;
  }
  LOG("'kitten_ticks' remaining %" PRIu64, remaining);
  kitten_set_ticks_limit(solver->kitten, remaining);
}

static bool kitten_ticks_limit_hit(kissat *solver, sweeper *sweeper,
      const char *when) {
  if (solver->statistics.kitten_ticks >= sweeper->limit) {
    LOG("'kitten_ticks' limit of %" PRIu64 " ticks hit after %"
          PRIu64 " ticks during %s",
          sweeper->limit, solver->statistics.kitten_ticks, when);
    return true;
  }
#ifndef LOGGING
  (void) when;
#endif
  return false;
}

static void init_sweeper(kissat *solver, sweeper *sweeper) {
  sweeper->solver = solver;
  sweeper->encoded = 0;
  sweeper->solved = 0;
  CALLOC(sweeper->depths, VARS);
  CALLOC(sweeper->reprs, LITS);
  for (all_literals(lit)) {
    sweeper->reprs[lit] = lit;
  }
  INIT_STACK(sweeper->vars);
  INIT_STACK(sweeper->refs);
  INIT_STACK(sweeper->clause);
  INIT_STACK(sweeper->schedule);
  INIT_STACK(sweeper->backbone);
  INIT_STACK(sweeper->partition);
  INIT_STACK(sweeper->core);
  assert(!solver->kitten);
  solver->kitten = kitten_embedded(solver);
  kitten_track_antecedents(solver->kitten);
  kissat_enter_dense_mode(solver, 0, 0);
  kissat_connect_irredundant_large_clauses(solver);
  SET_EFFORT_LIMIT(limit, sweep, kitten_ticks,
        kissat_linear(1 + solver->active));
  sweeper->limit = limit;
  set_kitten_ticks_limit(solver, sweeper);
}

static unsigned release_sweeper(kissat *solver, sweeper *sweeper) {
  unsigned merged = 0;
  for (all_variables(idx)) {
    if (!ACTIVE(idx)) {
      continue;
    }
    const unsigned lit = LIT(idx);
    if (sweeper->reprs[lit] != lit) {
      merged++;
    }
  }
  DEALLOC(sweeper->depths, VARS);
  DEALLOC(sweeper->reprs, LITS);
  RELEASE_STACK(sweeper->vars);
  RELEASE_STACK(sweeper->refs);
  RELEASE_STACK(sweeper->clause);
  RELEASE_STACK(sweeper->schedule);
  RELEASE_STACK(sweeper->backbone);
  RELEASE_STACK(sweeper->partition);
  RELEASE_STACK(sweeper->core);
  kitten_release(solver->kitten);
  solver->kitten = 0;
  kissat_resume_sparse_mode(solver, false, 0, 0);
  return merged;
}

static void clear_sweeper(kissat *solver, sweeper *sweeper) {
  LOG("clearing sweeping environment");
  kitten_clear(solver->kitten);
  kitten_track_antecedents(solver->kitten);
  for (all_stack(unsigned, idx, sweeper->vars)) {
    assert(sweeper->depths[idx]);
    sweeper->depths[idx] = 0;
  }
  CLEAR_STACK(sweeper->vars);
  for (all_stack(reference, ref, sweeper->refs)) {
    clause *c = kissat_dereference_clause(solver, ref);
    assert(c->sweeped);
    c->sweeped = false;
  }
  CLEAR_STACK(sweeper->refs);
  CLEAR_STACK(sweeper->backbone);
  CLEAR_STACK(sweeper->partition);
  sweeper->encoded = 0;
  set_kitten_ticks_limit(solver, sweeper);
}

static unsigned sweep_repr(kissat *solver, sweeper *sweeper, unsigned lit) {
  unsigned res;
  {
    unsigned prev = lit;
    while ((res = sweeper->reprs[prev]) != prev) {
      prev = res;
    }
  }
  if (res == lit) {
    return res;
  }
  LOG("sweeping repr[%s] = %s", LOGLIT(lit), LOGLIT(res));
  {
    const unsigned not_res = NOT(res);
    unsigned next, prev = lit;;
    while ((next = sweeper->reprs[prev]) != res) {
      const unsigned not_prev = NOT(prev);
      sweeper->reprs[not_prev] = not_res;
      sweeper->reprs[prev] = res;
      prev = next;
    }
    assert(sweeper->reprs[NOT(prev)] == not_res);
  }
#ifndef LOGGING
  (void) solver;
#endif
  return res;
}

static void add_literal_to_environment(kissat *solver, sweeper *sweeper,
      unsigned depth, unsigned lit) {
  for (;;) {
    const unsigned idx = IDX(lit);
    if (!sweeper->depths[idx]) {
      assert(depth < UINT_MAX);
      sweeper->depths[idx] = depth + 1;
      PUSH_STACK(sweeper->vars, idx);
      LOG("sweeping[%u] adding literal %s", depth, LOGLIT(lit));
    }
    const unsigned repr = sweep_repr(solver, sweeper, lit);
    if (repr == lit) {
      break;
    }
    const unsigned not_lit = NOT(lit);
    const unsigned not_repr = NOT(repr);
    LOG("adding equivalence %s = %s", LOGLIT(lit), LOGLIT(repr));
    kitten_binary(solver->kitten, not_lit, repr);
    kitten_binary(solver->kitten, lit, not_repr);
    lit = repr;
  }
}

static void sweep_binary(kissat *solver, sweeper *sweeper,
      unsigned depth, unsigned lit, unsigned other) {
  LOGBINARY(lit, other, "sweeping[%u]", depth);
  value *values = solver->values;
  assert(!values[lit]);
  const value other_value = values[other];
  if (other_value > 0) {
    LOGBINARY(lit, other, "skipping satisfied");
    return;
  }
  const unsigned *depths = sweeper->depths;
  const unsigned other_idx = IDX(other);
  const unsigned other_depth = depths[other_idx];
  if (other_depth < depth) {
    LOGBINARY(lit, other, "skipping depth %u copied", other_depth);
    return;
  }
  assert(!other_value);
  add_literal_to_environment(solver, sweeper, depth, lit);
  add_literal_to_environment(solver, sweeper, depth, other);
  kitten_binary(solver->kitten, lit, other);
  sweeper->encoded++;
}

static void sweep_reference(kissat *solver, sweeper *sweeper,
      unsigned depth, reference ref) {
  assert(EMPTY_STACK(sweeper->clause));
  clause *c = kissat_dereference_clause(solver, ref);
  if (c->sweeped) {
    return;
  }
  if (c->garbage) {
    return;
  }
  LOGCLS(c, "sweeping[%u]", depth);
  value *values = solver->values;
  for (all_literals_in_clause(lit, c)) {
    const value value = values[lit];
    if (value > 0) {
      kissat_mark_clause_as_garbage(solver, false, c);
      CLEAR_STACK(sweeper->clause);
      return;
    }
    if (value < 0) {
      continue;
    }
    PUSH_STACK(sweeper->clause, lit);
  }
  PUSH_STACK(sweeper->refs, ref);
  c->sweeped = true;
  assert(SIZE_STACK(sweeper->clause) > 1);
  for (all_stack(unsigned, repr, sweeper->clause)) {
    add_literal_to_environment(solver, sweeper, depth, repr);
  }
  kitten_clause(solver->kitten, SIZE_STACK(sweeper->clause),
        BEGIN_STACK(sweeper->clause));
  CLEAR_STACK(sweeper->clause);
  sweeper->encoded++;
}

static void add_core_clause(void *state, bool learned, size_t size,
      const unsigned *lits) {
  sweeper *sweeper = state;
  kissat *solver = sweeper->solver;
  if (solver->inconsistent) {
    return;
  }
  const value *const values = solver->values;
  unsigned unit = INVALID_LIT;
  bool satisfied = false;
  unsigned non_false = 0;
  size_t saved = SIZE_STACK(sweeper->core);
  const unsigned *end = lits + size;
  for (const unsigned *p = lits; p != end; p++) {
    const unsigned lit = *p;
    const value value = values[lit];
    if (value > 0) {
      LOGLITS(size, lits, "extracted %s satisfied lemma", LOGLIT(lit));
      satisfied = true;
      break;
    }
    PUSH_STACK(sweeper->core, lit);
    if (value < 0) {
      continue;
    }
    non_false++;
    unit = lit;
  }

  bool reset = true;

  if (satisfied)
    ;
  else if (!non_false) {
    if (!solver->inconsistent) {
      LOG("sweeping produced empty clause");
      CHECK_AND_ADD_EMPTY();
      ADD_EMPTY_TO_PROOF();
      solver->inconsistent = true;
    }
  } else if (non_false == 1) {
    assert(unit != INVALID_LIT);
    LOGLITS(size, lits, "unit %s forcing %s", LOGLIT(unit),
          (learned ? "extracted lemma" : "original"));
    CHECK_AND_ADD_UNIT(unit);
    ADD_UNIT_TO_PROOF(unit);
    kissat_assign_unit(solver, unit, "sweeping backbone reason");
    INC(sweep_units);
  } else if (learned) {
    assert(non_false > 1);
#ifdef LOGGING
    if (non_false == size) {
      LOGLITS(size, lits, "extracted lemma");
    } else {
      LOGLITS(size, lits, "actual size %u extracted lemma", non_false);
    }
#endif
    CHECK_AND_ADD_LITS(size, lits);
    ADD_LITS_TO_PROOF(size, lits);
    PUSH_STACK(sweeper->core, INVALID_LIT);
    reset = false;
  } else {
    LOGLITS(size, lits, "skipping original");
  }

  if (reset) {
    RESIZE_STACK(sweeper->core, saved);
  }
}

static void add_core(kissat *solver, sweeper *sweeper) {
  LOG("adding sub-solver core clauses");
  assert(EMPTY_STACK(sweeper->core));
  kitten_compute_clausal_core(solver->kitten, 0);
  kitten_traverse_core_clauses(solver->kitten, sweeper, add_core_clause);
}

static void delete_core(kissat *solver, sweeper *sweeper) {
#ifdef CHECKING_OR_PROVING
  LOG("deleting sub-solver core clauses");
  const unsigned *const end = END_STACK(sweeper->core);
  const unsigned *c = BEGIN_STACK(sweeper->core);
  for (const unsigned *p = c; c != end; c = ++p) {
    while (*p != INVALID_LIT) {
      p++;
    }
    const size_t size = p - c;
    REMOVE_CHECKER_LITS(size, c);
    DELETE_LITS_FROM_PROOF(size, c);
  }
#else
  (void) solver;
#endif
  CLEAR_STACK(sweeper->core);
}

static void init_backbone_and_partition(kissat *solver, sweeper *sweeper) {
  LOG("initializing backbone and equivalent literals candidates");
  for (all_stack(unsigned, idx, sweeper->vars)) {
    if (!ACTIVE(idx)) {
      continue;
    }
    const unsigned lit = LIT(idx);
    const unsigned not_lit = NOT(lit);
    const signed char tmp = kitten_value(solver->kitten, lit);
    const unsigned candidate = (tmp < 0) ? not_lit : lit;
    LOG("sweeping candidate %s", LOGLIT(candidate));
    PUSH_STACK(sweeper->backbone, candidate);
    PUSH_STACK(sweeper->partition, candidate);
  }
  LOG("initialized %zu literals", SIZE_STACK(sweeper->backbone));
  PUSH_STACK(sweeper->partition, INVALID_LIT);
}

static void sweep_empty_clause(kissat *solver, sweeper *sweeper) {
  assert(!solver->inconsistent);
  add_core(solver, sweeper);
  assert(solver->inconsistent);
}

static void sweep_refine_partition(kissat *solver, sweeper *sweeper) {
  LOG("refining partition");
  kitten *kitten = solver->kitten;
  unsigneds old_partition = sweeper->partition;
  unsigneds new_partition;
  INIT_STACK(new_partition);
  const value *const values = solver->values;
  const unsigned *const old_begin = BEGIN_STACK(old_partition);
  const unsigned *const old_end = END_STACK(old_partition);
#ifdef LOGGING
  unsigned old_classes = 0;
  unsigned new_classes = 0;
#endif
  for (const unsigned *p = old_begin, *q; p != old_end; p = q + 1) {
    unsigned assigned_true = 0, other;
    for (q = p; (other = *q) != INVALID_LIT; q++) {
      if (sweep_repr(solver, sweeper, other) != other) {
        continue;
      }
      if (values[other]) {
        continue;
      }
      signed char value = kitten_value(kitten, other);
      if (!value) {
        LOG("dropping sub-solver unassigned %s", LOGLIT(other));
      } else if (value > 0) {
        PUSH_STACK(new_partition, other);
        assigned_true++;
      }
    }
#ifdef LOGGING
    LOG("refining class %u of size %zu", old_classes, (size_t)(q - p));
    old_classes++;
#endif
    if (assigned_true == 0) {
      LOG("no positive literal in class");
    } else if (assigned_true == 1) {
#ifdef LOGGING
      other =
#else
      (void)
#endif
            POP_STACK(new_partition);
      LOG("dropping singleton class %s", LOGLIT(other));
    } else {
      LOG("%u positive literal in class", assigned_true);
      PUSH_STACK(new_partition, INVALID_LIT);
#ifdef LOGGING
      new_classes++;
#endif
    }

    unsigned assigned_false = 0;
    for (q = p; (other = *q) != INVALID_LIT; q++) {
      if (sweep_repr(solver, sweeper, other) != other) {
        continue;
      }
      if (values[other]) {
        continue;
      }
      signed char value = kitten_value(kitten, other);
      if (value < 0) {
        PUSH_STACK(new_partition, other);
        assigned_false++;
      }
    }

    if (assigned_false == 0) {
      LOG("no negative literal in class");
    } else if (assigned_false == 1) {
#ifdef LOGGING
      other =
#else
      (void)
#endif
            POP_STACK(new_partition);
      LOG("dropping singleton class %s", LOGLIT(other));
    } else {
      LOG("%u negative literal in class", assigned_false);
      PUSH_STACK(new_partition, INVALID_LIT);
#ifdef LOGGING
      new_classes++;
#endif
    }
  }
  RELEASE_STACK(old_partition);
  sweeper->partition = new_partition;
  LOG("refined %u classes into %u", old_classes, new_classes);
}

static void sweep_refine_backbone(kissat *solver, sweeper *sweeper) {
  LOG("refining backbone");
  const unsigned *const end = END_STACK(sweeper->backbone);
  unsigned *q = BEGIN_STACK(sweeper->backbone);
  const value *const values = solver->values;
  kitten *kitten = solver->kitten;
#ifdef LOGGING
  size_t old_size = SIZE_STACK(sweeper->backbone);
#endif
  for (const unsigned *p = q; p != end; p++) {
    const unsigned lit = *p;
    if (values[lit]) {
      continue;
    }
    signed char value = kitten_value(kitten, lit);
    if (!value) {
      LOG("dropping sub-solver unassigned %s", LOGLIT(lit));
    } else if (value >= 0) {
      *q++ = lit;
    }
  }
  SET_END_OF_STACK(sweeper->backbone, q);
#ifdef LOGGING
  size_t new_size = SIZE_STACK(sweeper->backbone);
  LOG("refined %zu backbone candidates into %zu", old_size, new_size);
#endif
}

static void sweep_refine(kissat *solver, sweeper *sweeper) {
  if (EMPTY_STACK(sweeper->backbone)) {
    LOG("no need to refine empty backbone candidates");
  } else {
    sweep_refine_backbone(solver, sweeper);
  }
  if (EMPTY_STACK(sweeper->partition)) {
    LOG("no need to refine empty partition candidates");
  } else {
    sweep_refine_partition(solver, sweeper);
  }
}

static void sweep_backbone_candidate(kissat *solver, sweeper *sweeper,
      unsigned lit) {
  LOG("trying backbone candidate %s", LOGLIT(lit));
  const unsigned not_lit = NOT(lit);
  kitten *kitten = solver->kitten;
  kitten_assume(kitten, not_lit);
  int res = sweep_solve(solver, sweeper);
  if (res == 10) {
    LOG("sweeping backbone candidate %s failed", LOGLIT(lit));
    sweep_refine(solver, sweeper);
  } else if (res == 20) {
    LOG("sweep unit %s", LOGLIT(lit));
    add_core(solver, sweeper);
    delete_core(solver, sweeper);
  }
}

static void add_binary(kissat *solver, unsigned lit, unsigned other) {
  kissat_new_binary_clause(solver, true, lit, other);
}

static unsigned sweep_equivalence_candidates(kissat *solver, sweeper *sweeper,
      unsigned lit, unsigned other) {
  LOG("trying equivalence candidates %s = %s", LOGLIT(lit), LOGLIT(other));
  const unsigned not_other = NOT(other);
  const unsigned not_lit = NOT(lit);
  kitten *kitten = solver->kitten;
  kitten_assume(kitten, not_lit);
  kitten_assume(kitten, other);
  int res = sweep_solve(solver, sweeper);
  if (res == 10) {
    LOG("first sweeping implication %s -> %s failed",
          LOGLIT(lit), LOGLIT(other));
    sweep_refine(solver, sweeper);
  } else if (!res) {
    LOG("first sweeping implication %s -> %s hit ticks limit",
          LOGLIT(lit), LOGLIT(other));
  }

  if (res != 20) {
    return INVALID_LIT;
  }

  LOG("first sweeping implication %s -> %s succeeded",
        LOGLIT(lit), LOGLIT(other));
  kitten_assume(kitten, lit);
  kitten_assume(kitten, not_other);
  res = sweep_solve(solver, sweeper);
  if (res == 10) {
    LOG("second sweeping implication %s <- %s failed",
          LOGLIT(lit), LOGLIT(other));
    sweep_refine(solver, sweeper);
  } else if (!res) {
    LOG("second sweeping implication %s <- %s hit ticks limit",
          LOGLIT(lit), LOGLIT(other));
  }

  if (res != 20) {
    return INVALID_LIT;
  }

  LOG("second sweeping implication %s <- %s succeeded too",
        LOGLIT(lit), LOGLIT(other));

  add_core(solver, sweeper);
  LOG("need to resolve first implication for proof again");
  kitten_assume(kitten, not_lit);
  kitten_assume(kitten, other);

  res = sweep_solve(solver, sweeper);

  if (!res) {
    LOG("resolving first implication hits ticks limit");
    delete_core(solver, sweeper);
    return INVALID_LIT;
  }

  assert(res == 20);

  LOG("sweep equivalence %s = %s", LOGLIT(lit), LOGLIT(other));
  INC(sweep_equivalences);
  add_binary(solver, not_lit, other);
  delete_core(solver, sweeper);
  add_core(solver, sweeper);
  add_binary(solver, lit, not_other);
  delete_core(solver, sweeper);
  assert(lit != other);
  assert(lit != not_other);
  LOG("adding equivalences to sub-solver");
  kitten_binary(kitten, not_lit, other);
  kitten_binary(kitten, lit, not_other);
  if (lit < other) {
    sweeper->reprs[other] = lit;
    sweeper->reprs[not_other] = not_lit;
    return lit;
  } else {
    sweeper->reprs[lit] = other;
    sweeper->reprs[not_lit] = not_other;
    return other;
  }
}

static void sweep_variable(kissat *solver, sweeper *sweeper, unsigned idx) {
  assert(!solver->inconsistent);
  if (!ACTIVE(idx)) {
    return;
  }
  const unsigned start = LIT(idx);
  if (sweeper->reprs[start] != start) {
    return;
  }
  assert(EMPTY_STACK(sweeper->vars));
  assert(EMPTY_STACK(sweeper->refs));
  assert(EMPTY_STACK(sweeper->backbone));
  assert(EMPTY_STACK(sweeper->partition));
  assert(!sweeper->encoded);
  INC(sweep_variables);
  LOG("sweeping %s", LOGVAR(idx));
  assert(!VALUE(start));
  LOG("starting sweeping[0]");
  add_literal_to_environment(solver, sweeper, 0, start);
  LOG("finished sweeping[0]");
  LOG("starting sweeping[1]");
  bool variable_limit_reached = false;
  size_t expand = 0, next = 1;
  uint64_t depth_limit = solver->statistics.sweep_completed;
  depth_limit += GET_OPTION(sweepdepth);
  const unsigned max_depth = GET_OPTION(sweepmaxdepth);
  if (depth_limit > max_depth) {
    depth_limit = max_depth;
  }
  unsigned depth = 1;
  while (!variable_limit_reached) {
    if (sweeper->encoded == (unsigned) GET_OPTION(sweepclauses)) {
      LOG("environment clause limit reached");
      break;
    }
    if (expand == next) {
      LOG("finished sweeping[%u]", depth);
      if (depth == (unsigned) GET_OPTION(sweepdepth)) {
        LOG("environment depth limit reached");
        break;
      }
      next = SIZE_STACK(sweeper->vars);
      if (expand == next) {
        LOG("completely copied all clauses");
        break;
      }
      depth++;
      LOG("starting sweeping[%u]", depth);
    }
    const unsigned idx = PEEK_STACK(sweeper->vars, expand);
    for (unsigned sign = 0; sign < 2; sign++) {
      const unsigned lit = LIT(idx) + sign;
      watches *watches = &WATCHES(lit);
      for (all_binary_large_watches(watch, *watches)) {
        if (watch.type.binary) {
          const unsigned other = watch.binary.lit;
          sweep_binary(solver, sweeper, depth, lit, other);
        } else {
          reference ref = watch.large.ref;
          sweep_reference(solver, sweeper, depth, ref);
        }
        if (SIZE_STACK(sweeper->vars) >=
              (unsigned) GET_OPTION(sweepvars)) {
          LOG("environment variable limit reached");
          variable_limit_reached = true;
          break;
        }
        if (variable_limit_reached) {
          break;
        }
      }
    }
    expand++;
  }
#if 0
  LOG("sweeper environment has %zu variables", SIZE_STACK(sweeper->vars));
  LOG("sweeper environment has %u clauses", sweeper->encoded);
  LOG("sweeper environment depth of %u generations", depth);
#elif !defined(QUIET)
  {
    int elit = kissat_export_literal(solver, LIT(idx));
    kissat_extremely_verbose(solver,
          "variable %d environment of "
          "%zu variables %u clauses depth %u",
          elit, SIZE_STACK(sweeper->vars),
          sweeper->encoded, depth);
  }
#endif
  int res = sweep_solve(solver, sweeper);
  LOG("sub-solver returns '%d'", res);
  if (res == 10) {
    init_backbone_and_partition(solver, sweeper);
#ifndef QUIET
    uint64_t units = solver->statistics.sweep_units;
    uint64_t solved = solver->statistics.sweep_solved;
#endif
    bool done = false;
    while (!EMPTY_STACK(sweeper->backbone)) {
      if (solver->inconsistent ||
            TERMINATED(sweep_terminated_1) ||
            kitten_ticks_limit_hit(solver, sweeper, "backbone refinement")) {
        done = true;
        break;
      }
      const unsigned lit = POP_STACK(sweeper->backbone);
      if (!ACTIVE(IDX(lit))) {
        continue;
      }
      sweep_backbone_candidate(solver, sweeper, lit);
    }
#ifndef QUIET
    units = solver->statistics.sweep_units - units;
    solved = solver->statistics.sweep_solved - solved;
    // if (units)
    {
      int elit = kissat_export_literal(solver, LIT(idx));
      kissat_extremely_verbose(solver,
            "%scomplete variable %d backbone with %"
            PRIu64 " units in %" PRIu64 " solver calls",
            done ? "in" : "", elit, units, solved);
    }
#endif
    if (!done) {
      assert(EMPTY_STACK(sweeper->backbone));
#ifndef QUIET
      uint64_t equivalences = solver->statistics.sweep_equivalences;
#endif
      while (!EMPTY_STACK(sweeper->partition)) {
        if (solver->inconsistent ||
              TERMINATED(sweep_terminated_2) ||
              kitten_ticks_limit_hit(solver, sweeper,
                    "partition refinement")) {
          done = true;
          break;
        }
        if (SIZE_STACK(sweeper->partition) > 2) {
          const unsigned *end = END_STACK(sweeper->partition);
          assert(end[-1] == INVALID_LIT);
          unsigned lit = end[-2];
          unsigned other = end[-3];
          unsigned repr =
                sweep_equivalence_candidates(solver, sweeper, lit,
                      other);
          if (repr != INVALID_LIT) {
            unsigned tmp = POP_STACK(sweeper->partition);
            assert(tmp == INVALID_LIT);
            tmp = POP_STACK(sweeper->partition);
            assert(tmp == lit);
            tmp = POP_STACK(sweeper->partition);
            assert(tmp == other);
#ifdef NDEBUG
            (void) tmp;
#endif
            if (!EMPTY_STACK(sweeper->partition) &&
                  TOP_STACK(sweeper->partition) != INVALID_LIT) {
              LOG("merged %s and %s into %s",
                    LOGLIT(lit), LOGLIT(other), LOGLIT(repr));
              PUSH_STACK(sweeper->partition, repr);
              PUSH_STACK(sweeper->partition, INVALID_LIT);
            } else
              LOG("squashed binary candidate class with %s %s",
                    LOGLIT(lit), LOGLIT(other));
          }
        } else {
          CLEAR_STACK(sweeper->partition);
        }
      }
#ifndef QUIET
      equivalences = solver->statistics.sweep_equivalences - equivalences;
      solved = solver->statistics.sweep_solved - solved;
      // if (equivalences)
      {
        int elit = kissat_export_literal(solver, LIT(idx));
        kissat_extremely_verbose(solver,
              "%scomplete variable %d partition with %"
              PRIu64 " equivalences in %" PRIu64
              " solver calls", done ? "in" : "", elit,
              equivalences, solved);
      }
#endif
    }
  } else if (res == 20) {
    sweep_empty_clause(solver, sweeper);
  }

  if (!solver->inconsistent && !kissat_propagated(solver)) {
    kissat_dense_propagate(solver);
  }

  clear_sweeper(solver, sweeper);
}

static unsigned schedule_sweeping(kissat *solver, sweeper *sweeper) {
  assert(EMPTY_STACK(sweeper->schedule));
  const size_t max_occurrences = GET_OPTION(sweepclauses);
#ifndef QUIET
  unsigned rescheduled = 0;
#endif
  for (unsigned reschedule = 0; reschedule < 2; reschedule++) {
    for (unsigned idx = VARS; idx--;) {
      if (!ACTIVE(idx)) {
        continue;
      }
      const unsigned sweep = FLAGS(idx)->sweep;
      if (reschedule != sweep) {
        continue;
      }
      const unsigned lit = LIT(idx);
      const unsigned not_lit = NOT(lit);
      const size_t pos = SIZE_WATCHES(WATCHES(lit));
      const size_t neg = SIZE_WATCHES(WATCHES(not_lit));
      if (pos + neg > max_occurrences) {
        continue;
      }
      LOG("scheduling %s with %zu + %zu occurrences",
            LOGVAR(idx), pos, neg);
      PUSH_STACK(sweeper->schedule, idx);
#ifndef QUIET
      if (reschedule) {
        rescheduled++;
      }
#endif
    }
  }
  const unsigned scheduled = SIZE_STACK(sweeper->schedule);
#ifndef QUIET
  kissat_phase(solver, "sweep", GET(sweep),
        "scheduled %u variables %.0f%% (%u rescheduled %.0f%%)",
        scheduled, kissat_percent(scheduled, solver->active),
        rescheduled, kissat_percent(rescheduled, scheduled));
#endif
  return scheduled;
}

static void unschedule_sweeping(kissat *solver, sweeper *sweeper,
      unsigned scheduled) {
  if (EMPTY_STACK(sweeper->schedule)) {
    INC(sweep_completed);
    kissat_phase(solver, "sweep", GET(sweep),
          "all scheduled variables swept");
    return;
  }
  unsigned rescheduled = 0;
  unsigned notrescheduled = 0;
  for (all_stack(unsigned, idx, sweeper->schedule)) {
    if (!ACTIVE(idx)) {
      continue;
    }
    struct flags *f = FLAGS(idx);
    const bool sweep = f->sweep;
    if (sweep) {
      rescheduled++;
    } else {
      notrescheduled++;
    }
  }
  if (!rescheduled && !notrescheduled) {
    kissat_phase(solver, "sweep", GET(sweep),
          "actually all scheduled variables swept");
    return;
  }
  if (!rescheduled) {
    INC(sweep_completed);
    for (all_stack(unsigned, idx, sweeper->schedule))
      if (ACTIVE(idx)) {
        FLAGS(idx)->sweep = true;
      }
  }
#ifndef QUIET
  const unsigned total = rescheduled + notrescheduled;
  kissat_phase(solver, "sweep", GET(sweep),
        "%u variables remain %.0f%% (%u rescheduled %.0f%%)",
        total, kissat_percent(total, scheduled),
        rescheduled, kissat_percent(rescheduled, total));
#else
  (void) scheduled;
#endif
}

void kissat_sweep(kissat *solver) {
  if (!GET_OPTION(sweep)) {
    return;
  }
  if (solver->inconsistent) {
    return;
  }
  assert(!solver->level);
  assert(!solver->unflushed);
  START(sweep);
  INC(sweep);
  statistics *statistics = &solver->statistics;
  uint64_t equivalences = statistics->sweep_equivalences;
  uint64_t units = statistics->sweep_units;
  sweeper sweeper;
  init_sweeper(solver, &sweeper);
  const unsigned scheduled = schedule_sweeping(solver, &sweeper);
#ifndef QUIET
  unsigned swept = 0;
#endif
  while (!EMPTY_STACK(sweeper.schedule)) {
    if (solver->inconsistent) {
      break;
    }
    if (TERMINATED(sweep_terminated_3)) {
      break;
    }
    if (solver->statistics.kitten_ticks > sweeper.limit) {
      break;
    }
    const unsigned idx = POP_STACK(sweeper.schedule);
    FLAGS(idx)->sweep = false;
    sweep_variable(solver, &sweeper, idx);
#ifndef QUIET
    int elit = kissat_export_literal(solver, LIT(idx));
    kissat_extremely_verbose(solver,
          "swept[%u] external variable %d",
          ++swept, elit);
#endif
  }
  unschedule_sweeping(solver, &sweeper, scheduled);
  unsigned inactive = release_sweeper(solver, &sweeper);
  equivalences = statistics->sweep_equivalences - equivalences;
  units = solver->statistics.sweep_units - units;
  kissat_phase(solver, "sweep", GET(sweep),
        "found %" PRIu64 " equivalences and %" PRIu64
        " units sweeping %u variables %.0f%%",
        equivalences, units, swept,
        kissat_percent(swept, scheduled));
#ifndef QUIET
  uint64_t eliminated = equivalences + units;
  assert(solver->active >= inactive);
  solver->active -= inactive;
  REPORT(!eliminated, '=');
  solver->active += inactive;
#else
  (void) inactive;
#endif
  STOP(sweep);
}

#include "allocate.h"
#include "backtrack.h"
#include "backward.h"
#include "collect.h"
#include "dense.h"
#include "eliminate.h"
#include "forward.h"
#include "inline.h"
#include "inlinescore.h"
#include "kitten.h"
#include "print.h"
#include "propdense.h"
#include "protect.h"
#include "report.h"
#include "resolve.h"
#include "terminate.h"
#include "trail.h"
#include "weaken.h"

#include <inttypes.h>
#include <math.h>

static uint64_t eliminate_adjustment(kissat *solver) {
  return 2 * CLAUSES + kissat_nlogn(1 + solver->active);
}

static bool really_eliminate(kissat *solver) {
  if (!GET_OPTION(really)) {
    return true;
  }
  const uint64_t limit = eliminate_adjustment(solver);
  statistics *statistics = &solver->statistics;
  return limit < statistics->search_ticks;
}

bool kissat_eliminating(kissat *solver) {
  if (!solver->enabled.eliminate) {
    return false;
  }
  statistics *statistics = &solver->statistics;
  if (!statistics->clauses_irredundant) {
    return false;
  }
  if (solver->waiting.eliminate.reduce > statistics->reductions) {
    return false;
  }
  limits *limits = &solver->limits;
  if (limits->eliminate.conflicts > statistics->conflicts) {
    return false;
  }
  if (!really_eliminate(solver)) {
    return false;
  }
  if (limits->eliminate.variables.added < statistics->variables_added) {
    return true;
  }
  if (limits->eliminate.variables.removed < statistics->variables_removed) {
    return true;
  }
  return false;
}

static inline void update_after_adding_variable(kissat *solver, heap *schedule,
      unsigned idx) {
  if (!GET_OPTION(eliminateheap)) {
    return;
  }
  assert(schedule->size);
  kissat_update_variable_score(solver, schedule, idx);
}

static inline void update_after_adding_stack(kissat *solver, unsigneds *stack) {
  if (!GET_OPTION(eliminateheap)) {
    return;
  }
  assert(!solver->probing);
  heap *schedule = &solver->schedule;
  if (!schedule->size) {
    return;
  }
  for (all_stack(unsigned, lit, *stack)) {
    update_after_adding_variable(solver, schedule, IDX(lit));
  }
}

static inline void update_after_removing_variable(kissat *solver,
      unsigned idx) {
  if (!GET_OPTION(eliminateheap)) {
    return;
  }
  assert(!solver->probing);
  flags *f = solver->flags + idx;
  if (f->fixed) {
    return;
  }
  assert(!f->eliminated);
  heap *schedule = &solver->schedule;
  if (!schedule->size) {
    return;
  }
  kissat_update_variable_score(solver, schedule, idx);
  if (!kissat_heap_contains(schedule, idx)) {
    kissat_push_heap(solver, schedule, idx);
  }
}

void kissat_update_after_removing_variable(kissat *solver, unsigned idx) {
  update_after_removing_variable(solver, idx);
}

static inline void update_after_removing_clause(kissat *solver, clause *c,
      unsigned except) {
  if (!GET_OPTION(eliminateheap)) {
    return;
  }
  assert(c->garbage);
  for (all_literals_in_clause(lit, c))
    if (lit != except) {
      update_after_removing_variable(solver, IDX(lit));
    }
}

void kissat_update_after_removing_clause(kissat *solver,
      clause *c, unsigned except) {
  update_after_removing_clause(solver, c, except);
}

void kissat_eliminate_binary(kissat *solver, bool weakened, unsigned lit,
      unsigned other) {
  kissat_disconnect_binary(solver, other, lit);
  kissat_delete_binary(solver, false, false, weakened, lit, other);
  update_after_removing_variable(solver, IDX(other));
}

void kissat_eliminate_clause(kissat *solver, bool weakened, clause *c,
      unsigned lit) {
  kissat_mark_clause_as_garbage(solver, weakened, c);
  update_after_removing_clause(solver, c, lit);
}

static unsigned schedule_variables(kissat *solver) {
  const bool eliminateheap = GET_OPTION(eliminateheap);

  LOG("initializing variable schedule");
  assert(!solver->schedule.size);

  if (eliminateheap) {
    kissat_resize_heap(solver, &solver->schedule, solver->vars);
  }

  flags *all_flags = solver->flags;

  size_t scheduled = 0;

  for (all_variables(idx)) {
    flags *flags = all_flags + idx;
    if (!flags->active) {
      continue;
    }
    if (!flags->eliminate) {
      continue;
    }
    LOG("scheduling %s", LOGVAR(idx));
    scheduled++;
    if (eliminateheap) {
      update_after_removing_variable(solver, idx);
    }
  }
#ifndef NDEBUG
  if (eliminateheap) {
    assert(scheduled == kissat_size_heap(&solver->schedule));
  }
#endif
#ifndef QUIET
  size_t active = solver->active;
  kissat_phase(solver, "eliminate", GET(eliminations),
        "scheduled %zu variables %.0f%%",
        scheduled, kissat_percent(scheduled, active));
#endif
  return scheduled;
}

void kissat_flush_units_while_connected(kissat *solver) {
  const unsigned *propagate = solver->propagate;
  const unsigned *end_trail = END_ARRAY(solver->trail);
  assert(propagate <= end_trail);
  const size_t units = end_trail - propagate;
  if (!units) {
    return;
  }
#ifdef LOGGING
  LOG("propagating and flushing %zu units", units);
#endif
  if (!kissat_dense_propagate(solver)) {
    return;
  }
  LOG("marking and flushing unit satisfied clauses");
  const value *const values = solver->values;

  end_trail = END_ARRAY(solver->trail);
  while (propagate != end_trail) {
    const unsigned unit = *propagate++;
    assert(values[unit] > 0);
    watches *unit_watches = &WATCHES(unit);
    watch *begin = BEGIN_WATCHES(*unit_watches), *q = begin;
    const watch *const end = END_WATCHES(*unit_watches), *p = q;
    if (begin == end) {
      continue;
    }
    LOG("marking %s satisfied clauses as garbage", LOGLIT(unit));
    while (p != end) {
      const watch watch = *q++ = *p++;
      if (watch.type.binary) {
        const unsigned other = watch.binary.lit;
        if (!values[other]) {
          update_after_removing_variable(solver, IDX(other));
        }
      } else {
        const reference ref = watch.large.ref;
        clause *c = kissat_dereference_clause(solver, ref);
        if (!c->garbage) {
          kissat_eliminate_clause(solver, false, c, unit);
        }
        assert(c->garbage);
        q--;
      }
    }
    assert(q <= end);
    size_t flushed = end - q;
    if (!flushed) {
      continue;
    }
    LOG("flushing %zu references satisfied by %s", flushed, LOGLIT(unit));
    SET_END_OF_WATCHES(*unit_watches, q);
  }
}

static void connect_resolvents(kissat *solver) {
  bool backward = GET_OPTION(backward);
  const value *const values = solver->values;
  assert(EMPTY_STACK(solver->clause));
  uint64_t added = 0;
  bool satisfied = false;
  for (all_stack(unsigned, other, solver->resolvents)) {
    if (other == INVALID_LIT) {
      if (satisfied) {
        satisfied = false;
      } else if (kissat_forward_subsume_temporary(solver)) {
        LOGTMP("temporary forward subsumed");
      } else {
        size_t size = SIZE_STACK(solver->clause);
        if (!size) {
          assert(!solver->inconsistent);
          LOG("resolved empty clause");
          CHECK_AND_ADD_EMPTY();
          ADD_EMPTY_TO_PROOF();
          solver->inconsistent = true;
          break;
        } else if (size == 1) {
          const unsigned unit = PEEK_STACK(solver->clause, 0);
          LOG("resolved unit clause %s", LOGLIT(unit));
          kissat_learned_unit(solver, unit);
        } else {
          assert(size > 1);
          reference ref = kissat_new_irredundant_clause(solver);
          update_after_adding_stack(solver, &solver->clause);
          added++;

          if (backward)
            backward =
                  kissat_backward_subsume_temporary(solver, ref);
        }
      }
      CLEAR_STACK(solver->clause);
    } else if (!satisfied) {
      const value value = values[other];
      if (value > 0) {
        LOGTMP("now %s satisfied resolvent", LOGLIT(other));
        satisfied = true;
      } else if (value < 0) {
        LOG2("dropping now falsified literal %s", LOGLIT(other));
      } else {
        PUSH_STACK(solver->clause, other);
      }
    }
  }
  LOG("added %" PRIu64 " new clauses", added);
  CLEAR_STACK(solver->resolvents);
}

static void weaken_clauses(kissat *solver, unsigned lit) {
  const unsigned not_lit = NOT(lit);

  const value *const values = solver->values;
  assert(!values[lit]);

  watches *pos_watches = &WATCHES(lit);

  for (all_binary_large_watches(watch, *pos_watches)) {
    if (watch.type.binary) {
      const unsigned other = watch.binary.lit;
      const value value = values[other];
      const bool weaken = value <= 0;
      if (weaken) {
        kissat_weaken_binary(solver, lit, other);
      }
      assert(!watch.binary.redundant);
      kissat_eliminate_binary(solver, weaken, lit, other);
    } else {
      const reference ref = watch.large.ref;
      clause *c = kissat_dereference_clause(solver, ref);
      if (c->garbage) {
        continue;
      }
      bool satisfied = false;
      for (all_literals_in_clause(other, c)) {
        const value value = values[other];
        if (value <= 0) {
          continue;
        }
        satisfied = true;
        break;
      }
      if (!satisfied) {
        kissat_weaken_clause(solver, lit, c);
      }
      LOGCLS(c, "removing %s", LOGLIT(lit));
      kissat_eliminate_clause(solver, !satisfied, c, lit);
    }
  }
  RELEASE_WATCHES(*pos_watches);

  watches *neg_watches = &WATCHES(not_lit);

  bool optimize = !GET_OPTION(incremental);
  for (all_binary_large_watches(watch, *neg_watches)) {
    if (watch.type.binary) {
      const unsigned other = watch.binary.lit;
      assert(!watch.binary.redundant);
      const value value = values[other];
      const bool weaken = !optimize && value <= 0;
      if (weaken) {
        kissat_weaken_binary(solver, not_lit, other);
      }
      kissat_eliminate_binary(solver, weaken, not_lit, other);
    } else {
      const reference ref = watch.large.ref;
      clause *d = kissat_dereference_clause(solver, ref);
      if (d->garbage) {
        continue;
      }
      bool satisfied = false;
      for (all_literals_in_clause(other, d)) {
        const value value = values[other];
        if (value <= 0) {
          continue;
        }
        satisfied = true;
        break;
      }
      const bool weaken = !optimize && !satisfied;
      if (weaken) {
        kissat_weaken_clause(solver, not_lit, d);
      }
      LOGCLS(d, "removing %s", LOGLIT(not_lit));
      kissat_eliminate_clause(solver, weaken, d, not_lit);
    }
  }
  if (optimize && !EMPTY_WATCHES(*neg_watches)) {
    kissat_weaken_unit(solver, not_lit);
  }
  RELEASE_WATCHES(*neg_watches);

  kissat_flush_units_while_connected(solver);
}

static void eliminate_equivalence(kissat *solver,
      unsigned idx, unsigned replacament) {
  const unsigned lit = LIT(idx);
  const unsigned not_lit = NOT(lit);
  const unsigned not_replacament = NOT(replacament);

  watches *pos_watches = &WATCHES(lit);

  for (all_binary_large_watches(watch, *pos_watches)) {
    if (watch.type.binary) {
      const unsigned other = watch.binary.lit;
      assert(!watch.binary.redundant);
      const bool weakened =
            other == not_replacament && GET_OPTION(incremental);
      kissat_eliminate_binary(solver, weakened, lit, other);
    } else {
      const reference ref = watch.large.ref;
      clause *c = kissat_dereference_clause(solver, ref);
      if (c->garbage) {
        continue;
      }
      LOGCLS(c, "removing %s", LOGLIT(lit));
      kissat_eliminate_clause(solver, false, c, lit);
    }
  }
  RELEASE_WATCHES(*pos_watches);

  watches *neg_watches = &WATCHES(not_lit);

  for (all_binary_large_watches(watch, *neg_watches)) {
    if (watch.type.binary) {
      const unsigned other = watch.binary.lit;
      assert(!watch.binary.redundant);
      const bool weakened =
            other == replacament && GET_OPTION(incremental);
      kissat_eliminate_binary(solver, weakened, not_lit, other);
    } else {
      const reference ref = watch.large.ref;
      clause *d = kissat_dereference_clause(solver, ref);
      if (d->garbage) {
        continue;
      }
      LOGCLS(d, "removing %s", LOGLIT(not_lit));
      kissat_eliminate_clause(solver, false, d, not_lit);
    }
  }
  RELEASE_WATCHES(*neg_watches);

  kissat_flush_units_while_connected(solver);
}

static void try_to_eliminate_all_variables_again(kissat *solver) {
  LOG("trying to elimination all variables again");
  flags *all_flags = solver->flags;
  for (all_variables(idx)) {
    flags *flags = all_flags + idx;
    flags->eliminate = true;
  }
  solver->limits.eliminate.variables.removed = 0;
}

static void set_next_elimination_bound(kissat *solver, bool complete) {
  const unsigned max_bound = GET_OPTION(eliminatebound);
  const unsigned current_bound = solver->bounds.eliminate.additional_clauses;
  assert(current_bound <= max_bound);

  if (complete) {
    if (current_bound == max_bound) {
      kissat_phase(solver, "eliminate", GET(eliminations),
            "completed maximum elimination bound %u",
            current_bound);
      limits *limits = &solver->limits;
      statistics *statistics = &solver->statistics;
      limits->eliminate.variables.added = statistics->variables_added;
      limits->eliminate.variables.removed = statistics->variables_removed;
#ifndef QUIET
      bool first = !solver->bounds.eliminate.max_bound_completed++;
      REPORT(!first, first ? '!' : ':');
#endif
    } else {
      const unsigned next_bound =
            !current_bound ? 1 : MIN(2 * current_bound, max_bound);
      kissat_phase(solver, "eliminate", GET(eliminations),
            "completed elimination bound %u next %u",
            current_bound, next_bound);
      solver->bounds.eliminate.additional_clauses = next_bound;
      try_to_eliminate_all_variables_again(solver);
      REPORT(0, '^');
    }
  } else
    kissat_phase(solver, "eliminate", GET(eliminations),
          "incomplete elimination bound %u", current_bound);
}

static bool can_eliminate_variable(kissat *solver, unsigned idx) {
  flags *flags = FLAGS(idx);

  if (!flags->active) {
    return false;
  }
  if (!flags->eliminate) {
    return false;
  }

  LOG("next elimination candidate %s", LOGVAR(idx));

  LOG("marking %s as not removed", LOGVAR(idx));
  flags->eliminate = false;

  return true;
}

static bool eliminate_variable(kissat *solver, unsigned idx) {
  assert(!solver->inconsistent);
  if (!can_eliminate_variable(solver, idx)) {
    return false;
  }
  unsigned lit;
  if (!kissat_generate_resolvents(solver, idx, &lit)) {
    return false;
  }
  connect_resolvents(solver);
  if (!solver->inconsistent) {
    if (solver->found_equivalence == INVALID_LIT) {
      weaken_clauses(solver, lit);
    } else {
      eliminate_equivalence(solver, idx, solver->found_equivalence);
    }
  }
  INC(eliminated);
  if (solver->found_equivalence == INVALID_LIT) {
    kissat_mark_eliminated_variable(solver, idx);
  } else {
    kissat_mark_substituted_variable(solver, idx, solver->found_equivalence);
  }
  if (solver->gate_eliminated) {
    INC(gates_eliminated);
#ifdef METRICS
    assert(*solver->gate_eliminated < UINT64_MAX);
    *solver->gate_eliminated += 1;
#endif
  }
  return true;
}

static void eliminate_variables(kissat *solver) {
  kissat_very_verbose(solver,
        "trying to eliminate variables with bound %u",
        solver->bounds.eliminate.additional_clauses);
#ifndef QUIET
  unsigned active_before = solver->active;
#endif
  unsigned last_round_eliminated;
  unsigned eliminated = 0;
  uint64_t tried = 0;

  SET_EFFORT_LIMIT(resolution_limit,
        eliminate, eliminate_resolutions,
        eliminate_adjustment(solver));

  bool forward_subsumption_complete = false;
  int round = 0;

  const bool forward = GET_OPTION(forward);

  for (;;) {
    round++;
    last_round_eliminated = 0;
    forward_subsumption_complete = true;
    LOG("starting new elimination round %d", round);

    kissat_release_heap(solver, &solver->schedule);
    if (forward) {
      unsigned *propagate = solver->propagate;
      forward_subsumption_complete =
            kissat_forward_subsume_during_elimination(solver);
      if (solver->inconsistent) {
        break;
      }
      kissat_flush_large_connected(solver);
      kissat_connect_irredundant_large_clauses(solver);
      solver->propagate = propagate;
      kissat_flush_units_while_connected(solver);
      if (solver->inconsistent) {
        break;
      }
    } else {
      kissat_connect_irredundant_large_clauses(solver);
    }

    const unsigned last_round_scheduled = schedule_variables(solver);
    kissat_very_verbose(solver,
          "scheduled %u variables %.0f%% to eliminate "
          "in round %d", last_round_scheduled,
          kissat_percent(last_round_scheduled,
                solver->active), round);

    if (forward_subsumption_complete && !last_round_scheduled) {
      break;
    }

    const bool eliminateheap = GET_OPTION(eliminateheap);
    unsigned idx = 0;
    bool done = false;

    while (!done && !solver->inconsistent) {
      if (eliminateheap && kissat_empty_heap(&solver->schedule)) {
        done = true;
      } else if (!eliminateheap && idx == solver->vars) {
        done = true;
      } else if (TERMINATED(eliminate_terminated_1)) {
        done = true;
      } else if (solver->statistics.eliminate_resolutions >
            resolution_limit) {
        kissat_extremely_verbose(solver,
              "eliminate round %u hits "
              "resolution limit %"
              PRIu64 " at %" PRIu64 " resolutions",
              round, resolution_limit,
              solver->statistics.
              eliminate_resolutions);
        done = true;
      } else {
        tried++;
        if (eliminateheap) {
          idx = kissat_pop_max_heap(solver, &solver->schedule);
        }
        if (eliminate_variable(solver, idx)) {
          eliminated++, last_round_eliminated++;
        }
        if (!solver->inconsistent) {
          kissat_flush_units_while_connected(solver);
        }
        if (!eliminateheap) {
          idx++;
        }
      }
    }
    if (!solver->inconsistent) {
      kissat_flush_large_connected(solver);
      kissat_dense_collect(solver);
    }
    kissat_phase(solver, "eliminate", GET(eliminations),
          "eliminated %u variables %.0f%% in round %u",
          last_round_eliminated,
          kissat_percent(last_round_eliminated,
                last_round_scheduled), round);
#ifndef QUIET
    {
      const bool round_successful =
            solver->inconsistent || last_round_eliminated;
      REPORT(!round_successful, 'e');
    }
#endif
    if (solver->inconsistent) {
      break;
    }
    if (eliminateheap) {
      kissat_release_heap(solver, &solver->schedule);
    }
    if (round == GET_OPTION(eliminaterounds)) {
      break;
    }
    if (solver->statistics.eliminate_resolutions > resolution_limit) {
      break;
    }
    if (TERMINATED(eliminate_terminated_2)) {
      break;
    }
  }

  const unsigned remain = kissat_size_heap(&solver->schedule);
  kissat_release_heap(solver, &solver->schedule);
#ifndef QUIET
  kissat_very_verbose(solver,
        "eliminated %u variables %.0f%% of %" PRIu64 " tried"
        " (%u remain %.0f%%)",
        eliminated, kissat_percent(eliminated, tried), tried,
        remain, kissat_percent(remain, solver->active));
  kissat_phase(solver, "eliminate", GET(eliminations),
        "eliminated %u variables %.0f%% out of %u in %d rounds",
        eliminated, kissat_percent(eliminated, active_before),
        active_before, round);
#endif
  if (!solver->inconsistent) {
    const bool complete =
          !remain && !last_round_eliminated && forward_subsumption_complete;
    set_next_elimination_bound(solver, complete);
    if (!complete && !GET_OPTION(eliminatekeep)) {
      const flags *end = solver->flags + VARS;
#ifndef QUIET
      unsigned dropped = 0;
#endif
      for (struct flags *f = solver->flags; f != end; f++)
        if (f->eliminate) {
          f->eliminate = false;
#ifndef QUIET
          dropped++;
#endif
        }

      kissat_very_verbose(solver,
            "dropping %u eliminate candidates", dropped);
    }
  }
}

static void setup_elim_bounds(kissat *solver) {
  solver->bounds.eliminate.clause_size =
        SCALE_LIMIT(eliminations, eliminateclslim);

  solver->bounds.eliminate.occurrences =
        SCALE_LIMIT(eliminations, eliminateocclim);

  kissat_phase(solver, "eliminate", GET(eliminations),
        "occurrence limit %u and clause limit %u",
        solver->bounds.eliminate.occurrences,
        solver->bounds.eliminate.clause_size);

  solver->bounds.subsume.clause_size =
        SCALE_LIMIT(eliminations, subsumeclslim);

  solver->bounds.subsume.occurrences =
        SCALE_LIMIT(eliminations, subsumeocclim);

  kissat_phase(solver, "subsume", GET(eliminations),
        "occurrence limit %u clause limit %u",
        solver->bounds.subsume.occurrences,
        solver->bounds.subsume.clause_size);

  solver->bounds.xor.clause_size =
        log(MAX(solver->bounds.eliminate.occurrences, 1)) / log(2.0) + 0.5;
  if (solver->bounds.xor.clause_size > (unsigned) GET_OPTION(xorsclslim)) {
    solver->bounds.xor.clause_size = (unsigned) GET_OPTION(xorsclslim);
  }
  assert(solver->bounds.xor.clause_size < 32);

  LOG("maximum XOR base clause size %u", solver->bounds.xor.clause_size);
}

static void init_map_and_kitten(kissat *solver) {
  if (!GET_OPTION(definitions)) {
    return;
  }
  assert(!solver->kitten);
  solver->kitten = kitten_embedded(solver);
}

static void reset_map_and_kitten(kissat *solver) {
  if (solver->kitten) {
    kitten_release(solver->kitten);
    solver->kitten = 0;
  }
}

static void eliminate(kissat *solver) {
  RETURN_IF_DELAYED(eliminate);
  kissat_backtrack_propagate_and_flush_trail(solver);
  assert(!solver->inconsistent);
  STOP_SEARCH_AND_START_SIMPLIFIER(eliminate);
  kissat_phase(solver, "eliminate", GET(eliminations),
        "elimination limit of %" PRIu64 " conflicts hit",
        solver->limits.eliminate.conflicts);
  const changes before = kissat_changes(solver);
  setup_elim_bounds(solver);
  init_map_and_kitten(solver);
  litwatches saved;
  INIT_STACK(saved);
  kissat_enter_dense_mode(solver, 0, &saved);
  eliminate_variables(solver);
  kissat_resume_sparse_mode(solver, true, 0, &saved);
  RELEASE_STACK(saved);
  reset_map_and_kitten(solver);
  kissat_check_statistics(solver);
  const changes after = kissat_changes(solver);
  const bool changed = kissat_changed(before, after);
  UPDATE_DELAY(changed, eliminate);
  STOP_SIMPLIFIER_AND_RESUME_SEARCH(eliminate);
}

int kissat_eliminate(kissat *solver) {
  assert(!solver->inconsistent);
  INC(eliminations);
  eliminate(solver);
  UPDATE_CONFLICT_LIMIT(eliminate, eliminations, NLOGNLOGN, true);
  solver->waiting.eliminate.reduce = solver->statistics.reductions + 1;
  solver->last.eliminate = solver->statistics.search_ticks;
  return solver->inconsistent ? 20 : 0;
}

#include "allocate.h"
#include "analyze.h"
#include "backbone.h"
#include "backtrack.h"
#include "decide.h"
#include "inline.h"
#include "internal.h"
#include "logging.h"
#include "print.h"
#include "proprobe.h"
#include "report.h"
#include "terminate.h"
#include "trail.h"
#include "utilities.h"

static void schedule_backbone_candidates(kissat *solver,
      unsigneds *candidates) {
  flags *flags = solver->flags;
  unsigned not_rescheduled = 0;
  for (all_variables(idx)) {
    const struct flags *f = flags + idx;
    if (!f->active) {
      continue;
    }
    const unsigned lit = LIT(idx);
    if (f->backbone0) {
      PUSH_STACK(*candidates, lit);
      LOG("rescheduling backbone literal candidate %s", LOGLIT(lit));
    } else {
      not_rescheduled++;
    }
    if (f->backbone1) {
      const unsigned not_lit = NOT(lit);
      PUSH_STACK(*candidates, not_lit);
      LOG("rescheduling backbone literal candidate %s",
            LOGLIT(not_lit));
    } else {
      not_rescheduled++;
    }
  }
  const size_t rescheduled = SIZE_STACK(*candidates);
#ifndef QUIET
  const unsigned active_literals = 2u * solver->active;
  kissat_very_verbose(solver,
        "rescheduled %zu backbone candidate literals %.0f%%",
        rescheduled,
        kissat_percent(rescheduled, active_literals));
#endif
  if (rescheduled && GET_OPTION(backbonefocus)) {
    kissat_extremely_verbose(solver,
          "dropping %u not-rescheduled "
          "backbone candidate literals %.0f%%",
          not_rescheduled,
          kissat_percent(not_rescheduled,
                active_literals));
    return;
  }
  if (not_rescheduled) {
    for (all_variables(idx)) {
      struct flags *f = flags + idx;
      if (!f->active) {
        continue;
      }
      const unsigned lit = LIT(idx);
      if (!f->backbone0) {
        LOG("scheduling backbone literal candidate %s", LOGLIT(lit));
        PUSH_STACK(*candidates, lit);
      }
      if (!f->backbone1) {
        const unsigned not_lit = NOT(lit);
        LOG("scheduling backbone literal candidate %s",
              LOGLIT(not_lit));
        PUSH_STACK(*candidates, not_lit);
      }
    }
  }
#ifndef QUIET
  const size_t total = SIZE_STACK(*candidates);
  kissat_very_verbose(solver,
        "scheduled %zu backbone candidate literals %.0f%%"
        " in total", total,
        kissat_percent(total, active_literals));
#endif
}

static void keep_backbone_candidates(kissat *solver, unsigneds *candidates) {
  flags *flags = solver->flags;
  size_t prioritized = 0;
  size_t remain = 0;
  for (all_stack(unsigned, lit, *candidates)) {
    const unsigned idx = IDX(lit);
    const struct flags *f = flags + idx;
    if (!f->active) {
      continue;
    }
    remain++;
    if (NEGATED(lit)) {
      prioritized += f->backbone1;
    } else {
      prioritized += f->backbone0;
    }
  }
  assert(prioritized <= remain);
  if (!remain) {
    kissat_very_verbose(solver, "no backbone candidates remain");
#ifndef NDEBUG
    for (all_variables(idx)) {
      const struct flags *f = flags + idx;
      if (!f->active) {
        continue;
      }
      assert(!f->backbone0);
      assert(!f->backbone1);
    }
#endif
    return;
  }
#ifndef QUIET
  const size_t active_literals = 2u * solver->active;
#endif
  if (prioritized == remain)
    kissat_very_verbose(solver, "keeping all remaining %zu backbone "
          "candidates %.0f%% prioritized (all were)",
          remain, kissat_percent(remain, active_literals));
  else if (!prioritized) {
    for (all_stack(unsigned, lit, *candidates)) {
      const unsigned idx = IDX(lit);
      struct flags *f = flags + idx;
      if (!f->active) {
        continue;
      }
      if (NEGATED(lit)) {
        assert(!f->backbone1);
        f->backbone1 = true;
      } else {
        assert(!f->backbone0);
        f->backbone0 = true;
      }
    }
    kissat_very_verbose(solver, "keeping all remaining %zu backbone "
          "candidates %.0f%% prioritized (none was)",
          remain, kissat_percent(remain, active_literals));
  } else {
    kissat_very_verbose(solver, "keeping %zu backbone candidates %.0f%% "
          "prioritized (%.0f%% of remaining %zu)",
          prioritized,
          kissat_percent(prioritized, active_literals),
          kissat_percent(prioritized, remain), remain);
  }
}

static inline void backbone_assign(kissat *solver, unsigned_array *trail,
      value *values, assigned *assigned,
      unsigned lit, bool redundant, unsigned reason) {
  const unsigned not_lit = NOT(lit);
  assert(!values[lit]);
  assert(!values[not_lit]);
  values[lit] = 1;
  values[not_lit] = -1;
  PUSH_ARRAY(*trail, lit);
  const unsigned idx = IDX(lit);
  struct assigned *a = assigned + idx;
  a->reason = reason;
  a->redundant = redundant;
  a->level = solver->level;
}

static inline clause *backbone_propagate_literal(kissat *solver,
      const bool stop_early,
      const watches *const all_watches,
      unsigned_array *trail, value *values,
      assigned *assigned, unsigned lit) {
  LOG("backbone propagating %s", LOGLIT(lit));
  assert(VALID_INTERNAL_LITERAL(lit));
  assert(values[lit] > 0);

  const unsigned not_lit = NOT(lit);
  assert(values[not_lit] < 0);

  assert(not_lit < LITS);
  const watches *const watches = all_watches + not_lit;

  const watch *const begin_watches = BEGIN_CONST_WATCHES(*watches);
  const watch *const end_watches = END_CONST_WATCHES(*watches);
  const watch *p = begin_watches;

  while (p != end_watches) {
    const watch watch = *p++;
    if (watch.type.binary) {
      const unsigned other = watch.binary.lit;
      assert(VALID_INTERNAL_LITERAL(other));
      const value value = values[other];
      if (value > 0) {
        continue;
      }
      const bool redundant = watch.binary.redundant;
      if (value < 0) {
        return kissat_binary_conflict(solver, redundant, not_lit, other);
      }
      assert(!value);
      backbone_assign(solver, trail, values, assigned,
            other, redundant, lit);
      LOG("backbone assign %s reason binary clause %s %s",
            LOGLIT(other), LOGLIT(other), LOGLIT(not_lit));
    } else {
      if (stop_early) {
#ifndef NDEBUG
        for (const union watch *q = p + 1; q != end_watches; q++) {
          const union watch watch = *q++;
          assert(!watch.type.binary);
        }
#endif
        break;
      }

      p++;
    }
  }

  const size_t touched = p - begin_watches;
  solver->ticks += 1 + kissat_cache_lines(touched, sizeof(watch));

  return 0;
}

static inline clause *backbone_propagate(kissat *solver, unsigned_array *trail,
      value *values, assigned *assigned) {
  const bool stop_early = solver->large_clauses_watched_after_binary_clauses;

  clause *conflict = 0;
  solver->ticks = 0;

  const watches *const watches = solver->watches;
  unsigned *propagate = solver->propagate;

  while (!conflict && propagate != END_ARRAY(*trail))
    conflict = backbone_propagate_literal(solver, stop_early, watches, trail,
                values, assigned, *propagate++);

  assert(solver->propagate <= propagate);
  const unsigned propagated = propagate - solver->propagate;
  solver->propagate = propagate;

  ADD(backbone_propagations, propagated);
  ADD(probing_propagations, propagated);
  ADD(propagations, propagated);

  const uint64_t ticks = solver->ticks;

  ADD(backbone_ticks, ticks);
  ADD(probing_ticks, ticks);
  ADD(ticks, ticks);

  return conflict;
}

static inline void backbone_backtrack(kissat *solver,
      unsigned_array *trail, value *values,
      unsigned *saved, unsigned decision_level) {
  assert(decision_level <= solver->level);
  unsigned *end_trail = END_ARRAY(*trail);
  assert(saved != end_trail);
  LOG("backbone backtracking to trail level %zu and decision level %u",
        (size_t)(saved - BEGIN_ARRAY(*trail)), decision_level);
  while (saved != end_trail) {
    const unsigned lit = *--end_trail;
    const unsigned not_lit = NOT(lit);
    LOG("backbone unassign %s", LOGLIT(lit));
    assert(values[lit] > 0);
    assert(values[not_lit] < 0);
    values[lit] = values[not_lit] = 0;
  }
  SET_END_OF_ARRAY(solver->trail, saved);
  solver->level = decision_level;
  solver->propagate = saved;
}

static unsigned backbone_analyze(kissat *solver, clause *conflict) {
  assert(conflict);
  LOGCLS(conflict, "backbone analyzing");
  assert(conflict->size == 2);

  assigned *const assigned = solver->assigned;

  kissat_push_analyzed(solver, assigned, IDX(conflict->lits[0]));
  kissat_push_analyzed(solver, assigned, IDX(conflict->lits[1]));

  const unsigned *t = END_ARRAY(solver->trail);

  for (;;) {
    assert(t > BEGIN_ARRAY(solver->trail));

    unsigned lit = *--t;

    const unsigned lit_idx = IDX(lit);
    const struct assigned *a = assigned + lit_idx;
    if (!a->analyzed) {
      continue;
    }

    LOG("backbone analyzing %s", LOGLIT(lit));
    const unsigned reason = a->reason;
    assert(reason != UNIT_REASON);
    assert(reason != DECISION_REASON);
    const unsigned reason_idx = IDX(reason);
    const struct assigned *b = assigned + reason_idx;
    if (!b->analyzed) {
      LOG("reason %s of %s not yet analyzed",
            LOGLIT(reason), LOGLIT(lit));
      kissat_push_analyzed(solver, assigned, reason_idx);
    } else {
      LOG("backbone UIP %s", LOGLIT(reason));
      kissat_reset_only_analyzed_literals(solver);
      return reason;
    }
  }
}

#ifndef NDEBUG

static void check_large_clauses_watched_after_binary_clauses(kissat *solver) {
  for (all_literals(lit)) {
    bool large = false;
    for (all_binary_blocking_watches(watch, WATCHES(lit)))
      if (watch.type.binary) {
        assert(!large);
      } else {
        large = true;
      }
  }
}

#endif

static unsigned compute_backbone(kissat *solver) {
#ifndef NDEBUG
  if (solver->large_clauses_watched_after_binary_clauses) {
    check_large_clauses_watched_after_binary_clauses(solver);
  }
#endif
  size_t failed = 0;
  unsigneds units;
  unsigneds candidates;
  INIT_STACK(candidates);
  INIT_STACK(units);
  schedule_backbone_candidates(solver, &candidates);
#ifndef QUIET
  const size_t scheduled = SIZE_STACK(candidates);
#endif
#if defined(METRICS) && (!defined(QUIET) || !defined(NDEBUG))
  const uint64_t implied_before = solver->statistics.backbone_implied;
#endif
  unsigned_array *trail = &solver->trail;
  value *values = solver->values;
  flags *flags = solver->flags;
  assigned *assigned = solver->assigned;

  assert(kissat_propagated(solver));
  assert(kissat_trail_flushed(solver));

  unsigned inconsistent = INVALID_LIT;

  SET_EFFORT_LIMIT(ticks_limit, backbone, backbone_ticks,
        kissat_linear(1 + solver->active));

  size_t round_limit = GET_OPTION(backbonerounds);
  assert(solver->statistics.backbone_computations);
  round_limit *= solver->statistics.backbone_computations;
  const size_t max_rounds = GET_OPTION(backbonemaxrounds);
  if (round_limit > max_rounds) {
    round_limit = max_rounds;
  }

  size_t round = 0;

  for (;;) {
    if (round >= round_limit) {
      kissat_very_verbose(solver, "backbone round limit %zu hit", round);
      break;
    }
    const uint64_t ticks = solver->statistics.backbone_ticks;
    if (ticks > ticks_limit) {
      kissat_very_verbose(solver,
            "backbone ticks limit %" PRIu64 " hit "
            "after %" PRIu64 " ticks", ticks_limit, ticks);
      break;
    }
    size_t previous = failed;
    assert(!solver->inconsistent);
    if (TERMINATED(backbone_terminated_2)) {
      break;
    }
    round++;
    INC(backbone_rounds);
    LOG("starting backbone round %zu", round);
    unsigned *const begin_candidates = BEGIN_STACK(candidates);
    assert(!solver->level);
#if !defined(QUIET) && defined(METRICS)
    size_t decisions = 0;
    uint64_t propagated = solver->statistics.backbone_propagations;
#endif
    unsigned active_before = solver->active;
    {
      unsigned *q = begin_candidates;
      const unsigned *p = begin_candidates;
      const unsigned *const end_candidates = END_STACK(candidates);
      while (p != end_candidates) {
        assert(!solver->inconsistent);
        const unsigned probe = *q++ = *p++;
        const value value = values[probe];
        if (value > 0) {
          q--;
          LOG("removing satisfied backbone probe %s", LOGLIT(probe));
          const unsigned idx = IDX(probe);
          struct flags *f = flags + idx;
          if (NEGATED(probe)) {
            f->backbone1 = false;
          } else {
            f->backbone0 = false;
          }
          continue;
        }
        if (value < 0) {
          const unsigned idx = IDX(probe);
          struct assigned *a = assigned + idx;
          if (a->level)
            LOG("skipping falsified backbone probe %s",
                  LOGLIT(probe));
          else {
            LOG("removing root-level falsified backbone probe %s",
                  LOGLIT(probe));
            q--;
          }
          continue;
        }
        if (solver->statistics.backbone_ticks > ticks_limit) {
          break;
        }
        if (TERMINATED(backbone_terminated_3)) {
          break;
        }
        const unsigned level = solver->level;
        unsigned *const saved = END_ARRAY(*trail);
        assert(level != UINT_MAX);
#if !defined(QUIET) && defined(METRICS)
        decisions++;
#endif
        solver->level = level + 1;
        INC(backbone_probes);
        backbone_assign(solver, trail, values, assigned,
              probe, false, DECISION_REASON);
        LOG("backbone assume %s", LOGLIT(probe));
        clause *conflict = backbone_propagate(solver,
                    trail, values, assigned);
        if (!conflict) {
          LOG("propagating backbone probe %s successful",
                LOGLIT(probe));
          continue;
        }

        failed++;
        INC(backbone_units);
        q--;

        LOG("propagating backbone probe %s failed", LOGLIT(probe));
        unsigned uip = backbone_analyze(solver, conflict);
        unsigned not_uip = NOT(uip);
        backbone_backtrack(solver, trail, values, saved, level);

        PUSH_STACK(units, not_uip);
        backbone_assign(solver, trail, values, assigned,
              not_uip, false, UNIT_REASON);
        LOG("backbone forced assign %s", LOGLIT(not_uip));
        assert(failed == SIZE_STACK(units));

        conflict = backbone_propagate(solver, trail, values, assigned);
        if (conflict) {
          LOG("propagating backbone forced %s failed",
                LOGLIT(not_uip));
          inconsistent = not_uip;
          break;
        }

        LOG("propagating backbone forced %s successful",
              LOGLIT(not_uip));
      }
#ifndef QUIET
      size_t remain = end_candidates - p;
      if (remain)
        kissat_extremely_verbose(solver,
              "backbone round %zu aborted with "
              "%zu candidates %.0f%% remaining",
              round, remain,
              kissat_percent(remain, scheduled));
      else
        kissat_extremely_verbose(solver,
              "backbone round %zu completed with "
              "all %zu scheduled candidates tried",
              round, scheduled);
#endif
      while (p != end_candidates) {
        *q++ = *p++;
      }

      SET_END_OF_STACK(candidates, q);
    }
    if (inconsistent == INVALID_LIT) {
      LOG("flushing satisfied probe candidates");
      unsigned *q = begin_candidates;
      const unsigned *p = begin_candidates;
      const unsigned *const end_candidates = END_STACK(candidates);
      while (p != end_candidates) {
        const unsigned probe = *q++ = *p++;
        const value value = values[probe];
        if (value > 0) {
          q--;
          LOG("removing satisfied backbone probe %s",
                LOGLIT(probe));
          const unsigned idx = IDX(probe);
          struct flags *f = flags + idx;
          if (NEGATED(probe)) {
            f->backbone1 = false;
          } else {
            f->backbone0 = false;
          }
          continue;
        }
        if (value < 0) {
          LOG("keeping falsified probe %s", LOGLIT(probe));
          continue;
        }
        assert(!value);
        LOG("keeping unassigned probe %s", LOGLIT(probe));
      }
      LOG("flushed %zu probe candidates",
            (size_t)(q - BEGIN_STACK(candidates)));
      SET_END_OF_STACK(candidates, q);
    }
    if (!EMPTY_ARRAY(*trail)) {
      backbone_backtrack(solver, trail, values, BEGIN_ARRAY(*trail), 0);
    }
    if (inconsistent == INVALID_LIT && previous < failed) {
      for (size_t i = previous; i < failed; i++) {
        const unsigned unit = PEEK_STACK(units, i);
        LOG("assigning backbone unit %s", LOGLIT(unit));
        kissat_learned_unit(solver, unit);
      }
      if (kissat_probing_propagate(solver, 0, true)) {
        break;
      }
    }
    assert(solver->active <= active_before);
    unsigned implied = active_before - solver->active;
    assert(failed <= failed);
    ADD(backbone_implied, implied);
#ifndef QUIET
#ifdef METRICS
    propagated = solver->statistics.backbone_propagations - propagated;
    kissat_very_verbose(solver,
          "backbone round %zu with %zu decisions "
          "(%.2f propagations per decision)",
          round, decisions,
          kissat_average(propagated, decisions));
#endif
    size_t left = SIZE_STACK(candidates);
    kissat_very_verbose(solver,
          "backbone round %zu produced %zu failed literals"
          " %u implied (%zu candidates left %.0f%%)",
          round, failed - previous, implied,
          left, kissat_percent(left, scheduled));
#endif
    if (inconsistent != INVALID_LIT) {
      break;
    }
    if (EMPTY_STACK(candidates)) {
      break;
    }
  }

  if (inconsistent != INVALID_LIT && !solver->inconsistent) {
    LOG("assuming forced unit %s", LOGLIT(inconsistent));
    kissat_learned_unit(solver, inconsistent);
    (void) kissat_probing_propagate(solver, 0, true);
    assert(solver->inconsistent);
  }
  RELEASE_STACK(units);
  if (solver->inconsistent)
    kissat_phase(solver, "backbone", GET(backbone_computations),
          "inconsistent binary clauses");
  else {
    if (GET_OPTION(backbonekeep)) {
      keep_backbone_candidates(solver, &candidates);
    }
#if defined(METRICS) && (!defined(QUIET) || !defined(NDEBUG))
    assert(implied_before <= solver->statistics.backbone_implied);
#endif
#if defined(METRICS) && !defined(QUIET)
    const uint64_t total_implied =
          solver->statistics.backbone_implied - implied_before;
    kissat_phase(solver, "backbone", GET(backbone_computations),
          "found %zu backbone literals %" PRIu64
          " implied in %zu rounds", failed, total_implied, round);
#endif
  }
  RELEASE_STACK(candidates);
  return failed;
}

void kissat_binary_clauses_backbone(kissat *solver) {
  if (solver->inconsistent) {
    return;
  }
  if (!GET_OPTION(backbone)) {
    return;
  }
  if (TERMINATED(backbone_terminated_1)) {
    return;
  }
  RETURN_IF_DELAYED(backbone);
  assert(solver->watching);
  assert(solver->probing);
  assert(!solver->level);
  START(backbone);
  INC(backbone_computations);
#if !defined(NDEBUG) || defined(METRICS)
  assert(!solver->backbone_computing);
  solver->backbone_computing = true;
#endif
  const unsigned failed = compute_backbone(solver);
  REPORT(!failed, 'b');
  const bool success = (failed > 0);
  UPDATE_DELAY(success, backbone);
#if !defined(NDEBUG) || defined(METRICS)
  assert(solver->backbone_computing);
  solver->backbone_computing = false;
#endif
  STOP(backbone);
}

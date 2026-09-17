#include "analyze.h"
#include "backtrack.h"
#include "bump.h"
#include "deduce.h"
#include "inline.h"
#include "learn.h"
#include "minimize.h"
#include "rank.h"
#include "shrink.h"
#include "sort.h"

#include <inttypes.h>

static bool one_literal_on_conflict_level(kissat *solver,
      clause *conflict,
      unsigned *conflict_level_ptr) {
  assert(conflict);
  assert(conflict->size > 1);

  unsigned conflict_level = INVALID_LEVEL;
  unsigned literals_on_conflict_level = 0;
  unsigned forced_lit = INVALID_LIT;

  assigned *all_assigned = solver->assigned;

  unsigned *lits = conflict->lits;
  const unsigned conflict_size = conflict->size;
  const unsigned *const end_of_lits = lits + conflict_size;

  for (const unsigned *p = lits; p != end_of_lits; p++) {
    const unsigned lit = *p;
    assert(VALUE(lit) < 0);
    const unsigned idx = IDX(lit);
    const unsigned level = all_assigned[idx].level;
    if (conflict_level == level) {
      if (++literals_on_conflict_level > 1 && level == solver->level) {
        break;
      }
    } else if (conflict_level == INVALID_LEVEL || conflict_level < level) {
      forced_lit = lit;
      conflict_level = level;
      literals_on_conflict_level = 1;
    }
  }
  assert(conflict_level != INVALID_LEVEL);
  assert(literals_on_conflict_level);

  LOG("found %u literals on conflict level %u",
        literals_on_conflict_level, conflict_level);
  *conflict_level_ptr = conflict_level;

  if (!conflict_level) {
    solver->inconsistent = true;
    LOG("learned empty clause from conflict at conflict level zero");
    CHECK_AND_ADD_EMPTY();
    ADD_EMPTY_TO_PROOF();
    return false;
  }

  if (conflict_level < solver->level) {
    LOG("forced backtracking due to conflict level %u < level %u",
          conflict_level, solver->level);
    kissat_backtrack_after_conflict(solver, conflict_level);
  }

  if (conflict_size > 2) {
    for (unsigned i = 0; i < 2; i++) {
      const unsigned lit = lits[i];
      const unsigned lit_idx = IDX(lit);
      unsigned highest_position = i;
      unsigned highest_literal = lit;
      unsigned highest_level = all_assigned[lit_idx].level;
      for (unsigned j = i + 1; j < conflict_size; j++) {
        const unsigned other = lits[j];
        const unsigned other_idx = IDX(other);
        const unsigned level = all_assigned[other_idx].level;
        if (highest_level >= level) {
          continue;
        }
        highest_literal = other;
        highest_position = j;
        highest_level = level;
        if (highest_level == conflict_level) {
          break;
        }
      }
      if (highest_position == i) {
        continue;
      }
      reference ref = INVALID_REF;
      if (highest_position > 1) {
        ref = kissat_reference_clause(solver, conflict);
        kissat_unwatch_blocking(solver, lit, ref);
      }
      lits[highest_position] = lit;
      lits[i] = highest_literal;
      if (highest_position > 1) {
        kissat_watch_blocking(solver, lits[i], lits[!i], ref);
      }
    }
  }

  if (literals_on_conflict_level > 1) {
    return false;
  }

  assert(literals_on_conflict_level == 1);
  assert(forced_lit != INVALID_LIT);

  LOG("reusing conflict as driving clause of %s", LOGLIT(forced_lit));
  kissat_backtrack_after_conflict(solver, solver->level - 1);
  if (conflict_size == 2) {
    assert(conflict == &solver->conflict);
    const unsigned other = lits[0] ^ lits[1] ^ forced_lit;
    kissat_assign_binary(solver, conflict->redundant, forced_lit, other);
  } else {
    const reference ref = kissat_reference_clause(solver, conflict);
    kissat_assign_reference(solver, forced_lit, ref, conflict);
  }

  return true;
}

static inline void mark_reason_side_literal(kissat *solver,
      assigned *all_assigned, unsigned lit) {
  const unsigned idx = IDX(lit);
  const assigned *a = all_assigned + idx;
  if (a->level && !a->analyzed) {
    kissat_push_analyzed(solver, all_assigned, idx);
  }
}

static inline void analyze_reason_side_literal(kissat *solver, size_t limit,
      ward *arena,
      assigned *all_assigned, unsigned lit) {
  const unsigned idx = IDX(lit);
  const assigned *a = all_assigned + idx;
  assert(a->level);
  assert(a->analyzed);
  assert(a->reason != UNIT_REASON);
  if (a->reason == DECISION_REASON) {
    return;
  }
  if (a->binary) {
    const unsigned other = a->reason;
    mark_reason_side_literal(solver, all_assigned, other);
  } else {
    const reference ref = a->reason;
    assert(ref < SIZE_STACK(solver->arena));
    clause *c = (clause *)(arena + ref);
    const unsigned not_lit = NOT(lit);
    INC(search_ticks);
    for (all_literals_in_clause(other, c))
      if (other != not_lit) {
        assert(other != lit);
        mark_reason_side_literal(solver, all_assigned, other);
        if (SIZE_STACK(solver->analyzed) > limit) {
          break;
        }
      }
  }
}

static void analyze_reason_side_literals(kissat *solver) {
  if (!GET_OPTION(bumpreasons)) {
    return;
  }
  if (solver->probing) {
    return;
  }
  if (solver->delays.bumpreasons.count) {
    solver->delays.bumpreasons.count--;
    LOG("bump reasons still delayed (%u more times)",
          solver->delays.bumpreasons.count);
    return;
  }
  const double decision_rate = AVERAGE(decision_rate);
  const int decision_rate_limit = GET_OPTION(bumpreasonsrate);
  if (decision_rate >= decision_rate_limit) {
    LOG("decision rate %g >= limit %d",
          decision_rate, decision_rate_limit);
    return;
  }
  assigned *all_assigned = solver->assigned;
#ifndef NDEBUG
  for (all_stack(unsigned, lit, solver->clause)) {
    assert(all_assigned[IDX(lit)].analyzed);
  }
#endif
  LOG("trying to bump reason side literals too");
  const size_t saved = SIZE_STACK(solver->analyzed);
  const size_t limit = GET_OPTION(bumpreasonslimit) * saved;
  LOG("analyzed already %zu literals thus limit %zu", saved, limit);
  ward *arena = BEGIN_STACK(solver->arena);
  for (all_stack(unsigned, lit, solver->clause)) {
    analyze_reason_side_literal(solver, limit, arena, all_assigned, lit);
    if (SIZE_STACK(solver->analyzed) > limit) {
      break;
    }
  }
  if (SIZE_STACK(solver->analyzed) > limit) {
    LOG("too many additional reason side literals");
    while (SIZE_STACK(solver->analyzed) > saved) {
      const unsigned idx = POP_STACK(solver->analyzed);
      struct assigned *a = all_assigned + idx;
      LOG("marking %s as not analyzed", LOGVAR(idx));
      assert(a->analyzed);
      a->analyzed = false;
    }
    if (solver->delays.bumpreasons.current < UINT_MAX) {
      solver->delays.bumpreasons.current++;
      LOG("solver delay bump reasons interval increased to %u",
            solver->delays.bumpreasons.current);

    }
  } else if (solver->delays.bumpreasons.current) {
    solver->delays.bumpreasons.current /= 2;
    LOG("bump reasons delay interval decreased to %u",
          solver->delays.bumpreasons.current);
  } else {
    LOG("keeping zero bump reasons delays");
  }

  solver->delays.bumpreasons.count = solver->delays.bumpreasons.current;
  LOG("next bump reasons delayed %u times",
        solver->delays.bumpreasons.count);
}

#if 0

#define RANK_LITERAL_BY_INVERSE_LEVEL(LIT) \
  (~assigned[IDX (LIT)].level)

static void sort_deduced_clause(kissat *solver) {
  unsigneds *clause = &solver->clause;
  assigned *assigned = solver->assigned;
  RADIX_STACK(unsigned, unsigned, *clause, RANK_LITERAL_BY_INVERSE_LEVEL);
}

#else

#define RADIX_SORT_LEVELS_LIMIT 32

#define RANK_LEVEL(A) (A)
#define SMALLER_LEVEL(A,B) (RANK_LEVEL(A) < RANK_LEVEL(B))

static void sort_levels(kissat *solver) {
  unsigneds *levels = &solver->levels;
  size_t glue = SIZE_STACK(*levels);
  if (glue < RADIX_SORT_LEVELS_LIMIT) {
    SORT_STACK(unsigned, *levels, SMALLER_LEVEL);
  } else {
    RADIX_STACK(unsigned, unsigned, *levels, RANK_LEVEL);
  }
  LOG("sorted %zu levels", glue);
}

static void sort_deduced_clause(kissat *solver) {
  sort_levels(solver);
#ifndef NDEBUG
  const size_t size_frames = SIZE_STACK(solver->frames);
#endif
  frame *frames = BEGIN_STACK(solver->frames);
  unsigned pos = 1;
  const unsigned *const begin_levels = BEGIN_STACK(solver->levels);
  const unsigned *const end_levels = END_STACK(solver->levels);
  unsigned const *p = end_levels;
  while (p != begin_levels) {
    const unsigned level = *--p;
    assert(level < size_frames);
    frame *f = frames + level;
    const unsigned used = f->used;
#ifndef NDEBUG
    f->saved = used;
#endif
    assert(used > 0);
    assert(UINT_MAX - used >= pos);
    f->used = pos;
    pos += used;
  }
  unsigneds *clause = &solver->clause;
  const size_t size_clause = SIZE_STACK(*clause);
#ifndef NDEBUG
  assert(pos == size_clause);
#endif
  unsigned const *begin_clause = BEGIN_STACK(*clause);
  const unsigned *const end_clause = END_STACK(*clause);
  assert(begin_clause < end_clause);

  unsigneds *shadow = &solver->shadow;
  while (SIZE_STACK(*shadow) < size_clause) {
    PUSH_STACK(*shadow, INVALID_LIT);
  }

  const unsigned not_uip = *begin_clause++;
  POKE_STACK(*shadow, 0, not_uip);

  const assigned *const assigned = solver->assigned;

  for (const unsigned *p = begin_clause; p != end_clause; p++) {
    const unsigned lit = *p;
    const unsigned idx = IDX(lit);
    const struct assigned *a = assigned + idx;
    const unsigned level = a->level;
    assert(level < size_frames);
    frame *f = frames + level;
    const unsigned pos = f->used++;
    POKE_STACK(*shadow, pos, lit);
  }

  assert(size_clause == SIZE_STACK(*shadow));
  SWAP(unsigneds, *clause, *shadow);

  pos = 1;
  p = end_levels;
  while (p != begin_levels) {
    const unsigned level = *--p;
    assert(level < size_frames);
    frame *f = frames + level;
    const unsigned end = f->used;
    assert(pos < end);
    f->used = end - pos;
    assert(f->used == f->saved);
    pos = end;
  }

  CLEAR_STACK(*shadow);
  LOGTMP("level sorted deduced");

#ifndef NDEBUG
  unsigned prev_level = solver->level;
  for (all_stack(unsigned, lit, solver->clause)) {
    const unsigned idx = IDX(lit);
    const unsigned lit_level = assigned[idx].level;
    assert(prev_level >= lit_level);
    prev_level = lit_level;
  }
#endif
}

#endif

static void reset_levels(kissat *solver) {
  LOG("reset %zu marked levels", SIZE_STACK(solver->levels));
  frame *frames = BEGIN_STACK(solver->frames);
#ifndef NDEBUG
  const size_t size_frames = SIZE_STACK(solver->frames);
#endif
  for (all_stack(unsigned, level, solver->levels)) {
    assert(level < size_frames);
    frame *f = frames + level;
    assert(f->used > 0);
    f->used = 0;
  }
  CLEAR_STACK(solver->levels);
}

void kissat_reset_only_analyzed_literals(kissat *solver) {
  LOG("reset %zu analyzed variables", SIZE_STACK(solver->analyzed));
  assigned *assigned = solver->assigned;
  for (all_stack(unsigned, idx, solver->analyzed)) {
    assert(idx < VARS);
    struct assigned *a = assigned + idx;
    assert(!a->poisoned);
    assert(!a->removable);
    assert(!a->shrinkable);
    a->analyzed = false;
  }
  CLEAR_STACK(solver->analyzed);
}

static void reset_removable(kissat *solver) {
  LOG("reset %zu removable variables", SIZE_STACK(solver->removable));
  assigned *assigned = solver->assigned;
#ifndef NDEBUG
  unsigned not_removable = 0;
#endif
  for (all_stack(unsigned, idx, solver->removable)) {
    assert(idx < VARS);
    struct assigned *a = assigned + idx;
    assert(a->removable || !not_removable++);
    a->removable = false;
  }
  CLEAR_STACK(solver->removable);
}

static void reset_analysis_but_not_analyzed_literals(kissat *solver) {
  reset_removable(solver);
  reset_levels(solver);
  LOG("reset %zu learned literals", SIZE_STACK(solver->clause));
  CLEAR_STACK(solver->clause);
}

static void update_trail_average(kissat *solver) {
  assert(!solver->probing);
#if defined(LOGGING) || !defined(QUIET)
  const unsigned size = SIZE_ARRAY(solver->trail);
  const unsigned assigned = size - solver->unflushed;
  const unsigned active = solver->active;
  const double filled = kissat_percent(assigned, active);
#else
  (void) solver;
#endif
  LOG("trail filled %.0f%% (size %u, unflushed %u, active %u)",
        filled, size, solver->unflushed, active);
#ifndef QUIET
  UPDATE_AVERAGE(trail, filled);
#endif
}

static void update_decision_rate_average(kissat *solver) {
  assert(!solver->probing);
  const uint64_t current = DECISIONS;
  const uint64_t previous = solver->averages[solver->stable].saved_decisions;
  assert(previous <= current);
  const uint64_t decisions = current - previous;
  solver->averages[solver->stable].saved_decisions = current;
  UPDATE_AVERAGE(decision_rate, decisions);
}

static void analyze_failed_literal(kissat *solver, clause *conflict) {
  assert(solver->level == 1);
  const unsigned failed = FRAME(1).decision;

  LOGCLS(conflict, "analyzing failed literal %s conflict", LOGLIT(failed));

  unsigneds *units = &solver->clause;
  assert(EMPTY_STACK(*units));
  assert(EMPTY_STACK(solver->analyzed));

  const unsigned not_failed = NOT(failed);
  assigned *all_assigned = solver->assigned;
#ifndef NDEBUG
  const value *const values = solver->values;
#endif
  unsigned const *t = END_ARRAY(solver->trail);
  unsigned unresolved = 0;
  unsigned unit = INVALID_LIT;

  for (all_literals_in_clause(lit, conflict)) {
    assert(lit != failed);
    if (lit == not_failed) {
      LOG("negation %s of failed literal %s occurs in conflict",
            LOGLIT(not_failed), LOGLIT(failed));
      goto DONE;
    }
    assert(values[lit] < 0);
    const unsigned idx = IDX(lit);
    assigned *a = all_assigned + idx;
    if (!a->level) {
      continue;
    }
    assert(a->level == 1);
    LOG("analyzing conflict literal %s", LOGLIT(lit));
    kissat_push_analyzed(solver, all_assigned, idx);
    unresolved++;
  }

  for (;;) {
    unsigned lit;
    assigned *a;
    do {
      assert(t > BEGIN_ARRAY(solver->trail));
      lit = *--t;
      assert(values[lit] > 0);
      const unsigned idx = IDX(lit);
      a = all_assigned + idx;
    } while (!a->analyzed);
    if (unresolved == 1) {
      unit = NOT(lit);
      LOG("learning additional unit %s", LOGLIT(unit));
      PUSH_STACK(*units, unit);
    }
    if (a->binary) {
      const unsigned other = a->reason;
      LOGBINARY(lit, other, "resolving %s reason", LOGLIT(lit));
      assert(other != failed);
      assert(other != unit);
      assert(values[other] < 0);
      if (other == not_failed) {
        LOG("negation %s of failed literal %s in reason",
              LOGLIT(not_failed), LOGLIT(failed));
        goto DONE;
      }
      const unsigned idx = IDX(other);
      assigned *b = all_assigned + idx;
      assert(b->level == 1);
      if (!b->analyzed) {
        LOG("analyzing reason literal %s", LOGLIT(other));
        kissat_push_analyzed(solver, all_assigned, idx);
        unresolved++;
      }
    } else {
      assert(a->reason != UNIT_REASON);
      assert(a->reason != DECISION_REASON);
      const reference ref = a->reason;
      LOGREF(ref, "resolving %s reason", LOGLIT(lit));
      clause *reason = kissat_dereference_clause(solver, ref);
      for (all_literals_in_clause(other, reason)) {
        assert(other != NOT(lit));
        assert(other != failed);
        if (other == lit) {
          continue;
        }
        if (other == unit) {
          continue;
        }
        if (other == not_failed) {
          LOG("negation %s of failed literal %s occurs in reason",
                LOGLIT(not_failed), LOGLIT(failed));
          goto DONE;
        }
        assert(values[other] < 0);
        const unsigned idx = IDX(other);
        assigned *b = all_assigned + idx;
        if (!b->level) {
          continue;
        }
        assert(b->level == 1);
        if (b->analyzed) {
          continue;
        }
        LOG("analyzing reason literal %s", LOGLIT(other));
        kissat_push_analyzed(solver, all_assigned, idx);
        unresolved++;
      }
    }
    assert(unresolved > 0);
    unresolved--;
    LOG("after resolving %s there are %u unresolved literals",
          LOGLIT(lit), unresolved);
  }
DONE:
  LOG("learning negated failed literal %s", LOGLIT(not_failed));
  PUSH_STACK(*units, not_failed);

  if (!solver->probing) {
    kissat_update_learned(solver, 0, 1);
  }

  LOG("failed literal %s produced %zu units",
        LOGLIT(failed), SIZE_STACK(*units));

  kissat_backtrack_without_updating_phases(solver, 0);

  for (all_stack(unsigned, lit, *units)) {
    kissat_learned_unit(solver, lit);
  }
  CLEAR_STACK(*units);
  solver->iterating = true;
}

int kissat_analyze(kissat *solver, clause *conflict) {
  if (solver->inconsistent) {
    assert(!solver->level);
    return 20;
  }

  START(analyze);
  if (!solver->probing) {
    update_trail_average(solver);
    update_decision_rate_average(solver);
#ifndef QUIET
    UPDATE_AVERAGE(level, solver->level);
#endif
  }
  int res;
  do {
    LOGCLS(conflict, "analyzing conflict %" PRIu64, CONFLICTS);
    unsigned conflict_level;
    if (one_literal_on_conflict_level(solver, conflict, &conflict_level)) {
      res = 1;
    } else if (!conflict_level) {
      res = -1;
    } else if (conflict_level == 1) {
      analyze_failed_literal(solver, conflict);
      res = 1;
    } else if ((conflict = kissat_deduce_first_uip_clause(solver, conflict))) {
      reset_analysis_but_not_analyzed_literals(solver);
      res = 0;
    } else {
      if (GET_OPTION(minimize)) {
        sort_deduced_clause(solver);
        kissat_minimize_clause(solver);
        if (GET_OPTION(shrink)) {
          kissat_shrink_clause(solver);
        }
      }
      analyze_reason_side_literals(solver);
      kissat_learn_clause(solver);
      reset_analysis_but_not_analyzed_literals(solver);
      res = 1;
    }
    if (!EMPTY_STACK(solver->analyzed)) {
      if (!solver->probing) {
        kissat_bump(solver);
      }
      kissat_reset_only_analyzed_literals(solver);
    }
  } while (!res);
  STOP(analyze);
  return res > 0 ? 0 : 20;
}

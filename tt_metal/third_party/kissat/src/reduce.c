#include "allocate.h"
#include "collect.h"
#include "inline.h"
#include "print.h"
#include "reduce.h"
#include "rank.h"
#include "report.h"
#include "trail.h"

#include <inttypes.h>
#include <math.h>

bool kissat_reducing(kissat *solver) {
  if (!GET_OPTION(reduce)) {
    return false;
  }
  if (!solver->statistics.clauses_redundant) {
    return false;
  }
  if (CONFLICTS < solver->limits.reduce.conflicts) {
    return false;
  }
  return true;
}

typedef struct reducible reducible;

struct reducible {
  uint64_t rank;
  unsigned ref;
};

#define RANK_REDUCIBLE(RED) \
  (RED).rank

// *INDENT-OFF*
typedef STACK (reducible) reducibles;
// *INDENT-ON*

static bool collect_reducibles(kissat *solver, reducibles *reds,
      reference start_ref) {
  assert(start_ref != INVALID_REF);
  assert(start_ref <= SIZE_STACK(solver->arena));
  ward *const arena = BEGIN_STACK(solver->arena);
  clause *start = (clause *)(arena + start_ref);
  const clause *const end = (clause *) END_STACK(solver->arena);
  assert(start < end);
  while (start != end && (!start->redundant || start->keep)) {
    start = kissat_next_clause(start);
  }
  if (start == end) {
    solver->first_reducible = INVALID_REF;
    LOG("no reducible clause candidate left");
    return false;
  }
  const reference redundant = (ward *) start - arena;
#ifdef LOGGING
  if (redundant < solver->first_reducible)
    LOG("updating redundant clauses start from %zu to %zu",
          (size_t) solver->first_reducible, (size_t) redundant);
  else
    LOG("no update to redundant clauses start %zu",
          (size_t) solver->first_reducible);
#endif
  solver->first_reducible = redundant;
  const unsigned tier2 = GET_OPTION(tier2);
#ifndef QUIET
  size_t flushed_hyper_ternary_clauses = 0;
  size_t used_hyper_ternary_clauses = 0;
#endif
  for (clause *c = start; c != end; c = kissat_next_clause(c)) {
    if (!c->redundant) {
      continue;
    }
    if (c->garbage) {
      continue;
    }
    if (c->reason) {
      continue;
    }
    if (c->hyper) {
      assert(c->size == 3);
      if (c->used) {
#ifndef QUIET
        used_hyper_ternary_clauses++;
#endif
        c->used = false;
      } else {
#ifndef QUIET
        flushed_hyper_ternary_clauses++;
#endif
        kissat_mark_clause_as_garbage(solver, false, c);
      }
      continue;
    }
    if (c->keep) {
      continue;
    }
    if (c->used) {
      c->used--;
      if (c->glue <= tier2) {
        continue;
      }
    }
    assert(!c->garbage);
    assert(kissat_clause_in_arena(solver, c));
    reducible red;
    const uint64_t negative_size = ~c->size;
    const uint64_t negative_glue = ~c->glue;
    red.rank = negative_size | (negative_glue << 32);
    red.ref = (ward *) c - arena;
    PUSH_STACK(*reds, red);
  }
#ifndef QUIET
// *INDENT-OFF*
  size_t total_hyper_ternary_clauses =
    flushed_hyper_ternary_clauses + used_hyper_ternary_clauses;
  if (flushed_hyper_ternary_clauses)
    kissat_phase (solver, "reduced", GET (reductions),
      "reduced %zu unused hyper ternary clauses %.0f%% out of %zu",
      flushed_hyper_ternary_clauses,
	kissat_percent (flushed_hyper_ternary_clauses,
	                total_hyper_ternary_clauses),
      total_hyper_ternary_clauses);
// *INDENT-ON*
#endif
  if (EMPTY_STACK(*reds)) {
    LOG("did not find any reducible redundant clause");
    return false;
  }
  return true;
}

#define USEFULNESS RANK_REDUCIBLE

static void sort_reducibles(kissat *solver, reducibles *reds) {
  RADIX_STACK(reducible, uint64_t, *reds, USEFULNESS);
}

static void mark_less_useful_clauses_as_garbage(kissat *solver,
      reducibles *reds) {
  const size_t size = SIZE_STACK(*reds);
  size_t target = size * (GET_OPTION(reducefraction) / 100.0);
#ifndef QUIET
  statistics *statistics = &solver->statistics;
  const size_t clauses =
        statistics->clauses_irredundant + statistics->clauses_redundant;
  kissat_phase(solver, "reduce",
        GET(reductions),
        "reducing %zu (%.0f%%) out of %zu (%.0f%%) "
        "reducible clauses",
        target, kissat_percent(target, size),
        size, kissat_percent(size, clauses));
#endif
  unsigned reduced = 0;
  ward *arena = BEGIN_STACK(solver->arena);
  const reducible *const begin = BEGIN_STACK(*reds);
  const reducible *const end = END_STACK(*reds);
  for (const reducible *p = begin; p != end && target--; p++) {
    clause *c = (clause *)(arena + p->ref);
    assert(kissat_clause_in_arena(solver, c));
    assert(!c->garbage);
    assert(!c->keep);
    assert(!c->reason);
    assert(c->redundant);
    LOGCLS(c, "reducing");
    kissat_mark_clause_as_garbage(solver, false, c);
    reduced++;
  }
  ADD(clauses_reduced, reduced);
}

static bool compacting(kissat *solver) {
  if (!GET_OPTION(compact)) {
    return false;
  }
  unsigned inactive = solver->vars - solver->active;
  unsigned limit = GET_OPTION(compactlim) / 1e2 * solver->vars;
  bool compact = (inactive > limit);
  LOG("%u inactive variables %.0f%% <= limit %u %.0f%%",
        inactive, kissat_percent(inactive, solver->vars),
        limit, kissat_percent(limit, solver->vars));
  return compact;
}

static void force_restart_before_reduction(kissat *solver) {
  if (!GET_OPTION(reducerestart)) {
    return;
  }
  if (!solver->stable && (GET_OPTION(reducerestart) < 2)) {
    return;
  }
  LOG("forcing restart before reduction");
  kissat_restart_and_flush_trail(solver);
}

int kissat_reduce(kissat *solver) {
  START(reduce);
  INC(reductions);
  kissat_phase(solver, "reduce", GET(reductions),
        "reduce limit %" PRIu64 " hit after %" PRIu64
        " conflicts", solver->limits.reduce.conflicts, CONFLICTS);
  force_restart_before_reduction(solver);
  bool compact = compacting(solver);
  reference start = compact ? 0 : solver->first_reducible;
  if (start != INVALID_REF) {
#ifndef QUIET
    size_t arena_size = SIZE_STACK(solver->arena);
    size_t words_to_sweep = arena_size - start;
    size_t bytes_to_sweep = sizeof(word) * words_to_sweep;
    kissat_phase(solver, "reduce", GET(reductions),
          "reducing clauses after offset %"
          REFERENCE_FORMAT " in arena", start);
    kissat_phase(solver, "reduce", GET(reductions),
          "reducing %zu words %s %.0f%%",
          words_to_sweep, FORMAT_BYTES(bytes_to_sweep),
          kissat_percent(words_to_sweep, arena_size));
#endif
    if (kissat_flush_and_mark_reason_clauses(solver, start)) {
      reducibles reds;
      INIT_STACK(reds);
      if (collect_reducibles(solver, &reds, start)) {
        sort_reducibles(solver, &reds);
        mark_less_useful_clauses_as_garbage(solver, &reds);
        RELEASE_STACK(reds);
        kissat_sparse_collect(solver, compact, start);
      } else if (compact) {
        kissat_sparse_collect(solver, compact, start);
      } else {
        kissat_unmark_reason_clauses(solver, start);
      }
    } else {
      assert(solver->inconsistent);
    }
  } else {
    kissat_phase(solver, "reduce", GET(reductions), "nothing to reduce");
  }
  UPDATE_CONFLICT_LIMIT(reduce, reductions, SQRT, false);
  REPORT(0, '-');
  STOP(reduce);
  return solver->inconsistent ? 20 : 0;
}

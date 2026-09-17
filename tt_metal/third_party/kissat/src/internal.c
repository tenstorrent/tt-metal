#include "add.h"
#include "allocate.h"
#include "assume.h"
#include "backtrack.h"
#include "error.h"
#include "protect.h"
#include "search.h"
#include "import.h"
#include "inline.h"
#include "inlineframes.h"
#include "print.h"
#include "propsearch.h"
#include "require.h"
#include "resize.h"
#include "resources.h"

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

kissat *kissat_init(void) {
  kissat *solver = kissat_calloc(0, 1, sizeof * solver);
#ifndef NOPTIONS
  kissat_init_options(&solver->options);
#else
  kissat_init_options();
#endif
#ifndef QUIET
  kissat_init_profiles(&solver->profiles);
#endif
  START(total);
  kissat_init_queue(solver);
  assert(INTERNAL_MAX_LIT < UINT_MAX);
  kissat_push_frame(solver, UINT_MAX);
  kissat_init_reap(solver, &solver->reap);
  solver->watching = true;
  solver->conflict.size = 2;
  solver->conflict.keep = true;
  solver->scinc = 1.0;
  solver->first_reducible = INVALID_REF;
  solver->last_irredundant = INVALID_REF;
  solver->rephased.last = 'O';
#ifndef NDEBUG
  kissat_init_checker(solver);
#endif
  return solver;
}

#define DEALLOC_GENERIC(NAME, ELEMENTS_PER_BLOCK) \
do { \
  const size_t block_size = ELEMENTS_PER_BLOCK * sizeof *solver->NAME; \
  kissat_dealloc (solver, solver->NAME, solver->size, block_size); \
  solver->NAME = 0; \
} while (0)

#define DEALLOC_VARIABLE_INDEXED(NAME) \
  DEALLOC_GENERIC (NAME, 1)

#define DEALLOC_LITERAL_INDEXED(NAME) \
  DEALLOC_GENERIC (NAME, 2)

#define RELEASE_LITERAL_INDEXED_STACKS(NAME,ACCESS) \
do { \
  for (all_stack (unsigned, IDX_RILIS, solver->active)) \
    { \
      const unsigned LIT_RILIS = LIT (IDX_RILIS); \
      const unsigned NOT_LIT_RILIS = NOT (LIT_RILIS); \
      RELEASE_STACK (ACCESS (LIT_RILIS)); \
      RELEASE_STACK (ACCESS (NOT_LIT_RILIS)); \
    } \
  DEALLOC_LITERAL_INDEXED (NAME); \
} while (0)

void kissat_release(kissat *solver) {
  kissat_require_initialized(solver);

  kissat_release_heap(solver, &solver->scores);
  kissat_release_heap(solver, &solver->schedule);

  kissat_release_clueue(solver, &solver->clueue);
  kissat_release_reap(solver, &solver->reap);

  kissat_release_phases(solver);
  kissat_release_cache(solver);
  RELEASE_STACK(solver->nonces);

  RELEASE_STACK(solver->export);
  RELEASE_STACK(solver->import);

  DEALLOC_VARIABLE_INDEXED(assigned);
  DEALLOC_VARIABLE_INDEXED(flags);
  DEALLOC_VARIABLE_INDEXED(protect);
  DEALLOC_VARIABLE_INDEXED(links);

  DEALLOC_LITERAL_INDEXED(marks);
  DEALLOC_LITERAL_INDEXED(values);
  DEALLOC_LITERAL_INDEXED(watches);

  RELEASE_STACK(solver->import);
  RELEASE_STACK(solver->eliminated);
  RELEASE_STACK(solver->extend);
  RELEASE_STACK(solver->witness);
  RELEASE_STACK(solver->etrail);

  RELEASE_STACK(solver->vectors.stack);
  RELEASE_STACK(solver->delayed);

  RELEASE_STACK(solver->clause);
  RELEASE_STACK(solver->shadow);
#if defined(LOGGING) || !defined(NDEBUG)
  RELEASE_STACK(solver->resolvent);
#endif

  RELEASE_STACK(solver->arena);

  RELEASE_STACK(solver->units);
  RELEASE_STACK(solver->frames);
  RELEASE_STACK(solver->sorter);

  RELEASE_ARRAY(solver->trail, solver->size);
  RELEASE_STACK(solver->assumptions);
  RELEASE_STACK(solver->redundant_assumptions);
  RELEASE_STACK(solver->failed);
  RELEASE_STACK(solver->assumption_implied);

  RELEASE_STACK(solver->analyzed);
  RELEASE_STACK(solver->levels);
  RELEASE_STACK(solver->minimize);
  RELEASE_STACK(solver->poisoned);
  RELEASE_STACK(solver->promote);
  RELEASE_STACK(solver->removable);
  RELEASE_STACK(solver->shrinkable);
  RELEASE_STACK(solver->xorted[0]);
  RELEASE_STACK(solver->xorted[1]);

  RELEASE_STACK(solver->ranks);

  RELEASE_STACK(solver->antecedents[0]);
  RELEASE_STACK(solver->antecedents[1]);
  RELEASE_STACK(solver->gates[0]);
  RELEASE_STACK(solver->gates[1]);
  RELEASE_STACK(solver->resolvents);

#if !defined(NDEBUG) || !defined(NPROOFS)
  RELEASE_STACK(solver->added);
  RELEASE_STACK(solver->removed);
#endif

#if !defined(NDEBUG) || !defined(NPROOFS) || defined(LOGGING)
  RELEASE_STACK(solver->original);
#endif

#ifndef QUIET
  RELEASE_STACK(solver->profiles.stack);
#endif

#ifndef NDEBUG
  kissat_release_checker(solver);
#endif
#if !defined(NDEBUG) && defined(METRICS)
  uint64_t leaked = solver->statistics.allocated_current;
  if (leaked)
    if (!getenv("LEAK")) {
      kissat_fatal("internally leaking %" PRIu64 " bytes", leaked);
    }
#endif

  kissat_free(0, solver, sizeof * solver);
}

void kissat_reserve(kissat *solver, int max_var) {
  kissat_require_initialized(solver);
  kissat_require(0 <= max_var,
        "negative maximum variable argument '%d'", max_var);
  kissat_require(max_var <= EXTERNAL_MAX_VAR,
        "invalid maximum variable argument '%d'", max_var);
  kissat_increase_size(solver, (unsigned) max_var);
}

int kissat_get_option(kissat *solver, const char *name) {
  kissat_require_initialized(solver);
  kissat_require(name, "name zero pointer");
#ifndef NOPTIONS
  return kissat_options_get(&solver->options, name);
#else
  (void) solver;
  return kissat_options_get(name);
#endif
}

int kissat_set_option(kissat *solver, const char *name, int new_value) {
#ifndef NOPTIONS
  kissat_require_initialized(solver);
  kissat_require(name, "name zero pointer");
  return kissat_options_set(&solver->options, name, new_value, !GET(searches));
#else
  (void) solver, (void) new_value;
  return kissat_options_get(name);
#endif
}

void kissat_set_decision_limit(kissat *solver, unsigned limit) {
  kissat_require_initialized(solver);
  limits *limits = &solver->limits;
  limited *limited = &solver->limited;
  statistics *statistics = &solver->statistics;
  limited->decisions = true;
  assert(UINT64_MAX - limit >= statistics->decisions);
  limits->decisions = statistics->decisions + limit;
  LOG("set decision limit to %" PRIu64 " after %u decisions",
        limits->decisions, limit);
}

void kissat_set_conflict_limit(kissat *solver, unsigned limit) {
  kissat_require_initialized(solver);
  limits *limits = &solver->limits;
  limited *limited = &solver->limited;
  statistics *statistics = &solver->statistics;
  limited->conflicts = true;
  assert(UINT64_MAX - limit >= statistics->conflicts);
  limits->conflicts = statistics->conflicts + limit;
  LOG("set conflict limit to %" PRIu64 " after %u conflicts",
        limits->conflicts, limit);
}

void kissat_print_statistics(kissat *solver) {
#ifndef QUIET
  kissat_require_initialized(solver);
  const int verbosity = kissat_verbosity(solver);
  if (verbosity < 0) {
    return;
  }
  if (GET_OPTION(profile)) {
    kissat_section(solver, "profiling");
    kissat_profiles_print(solver);
  }
  const bool complete = GET_OPTION(statistics);
  kissat_section(solver, "statistics");
  const bool verbose = (complete || verbosity > 0);
  kissat_statistics_print(solver, verbose);
#ifndef NPROOFS
  if (solver->proof) {
    kissat_section(solver, "proof");
    kissat_print_proof_statistics(solver, verbose);
  }
#endif
#ifndef NDEBUG
  if (GET_OPTION(check) > 1) {
    kissat_section(solver, "checker");
    kissat_print_checker_statistics(solver, verbose);
  }
#endif
  kissat_section(solver, "resources");
  kissat_print_resources(solver);
#endif
  (void) solver;
}

void kissat_add(kissat *solver, int elit) {
  kissat_require_initialized(solver);
  solver->extended = false;
  if (elit) {
    kissat_require_valid_external_internal(elit);
    if (!kissat_add_external_literal(solver, elit)) {
      kissat_require(false, "use 'kissat_protect' or 'incremental' option");
    }
  } else {
    kissat_finish_external_clause(solver, false);
  }
}

int kissat_solve(kissat *solver) {
  kissat_require_initialized(solver);
  kissat_require(EMPTY_STACK(solver->clause),
        "incomplete clause (terminating zero not added)");
  return kissat_search(solver);
}

void kissat_terminate(kissat *solver) {
  kissat_require_initialized(solver);
  solver->termination.flagged = ~(unsigned) 0;
  assert(solver->termination.flagged);
}

void kissat_set_terminate(kissat *solver, void *state,
      int (*terminate)(void *)) {
  solver->termination.terminate = 0;
  solver->termination.state = state;
  solver->termination.terminate = terminate;
}

int kissat_value(kissat *solver, int elit) {
  kissat_require_initialized(solver);
  kissat_require_valid_external_internal(elit);
  const unsigned eidx = ABS(elit);
  if (eidx >= SIZE_STACK(solver->import)) {
    return 0;
  }
  const int repr_elit = kissat_external_representative(solver, elit);
  const unsigned repr_eidx = ABS(repr_elit);

  const import *const import = &PEEK_STACK(solver->import, repr_eidx);
  if (!import->imported) {
    return 0;
  }
  value tmp;
  if (import->eliminated) {
    if (!solver->extended && !EMPTY_STACK(solver->extend)) {
      kissat_extend(solver);
    }
    const unsigned eliminated = import->lit;
    tmp = PEEK_STACK(solver->eliminated, eliminated);
  } else {
    const unsigned ilit = import->lit;
    tmp = VALUE(ilit);
  }
  if (!tmp) {
    return 0;
  }
  if (repr_elit < 0) {
    tmp = -tmp;
  }
  return tmp < 0 ? -elit : elit;
}

void kissat_protect(kissat *solver, int elit) {
  kissat_require_initialized(solver);
  kissat_require_valid_external_internal(elit);

  unsigned ilit = kissat_import_literal(solver, elit);
  kissat_require(ilit != INVALID_LIT,
        "use 'kissat_protect' or 'incremental' option");
  unsigned iidx = IDX(ilit);
  if (GET(searches) && !GET_OPTION(incremental)) {
    kissat_require(
          PROTECT(iidx) || FLAGS(iidx)->fixed || !FLAGS(iidx)->active,
          "use 'kissat_protect' or 'incremental' option");
  }

  kissat_protect_variable(solver, IDX(ilit));
}

void kissat_unprotect(kissat *solver, int elit) {
  kissat_require_initialized(solver);
  kissat_require_valid_external_internal(elit);

  unsigned ilit = kissat_import_literal(solver, elit);
  unsigned iidx = IDX(ilit);
  unsigned eidx = ABS(elit);

  kissat_require(ilit != INVALID_LIT, "variable %u not protected", eidx);
  kissat_require(
        PROTECT(iidx) || FLAGS(iidx)->fixed,
        "variable %u not protected", eidx);

  kissat_unprotect_variable(solver, iidx);
}

void kissat_assume(kissat *solver, int elit) {
  kissat_require_initialized(solver);
  kissat_require_valid_external_internal(elit);

  unsigned ilit = kissat_import_literal(solver, elit);
  kissat_require(ilit != INVALID_LIT,
        "use 'kissat_protect' or 'incremental' option");
  unsigned iidx = IDX(ilit);
  if (!GET_OPTION(incremental)) {
    kissat_require(
          PROTECT(iidx) || FLAGS(iidx)->fixed,
          "use 'kissat_protect' or 'incremental' option");
  }

  kissat_assume_literal(solver, elit, ilit);
}

int kissat_failed(kissat *solver, int elit) {
  kissat_require_initialized(solver);
  kissat_require_valid_external_internal(elit);

  unsigned low = 0;
  unsigned high = SIZE_STACK(solver->failed);

  while (low < high) {
    unsigned mid = (low + high) / 2;
    int failed = PEEK_STACK(solver->failed, mid);
    if (failed == elit) {
      return true;
    } else if (failed < elit) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return false;
}

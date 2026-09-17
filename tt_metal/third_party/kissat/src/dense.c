#define INLINE_SORT

#include "dense.h"
#include "inline.h"
#include "propsearch.h"
#include "proprobe.h"
#include "trail.h"

#include "sort.c"

static void flush_large_watches(kissat *solver,
      litpairs *irredundant, litwatches *redundant) {
  assert(!solver->level);
  assert(solver->watching);
#ifndef LOGGING
  LOG("flushing large watches");
  if (irredundant) {
    LOG("flushing and saving irredundant binary clauses too");
  } else {
    LOG("keep watching irredundant binary clauses");
  }
  if (redundant) {
    LOG("flushing and saving redundant clauses too");
  } else {
    LOG("keep watching redundant binary clauses");
  }
#endif
  const value *const values = solver->values;
  size_t flushed = 0, collected = 0;
  watches *all_watches = solver->watches;
  for (all_literals(lit)) {
    const value lit_value = values[lit];
    watches *watches = all_watches + lit;
    watch *begin = BEGIN_WATCHES(*watches), *q = begin;
    const watch *const end_watches = END_WATCHES(*watches), *p = q;
    while (p != end_watches) {
      const watch watch = *p++;
      if (watch.type.binary) {
        const unsigned other = watch.binary.lit;
        const value other_value = values[other];
        if (!lit_value && !other_value) {
          if (irredundant && !watch.binary.redundant) {
            const unsigned other = watch.binary.lit;
            if (lit < other) {
              const litpair litpair = {.lits = {lit, other} };
              PUSH_STACK(*irredundant, litpair);
            }
          } else if (redundant && watch.binary.redundant) {
            const unsigned other = watch.binary.lit;
            if (lit < other) {
              const litwatch litwatch = { lit, watch };
              PUSH_STACK(*redundant, litwatch);
            }
          } else {
            *q++ = watch;
          }
        } else {
          assert(lit_value > 0 || other_value > 0);
          if (lit < other) {
            const bool red = watch.binary.redundant;
            const bool hyper = watch.binary.hyper;
            kissat_delete_binary(solver, red, hyper, false, lit, other);
            collected++;
          }
        }
      } else {
        flushed++;
        p++;
      }

    }
    SET_END_OF_WATCHES(*watches, q);
  }
  LOG("flushed %zu large watches", flushed);
  LOG("collected %zu satisfied binary clauses", collected);
  if (irredundant) {
    LOG("saved %zu irredundant binary clauses", SIZE_STACK(*irredundant));
  }
  if (redundant) {
    LOG("saved %zu redundant binary clauses", SIZE_STACK(*redundant));
  }
  (void) collected;
  (void) flushed;
}

void kissat_enter_dense_mode(kissat *solver,
      litpairs *irredundant, litwatches *redundant) {
  assert(!solver->level);
  assert(solver->watching);
  assert(kissat_propagated(solver));
  LOG("entering dense mode with full occurrence lists");
  if (irredundant || redundant) {
    flush_large_watches(solver, irredundant, redundant);
  } else {
    kissat_flush_large_watches(solver);
  }
  LOG("switched to full occurrence lists");
  solver->watching = false;
}

static void resume_watching_binaries_after_elimination(kissat *solver,
      litwatches *binaries) {
  assert(binaries);
#ifdef LOGGING
  size_t resumed_watching = 0;
  size_t flushed_eliminated = 0;
#endif
  const flags *const flags = solver->flags;
  watches *all_watches = solver->watches;
  for (all_stack(litwatch, litwatch, *binaries)) {
    const unsigned first = litwatch.lit;
    watch watch = litwatch.watch;
    const unsigned second = watch.binary.lit;
    const unsigned first_idx = IDX(first);
    const unsigned second_idx = IDX(second);
    if (!flags[first_idx].eliminated && !flags[second_idx].eliminated) {
      watches *first_watches = all_watches + first;
      PUSH_WATCHES(*first_watches, watch);
      watches *second_watches = all_watches + second;
      watch.binary.lit = first;
      PUSH_WATCHES(*second_watches, watch);
#ifdef LOGGING
      resumed_watching++;
#endif
    } else {
      const bool redundant = watch.binary.redundant;
      const bool hyper = watch.binary.hyper;
      kissat_delete_binary(solver, redundant, hyper, false, first, second);
#ifdef LOGGING
      flushed_eliminated++;
#endif
    }
  }
  LOG("resumed watching %zu binary clauses flushed %zu eliminated",
        resumed_watching, flushed_eliminated);
}

static void completely_resume_watching_binaries(kissat *solver,
      litwatches *binaries) {
  assert(binaries);
#ifdef LOGGING
  size_t resumed_watching = 0;
#endif
  watches *all_watches = solver->watches;
  for (all_stack(litwatch, litwatch, *binaries)) {
    const unsigned first = litwatch.lit;
    watch watch = litwatch.watch;
    const unsigned second = watch.binary.lit;
    assert(!ELIMINATED(IDX(first)));
    assert(!ELIMINATED(IDX(second)));
    watches *first_watches = all_watches + first;
    PUSH_WATCHES(*first_watches, watch);
    watches *second_watches = all_watches + second;
    watch.binary.lit = first;
    PUSH_WATCHES(*second_watches, watch);
#ifdef LOGGING
    resumed_watching++;
#endif
  }
  LOG("resumed watching %zu binary clauses", resumed_watching);
}

static void resume_watching_irredundant_binaries(kissat *solver,
      litpairs *binaries) {
  assert(binaries);
#ifdef LOGGING
  size_t resumed_watching = 0;
#endif
  watches *all_watches = solver->watches;
  for (all_stack(litpair, litpair, *binaries)) {
    const unsigned first = litpair.lits[0];
    const unsigned second = litpair.lits[1];

    assert(!ELIMINATED(IDX(first)));
    assert(!ELIMINATED(IDX(second)));

    watches *first_watches = all_watches + first;
    watch first_watch = kissat_binary_watch(second, false, false);
    PUSH_WATCHES(*first_watches, first_watch);

    watches *second_watches = all_watches + second;
    watch second_watch = kissat_binary_watch(first, false, false);
    PUSH_WATCHES(*second_watches, second_watch);

#ifdef LOGGING
    resumed_watching++;
#endif
  }
  LOG("resumed watching %zu binary clauses", resumed_watching);
}

static void resume_watching_large_clauses_after_elimination(kissat *solver) {
#ifdef LOGGING
  size_t resumed_watching_redundant = 0;
  size_t resumed_watching_irredundant = 0;
#endif
  const flags *const flags = solver->flags;
  watches *watches = solver->watches;
  const value *const values = solver->values;
  const assigned *const assigned = solver->assigned;
  ward *const arena = BEGIN_STACK(solver->arena);

  for (all_clauses(c)) {
    if (c->garbage) {
      continue;
    }
    bool collect = false;
    for (all_literals_in_clause(lit, c)) {
      if (values[lit] > 0) {
        LOGCLS(c, "%s satisfied", LOGLIT(lit));
        collect = true;
        break;
      }
      const unsigned idx = IDX(lit);
      if (flags[idx].eliminated) {
        LOGCLS(c, "containing eliminated %s", LOGLIT(lit));
        collect = true;
        break;
      }
    }
    if (collect) {
      kissat_mark_clause_as_garbage(solver, false, c);
      continue;
    }

    assert(c->size > 2);

    unsigned *lits = c->lits;
    kissat_sort_literals(solver, values, assigned, c->size, lits);
    c->searched = 2;

    const reference ref = (ward *) c - arena;
    const unsigned l0 = lits[0];
    const unsigned l1 = lits[1];

    kissat_push_blocking_watch(solver, watches + l0, l1, ref);
    kissat_push_blocking_watch(solver, watches + l1, l0, ref);

#ifdef LOGGING
    if (c->redundant) {
      resumed_watching_redundant++;
    } else {
      resumed_watching_irredundant++;
    }
#endif
  }
  LOG("resumed watching %zu irredundant and %zu redundant large clauses",
        resumed_watching_irredundant, resumed_watching_redundant);
}

void kissat_resume_sparse_mode(kissat *solver, bool flush_eliminated,
      litpairs *irredundant, litwatches *redundant) {
  assert(!solver->level);
  assert(!solver->watching);
  if (solver->inconsistent) {
    return;
  }
  LOG("resuming sparse mode watching clauses");
  kissat_flush_large_connected(solver);
  LOG("switched to watching clauses");
  solver->watching = true;
  if (irredundant) {
    LOG("resuming watching %zu irredundant binaries",
          SIZE_STACK(*irredundant));
    resume_watching_irredundant_binaries(solver, irredundant);
  }
  if (redundant) {
    LOG("resuming watching %zu redundant binaries",
          SIZE_STACK(*redundant));
    if (flush_eliminated) {
      resume_watching_binaries_after_elimination(solver, redundant);
    } else {
      completely_resume_watching_binaries(solver, redundant);
    }
  }
  if (flush_eliminated) {
    resume_watching_large_clauses_after_elimination(solver);
  } else {
    kissat_watch_large_clauses(solver);
  }
  LOG("forcing to propagate units on all clauses");
  kissat_reset_propagate(solver);

  clause *conflict;
  if (solver->probing) {
    conflict = kissat_probing_propagate(solver, 0, true);
  } else {
    conflict = kissat_search_propagate(solver);
  }

#ifndef NDEBUG
  if (conflict) {
    assert(solver->inconsistent);
  } else {
    assert(kissat_trail_flushed(solver));
  }
#else
  (void) conflict;
#endif
}

#include "allocate.h"
#include "backtrack.h"
#include "inline.h"
#include "print.h"
#include "proprobe.h"
#include "protect.h"
#include "report.h"
#include "substitute.h"
#include "terminate.h"
#include "trail.h"
#include "weaken.h"

#include <string.h>

static uint64_t substitute_effort(kissat *solver) {
  return 10 * CLAUSES;
}

static bool really_substitute(kissat *solver) {
  if (!GET_OPTION(really)) {
    return true;
  }
  const uint64_t needed = substitute_effort(solver);
  const uint64_t search_ticks = solver->statistics.search_ticks;
  return needed < search_ticks;
}

static void assign_and_propagate_units(kissat *solver, unsigneds *units) {
  if (EMPTY_STACK(*units)) {
    return;
  }
  LOG("assigning and propagating %zu units", SIZE_STACK(*units));
  while (!solver->inconsistent && !EMPTY_STACK(*units)) {
    const unsigned unit = POP_STACK(*units);
    LOG("trying to assign and propagate unit %s now", LOGLIT(unit));
    const value value = VALUE(unit);
    if (value > 0) {
      LOG("skipping satisfied unit %s", LOGLIT(unit));
    } else if (value < 0) {
      LOG("inconsistent unit %s", LOGLIT(unit));
      CHECK_AND_ADD_EMPTY();
      ADD_EMPTY_TO_PROOF();
      solver->inconsistent = true;
    } else {
      kissat_learned_unit(solver, unit);
      INC(substitute_units);
      assert(!solver->level);
      (void) kissat_probing_propagate(solver, 0, false);
    }
  }
}

static void determine_representatives(kissat *solver, unsigned *repr) {
  size_t bytes = LITS * sizeof(unsigned);
  unsigned *mark = kissat_calloc(solver, LITS, sizeof * mark);
  unsigned *reach = kissat_malloc(solver, LITS * sizeof * reach);
  watches *all_watches = solver->watches;
  const flags *const flags = solver->flags;
  unsigned reached = 0;
  unsigneds scc;
  unsigneds work;
  unsigneds units;
  INIT_STACK(scc);
  INIT_STACK(work);
  INIT_STACK(units);
  unsigned trivial_sccs = 0;
  unsigned non_trivial_sccs = 0;
  bool inconsistent = false;
  uint64_t ticks = 0;
  for (all_literals(root)) {
    if (inconsistent) {
      break;
    }
    if (mark[root]) {
      continue;
    }
    if (!ACTIVE(IDX(root))) {
      continue;
    }
    assert(EMPTY_STACK(scc));
    assert(EMPTY_STACK(work));
    LOG("substitute root %s", LOGLIT(root));
    PUSH_STACK(work, root);
    bool failed = false;
    const unsigned mark_root = reached + 1;
    while (!inconsistent && !EMPTY_STACK(work)) {
      unsigned lit = TOP_STACK(work);
      if (lit == INVALID_LIT) {
        (void) POP_STACK(work);
        lit = POP_STACK(work);
        const unsigned not_lit = NOT(lit);
        unsigned reach_lit = reach[lit];
        unsigned mark_lit = mark[lit];
        assert(reach_lit == mark_lit);
        assert(repr[lit] == INVALID_LIT);
        watches *watches = all_watches + not_lit;
        const size_t size_watches = SIZE_WATCHES(*watches);
        ticks += 1 + kissat_cache_lines(size_watches, sizeof(watch));
        for (all_binary_blocking_watches(watch, *watches)) {
          if (!watch.type.binary) {
            continue;
          }
          const unsigned other = watch.binary.lit;
          const unsigned idx_other = IDX(other);
          if (!flags[idx_other].active) {
            continue;
          }
          assert(mark[other]);
          unsigned reach_other = reach[other];
          if (reach_other < reach_lit) {
            reach_lit = reach_other;
          }
        }
        if (reach_lit != mark_lit) {
          LOG("reach[%s] = %u", LOGLIT(lit), reach_lit);
          reach[lit] = reach_lit;
          continue;
        }
        unsigned *end_scc = END_STACK(scc);
        unsigned *begin_scc = end_scc;
        do {
          assert(begin_scc != BEGIN_STACK(scc));
        } while (*--begin_scc != lit);
        SET_END_OF_STACK(scc, begin_scc);
        const size_t size_scc = end_scc - begin_scc;
        unsigned min_lit = lit;
        if (size_scc > 1) {
          for (const unsigned *p = begin_scc; p != end_scc; p++) {
            const unsigned other = *p;
            if (other < min_lit) {
              min_lit = other;
            }
          }
          non_trivial_sccs++;
          LOG("size %zu SCC entered trough %s representative %s",
                size_scc, LOGLIT(lit), LOGLIT(min_lit));
        } else {
          trivial_sccs++;
          LOG("trivial size one SCC with %s", LOGLIT(lit));
          assert(min_lit == lit);
        }
        for (const unsigned *p = begin_scc; p != end_scc; p++) {
          const unsigned other = *p;
          LOG("repr[%s] = %s", LOGLIT(other), LOGLIT(min_lit));
          repr[other] = min_lit;
          reach[other] = UINT_MAX;

          const unsigned not_other = NOT(other);
          const unsigned repr_not_other = repr[not_other];
          if (repr_not_other == INVALID_LIT) {
            continue;
          }
          if (min_lit == repr_not_other) {
            LOG("clashing literals %s and %s in same SCC",
                  LOGLIT(other), LOGLIT(not_other));
            PUSH_STACK(units, min_lit);
            inconsistent = true;
            break;
          }
          assert(NOT(min_lit) == repr_not_other);
          if (failed) {
            continue;
          }
          const unsigned mark_not_other = mark[not_other];
          assert(mark_not_other != INVALID_LIT);
          assert(mark[root] == mark_root);
          if (mark_root > mark_not_other) {
            continue;
          }
          LOG("root %s implies both %s and %s",
                LOGLIT(root), LOGLIT(other), LOGLIT(not_other));
          const unsigned unit = NOT(root);
          PUSH_STACK(units, unit);
          failed = true;
        }
        if (inconsistent) {
          break;
        }
      } else if (!mark[lit]) {
        PUSH_STACK(work, INVALID_LIT);
        PUSH_STACK(scc, lit);
        mark[lit] = reach[lit] = ++reached;
        LOG("substitute mark[%s] = %u", LOGLIT(lit), reached);
        const unsigned not_lit = NOT(lit);
        watches *watches = all_watches + not_lit;
        const size_t size_watches = SIZE_WATCHES(*watches);
        ticks += 1 + kissat_cache_lines(size_watches, sizeof(watch));
        for (all_binary_blocking_watches(watch, *watches)) {
          if (!watch.type.binary) {
            continue;
          }
          const unsigned other = watch.binary.lit;
          const unsigned idx_other = IDX(other);
          if (!flags[idx_other].active) {
            continue;
          }
          if (!mark[other]) {
            PUSH_STACK(work, other);
          }
        }
      } else {
        (void) POP_STACK(work);
      }
    }
  }
  RELEASE_STACK(work);
  RELEASE_STACK(scc);
  kissat_extremely_verbose(solver, "determining substitution "
        "representatives took %" PRIu64
        " 'substitute_ticks'", ticks);
  ADD(substitute_ticks, ticks);
  LOG("reached %u literals", reached);
  LOG("found %u non-trivial SCCs", non_trivial_sccs);
  LOG("found %u trivial SCCs", trivial_sccs);
  LOG("found %zu units", SIZE_STACK(units));
  assign_and_propagate_units(solver, &units);
  assert(!inconsistent || solver->inconsistent);
  RELEASE_STACK(units);
  kissat_free(solver, reach, bytes);
  kissat_free(solver, mark, bytes);
  for (all_literals(lit))
    if (repr[lit] == INVALID_LIT) {
      repr[lit] = lit;
    }
}

static bool *add_representative_equivalences(kissat *solver, unsigned *repr) {
  if (solver->inconsistent) {
    return 0;
  }
  bool *eliminate = kissat_calloc(solver, VARS, sizeof * eliminate);
  for (all_variables(idx)) {
    if (!ACTIVE(idx)) {
      continue;
    }
    const unsigned lit = LIT(idx);
    const unsigned other = repr[lit];
    if (lit == other) {
      continue;
    }
    assert(other < lit);
#ifdef CHECKING_OR_PROVING
    const unsigned not_lit = NOT(lit);
    const unsigned not_other = NOT(other);

    CHECK_AND_ADD_BINARY(not_lit, other);
    ADD_BINARY_TO_PROOF(not_lit, other);

    CHECK_AND_ADD_BINARY(lit, not_other);
    ADD_BINARY_TO_PROOF(lit, not_other);
#endif
    eliminate[idx] = true;
  }
  return eliminate;
}

static void remove_representative_equivalences(kissat *solver,
      unsigned *repr, bool *eliminate) {

  bool incremental = GET_OPTION(incremental);

  if (!solver->inconsistent) {
    for (all_variables(idx)) {
      if (!eliminate[idx]) {
        continue;
      }

      assert(ACTIVE(idx));

      const unsigned lit = LIT(idx);
      const unsigned other = repr[lit];
      const unsigned not_lit = NOT(lit);
      const unsigned not_other = NOT(other);
      assert(other < lit);
      assert(not_other < not_lit);

      if (!incremental && !PROTECT(idx)) {
        REMOVE_CHECKER_BINARY(not_lit, other);
        DELETE_BINARY_FROM_PROOF(not_lit, other);

        REMOVE_CHECKER_BINARY(lit, not_other);
        DELETE_BINARY_FROM_PROOF(lit, not_other);
      }

      INC(substituted);
      kissat_mark_substituted_variable(solver, idx, other);
    }
  }
  if (eliminate) {
    kissat_dealloc(solver, eliminate, VARS, sizeof * eliminate);
  }
}

static void substitute_binaries(kissat *solver, unsigned *repr) {
  if (solver->inconsistent) {
    return;
  }
  assert(sizeof(watch) == sizeof(unsigned));
  statches *delayed_watched = (statches *) & solver->delayed;
  watches *all_watches = solver->watches;
  size_t removed = 0;
  size_t substituted = 0;
  unsigneds units;
  INIT_STACK(units);
  litwatches delayed_deleted;
  INIT_STACK(delayed_deleted);
#ifdef CHECKING_OR_PROVING
  litpairs delayed_removed;
  INIT_STACK(delayed_removed);
#endif
  for (all_literals(lit)) {
    const unsigned repr_lit = repr[lit];
    const unsigned not_repr_lit = NOT(repr_lit);
    assert(EMPTY_STACK(*delayed_watched));
    watches *watches = all_watches + lit;
    watch *begin = BEGIN_WATCHES(*watches), *q = begin;
    const watch *const end = END_WATCHES(*watches), *p = q;
    while (p != end) {
      const watch src = *p++;
      if (!src.type.binary) {
        continue;
      }
      const unsigned other = src.binary.lit;
      const unsigned repr_other = repr[other];
      LOGBINARY(lit, other, "substituting");
      const litwatch litwatch = { lit, src };
      if (repr_other == not_repr_lit) {
        LOGBINARY(repr_other, repr_lit, "becomes tautological");
        if (lit < other) {
          removed++;
          PUSH_STACK(delayed_deleted, litwatch);
        }
      } else if (repr_other == repr_lit) {
        const unsigned unit = repr_lit;
        LOG("simplifies to unit %s", LOGLIT(unit));
        if (lit < other) {
          removed++;
          PUSH_STACK(units, unit);
          PUSH_STACK(delayed_deleted, litwatch);
        }
      } else {
        watch dst = src;
        dst.binary.lit = repr_other;
        if (lit == repr_lit && other == repr_other) {
          LOGBINARY(lit, other, "unchanged");
          *q++ = dst;
        } else {
          if (lit == repr_lit) {
            LOGBINARY(repr_lit, repr_other,
                  "substituted in place");
            *q++ = dst;
          } else {
            LOGBINARY(repr_lit, repr_other, "delayed substituted");
            PUSH_STACK(*delayed_watched, dst);
          }

          if (lit < other) {
            substituted++;
#ifdef CHECKING_OR_PROVING
            ADD_BINARY_TO_PROOF(repr_lit, repr_other);
            CHECK_AND_ADD_BINARY(repr_lit, repr_other);
            const litpair litpair = { {lit, other} };
            PUSH_STACK(delayed_removed, litpair);
#endif
          }
        }
      }
    }
    SET_END_OF_WATCHES(*watches, q);
    if (lit == repr_lit) {
      continue;
    }
    watches = all_watches + repr_lit;
    for (all_stack(watch, watch, *delayed_watched)) {
      PUSH_WATCHES(*watches, watch);
    }
    CLEAR_STACK(*delayed_watched);
  }
  assign_and_propagate_units(solver, &units);
  RELEASE_STACK(units);
  for (all_stack(litwatch, litwatch, delayed_deleted)) {
    const unsigned lit = litwatch.lit;
    const watch watch = litwatch.watch;
    assert(watch.type.binary);
    const bool redundant = watch.binary.redundant;
    const bool hyper = watch.binary.hyper;
    const unsigned other = watch.binary.lit;
    kissat_delete_binary(solver, redundant, hyper, false, lit, other);
  }
  RELEASE_STACK(delayed_deleted);
#ifdef CHECKING_OR_PROVING
  for (all_stack(litpair, litpair, delayed_removed)) {
    const unsigned lit = litpair.lits[0];
    const unsigned other = litpair.lits[1];
    DELETE_BINARY_FROM_PROOF(lit, other);
    REMOVE_CHECKER_BINARY(lit, other);
  }
  RELEASE_STACK(delayed_removed);
#endif
  LOG("substituted %zu binary clauses", substituted);
  LOG("removed %zu binary clauses", removed);
}

static void substitute_clauses(kissat *solver, unsigned *repr) {
  if (solver->inconsistent) {
    return;
  }
  const value *const values = solver->values;
  value *marks = solver->marks;
  size_t substituted = 0;
  size_t removed = 0;
  unsigneds units;
  INIT_STACK(units);
  references delayed_garbage;
  INIT_STACK(delayed_garbage);
  for (all_clauses(c)) {
    if (c->garbage) {
      continue;
    }
    LOGCLS(c, "substituting");
    assert(EMPTY_STACK(solver->clause));
    bool shrink = false;
    bool satisfied = false;
    bool substitute = false;
    bool tautological = false;
    for (all_literals_in_clause(lit, c)) {
      const value lit_value = values[lit];
      if (lit_value < 0) {
        LOG("dropping falsified %s", LOGLIT(lit));
        shrink = true;
        continue;
      }
      if (lit_value > 0) {
        LOGCLS(c, "satisfied by %s", LOGLIT(lit));
        satisfied = true;
        break;
      }
      const unsigned repr_lit = repr[lit];
      const value repr_value = values[repr_lit];
      if (repr_value < 0) {
        LOG("dropping falsified substituted %s (was %s)",
              LOGLIT(repr_lit), LOGLIT(lit));
        shrink = true;
        continue;
      }
      if (repr_value > 0) {
        LOGCLS(c, "satisfied by substituted %s (was %s)",
              LOGLIT(repr_lit), LOGLIT(lit));
        satisfied = true;
        break;
      }
      if (lit != repr_lit) {
        assert(!values[repr_lit]);
        LOG("substituted literal %s (was %s)",
              LOGLIT(repr_lit), LOGLIT(lit));
        substitute = true;
      } else {
        LOG("copying literal %s", LOGLIT(lit));
      }
      if (marks[repr_lit]) {
        shrink = true;
        LOG("skipping duplicated %s", LOGLIT(repr_lit));
        continue;
      }
      const unsigned not_repr_lit = NOT(repr_lit);
      if (marks[not_repr_lit]) {
        LOG("substituted clause tautological "
              "containing both %s and %s",
              LOGLIT(not_repr_lit), LOGLIT(repr_lit));
        tautological = true;
        break;
      }
      marks[repr_lit] = true;
      PUSH_STACK(solver->clause, repr_lit);
    }
    if (satisfied || tautological) {
      kissat_mark_clause_as_garbage(solver, false, c);
      removed++;
    } else if (substitute || shrink) {
      const unsigned size = SIZE_STACK(solver->clause);
      if (!size) {
        LOG("simplifies to empty clause");

        CHECK_AND_ADD_EMPTY();
        ADD_EMPTY_TO_PROOF();

        solver->inconsistent = true;
        break;
      } else if (size == 1) {
        assert(shrink);
        removed++;
        const unsigned unit = PEEK_STACK(solver->clause, 0);
        LOGCLS(c, "simplifies to unit %s", LOGLIT(unit));
        PUSH_STACK(units, unit);
        const reference ref = kissat_reference_clause(solver, c);
        PUSH_STACK(delayed_garbage, ref);
      } else if (size == 2) {
        assert(shrink);
        substituted++;
        const unsigned first = PEEK_STACK(solver->clause, 0);
        const unsigned second = PEEK_STACK(solver->clause, 1);
        LOGCLS(c, "unsubstituted");
        const bool redundant = c->redundant;
        LOGBINARY(first, second, "substituted %s",
              redundant ? "redundant" : "irredundant");
        kissat_new_binary_clause(solver, redundant, first, second);
        kissat_mark_clause_as_garbage(solver, false, c);
      } else {
        substituted++;
        LOGCLS(c, "unsubstituted");

        const unsigned new_size = SIZE_STACK(solver->clause);
        unsigned *new_lits = BEGIN_STACK(solver->clause);

        ADD_LITS_TO_PROOF(new_size, new_lits);
        CHECK_AND_ADD_LITS(new_size, new_lits);

        DELETE_CLAUSE_FROM_PROOF(c);
        REMOVE_CHECKER_CLAUSE(c);

        const unsigned old_size = c->size;
        unsigned *old_lits = c->lits;

        assert(new_size <= old_size);
        memcpy(old_lits, new_lits, new_size * sizeof * old_lits);

        assert(shrink == (new_size < old_size));
        if (new_size < old_size) {
          c->size = new_size;
          c->searched = 2;
          if (!c->shrunken) {
            c->shrunken = true;
            old_lits[old_size - 1] = INVALID_LIT;
          }
        }
        LOGCLS(c, "unsorted substituted");
      }
    } else {
      LOGCLS(c, "unchanged");
    }
    for (all_stack(unsigned, lit, solver->clause)) {
      marks[lit] = 0;
    }
    CLEAR_STACK(solver->clause);
  }
  assign_and_propagate_units(solver, &units);
  RELEASE_STACK(units);
  for (all_stack(reference, ref, delayed_garbage)) {
    clause *c = kissat_dereference_clause(solver, ref);
    kissat_mark_clause_as_garbage(solver, false, c);
  }
  RELEASE_STACK(delayed_garbage);
  LOG("removed %zu substituted large clauses", removed);
}

static bool substitute_round(kissat *solver, unsigned round) {
  assert(!solver->inconsistent);
  const unsigned active = solver->active;
  size_t bytes = LITS * sizeof(unsigned);
  unsigned *repr = kissat_malloc(solver, bytes);
  memset(repr, 0xff, bytes);
  determine_representatives(solver, repr);
  bool *eliminate = add_representative_equivalences(solver, repr);
  substitute_binaries(solver, repr);
  substitute_clauses(solver, repr);
  remove_representative_equivalences(solver, repr, eliminate);
  kissat_dealloc(solver, repr, LITS, sizeof * repr);
  unsigned removed = active - solver->active;
  kissat_phase(solver, "substitute", GET(substitutions),
        "round %u removed %u variables %.0f%%",
        round, removed, kissat_percent(removed, active));
  kissat_check_statistics(solver);
  REPORT(!removed, 'd');
#ifdef QUIET
  (void) round;
#endif
  return !solver->inconsistent && removed;
}

static void substitute_rounds(kissat *solver) {
  START(substitute);
  INC(substitutions);
  const unsigned maxrounds = GET_OPTION(substituterounds);
  for (unsigned round = 1; round <= maxrounds; round++) {
    const uint64_t before = solver->statistics.substitute_ticks;
    if (!substitute_round(solver, round)) {
      break;
    }
    const uint64_t after = solver->statistics.substitute_ticks;
    const uint64_t ticks = after - before;
    const uint64_t reference = solver->statistics.search_ticks -
          solver->last.probe;
    const double fraction = GET_OPTION(substituteeffort) * 1e-3;
    const uint64_t limit = fraction * reference;
    if (ticks > limit) {
      kissat_extremely_verbose(solver,
            "last substitute round took %" PRIu64
            " 'substitute_ticks' " "> limit %" PRIu64
            " = %g * %" PRIu64 " 'search_ticks'",
            ticks, limit, fraction, reference);
      break;
    }
  }
  if (!solver->inconsistent) {
    kissat_watch_large_clauses(solver);
    LOG("now all large clauses are watched after binary clauses");
    solver->large_clauses_watched_after_binary_clauses = true;
    kissat_reset_propagate(solver);
    assert(!solver->level);
    (void) kissat_probing_propagate(solver, 0, true);
  }
  STOP(substitute);
}

void kissat_substitute(kissat *solver) {
  if (solver->inconsistent) {
    return;
  }
  assert(solver->probing);
  assert(solver->watching);
  assert(!solver->level);
  LOG("assuming not all large clauses watched after binary clauses");
  solver->large_clauses_watched_after_binary_clauses = false;
  if (!GET_OPTION(substitute)) {
    return;
  }
  if (TERMINATED(substitute_terminated_1)) {
    return;
  }
  if (!really_substitute(solver)) {
    return;
  }
  substitute_rounds(solver);
}

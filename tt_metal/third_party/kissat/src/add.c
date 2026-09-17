#include "add.h"
#include "backtrack.h"
#include "import.h"
#include "inline.h"
#include "internal.h"
#include "logging.h"
#include "protect.h"
#include "propsearch.h"

bool kissat_add_external_literal(kissat *solver, int elit) {
#if !defined(NDEBUG) || !defined(NPROOFS) || defined(LOGGING)
  const int checking = kissat_checking(solver);
  const bool logging = kissat_logging(solver);
  const bool proving = kissat_proving(solver);
  if (checking || logging || proving) {
    PUSH_STACK(solver->original, elit);
  }
#endif
  unsigned ilit = kissat_import_literal(solver, elit);

  if (ilit == INVALID_LIT) {
    return false;
  }

  if (kissat_export_literal(solver, ilit) != elit) {
    solver->clause_elits_substituted = true;
  }

  unsigned iidx = IDX(ilit);
  if (GET(searches) && !GET_OPTION(incremental) && !PROTECT(iidx)
        && !FLAGS(iidx)->fixed) {
    return false;
  }

  const mark mark = MARK(ilit);
  if (!mark) {
    const value value = kissat_fixed(solver, ilit);
    if (value > 0) {
      if (!solver->clause_satisfied) {
        LOG("adding root level satisfied literal %u(%d)@0=1",
              ilit, elit);
        solver->clause_satisfied = true;
      }
    } else if (value < 0) {
      LOG("adding root level falsified literal %u(%d)@0=-1",
            ilit, elit);
      if (!solver->clause_shrink) {
        solver->clause_shrink = true;
        LOG("thus original clause needs shrinking");
      }
    } else {
      MARK(ilit) = 1;
      MARK(NOT(ilit)) = -1;
      assert(SIZE_STACK(solver->clause) < UINT_MAX);
      PUSH_STACK(solver->clause, ilit);
    }
  } else if (mark < 0) {
    assert(mark < 0);
    if (!solver->clause_trivial) {
      LOG("adding dual literal %u(%d) and %u(%d)",
            NOT(ilit), -elit, ilit, elit);
      solver->clause_trivial = true;
    }
  } else {
    assert(mark > 0);
    LOG("adding duplicated literal %u(%d)", ilit, elit);
    if (!solver->clause_shrink) {
      solver->clause_shrink = true;
      LOG("thus original clause needs shrinking");
    }
  }

  return true;
}

void kissat_finish_external_clause(kissat *solver, bool restored) {
#if !defined(NDEBUG) || !defined(NPROOFS) || defined(LOGGING)
  const int checking = kissat_checking(solver);
  const bool logging = kissat_logging(solver);
  const bool proving = kissat_proving(solver);

  const size_t offset = solver->offset_of_last_original_clause;
  size_t esize = SIZE_STACK(solver->original) - offset;
  int *elits = BEGIN_STACK(solver->original) + offset;
  assert(esize <= UINT_MAX);
#endif
  if (!restored) {
    ADD_UNCHECKED_EXTERNAL(esize, elits);
  }

#if defined(LOGGING)
  const char *kind = restored ? "restored" : "original";
#endif

  const size_t isize = SIZE_STACK(solver->clause);
  unsigned *ilits = BEGIN_STACK(solver->clause);
  assert(isize < (unsigned) INT_MAX);

  if (solver->inconsistent) {
    LOG("inconsistent thus skipping %s clause", kind);
  } else if (solver->clause_satisfied) {
    LOG("skipping satisfied %s clause", kind);
  } else if (solver->clause_trivial) {
    LOG("skipping trivial %s clause", kind);
  } else {
    kissat_activate_literals(solver, isize, ilits);

    if (!isize) {
      if (solver->clause_shrink) {
        LOG("all %s clause literals root level falsified", kind);
      } else {
        LOG("found empty %s clause", kind);
      }

      if (!solver->inconsistent) {
        LOG("thus solver becomes inconsistent");
        solver->inconsistent = true;
        CHECK_AND_ADD_EMPTY();
        ADD_EMPTY_TO_PROOF();
      }
    } else if (isize == 1) {
      unsigned unit = TOP_STACK(solver->clause);

      if (solver->clause_shrink) {
        LOGUNARY(unit, "%s clause shrinks to", kind);
      } else {
        LOGUNARY(unit, "found %s", kind);
      }

      if (VALUE(unit)) {
        assert(LEVEL(unit));
        const unsigned l = LEVEL(unit);

        if (VALUE(unit) > 0) {
          kissat_backtrack_in_consistent_state(solver, l - 1);
        } else {
          kissat_backtrack_without_updating_phases(solver, l - 1);
        }
      }

      kissat_original_unit(solver, unit);

      if (!solver->level) {
        // TODO we could always do this, if we handle non-root-level conflicts
        (void) kissat_search_propagate(solver);
      }
    } else {
      reference res = kissat_new_original_clause(solver);

      const unsigned a = ilits[0];
      const unsigned b = ilits[1];

      const value u = VALUE(a);
      const value v = VALUE(b);

      const unsigned k = u ? LEVEL(a) : UINT_MAX;
      const unsigned l = v ? LEVEL(b) : UINT_MAX;

      bool assign = false;

      LOG("watched assignments %d @ %u, %d @ %u", u, k, v, l);

      if (!u && v < 0) {
        LOG("%s clause immediately forcing", kind);
        assign = true;
      } else if (u < 0 && k == l) {
        LOG("both watches falsified at level @%u", k);
        assert(v < 0);
        assert(k > 0);
        kissat_backtrack_without_updating_phases(solver, k - 1);
      } else if (u < 0) {
        LOG("watches falsified at levels @%u and @%u", k, l);
        assert(v < 0);
        assert(k > l);
        assert(l > 0);
        kissat_backtrack_without_updating_phases(solver, k - 1);
        assign = true;
      } else if (u > 0 && v < 0) {
        LOG("first watch satisfied at level @%u "
              "second falsified at level @%u", k, l);
        if (k > l) {
          kissat_backtrack_in_consistent_state(solver, k - 1);
          assign = true;
        }
      } else if (!u && v < 0) {
        LOG("first watch unassigned "
              "second falsified at level @%u", l);
        assign = true;
      } else {
        assert((!u && !v) || (v > 0 && u >= 0));
      }

      if (assign) {
        assert(solver->level > 0);

        if (isize == 2) {
          assert(res == INVALID_REF);
          kissat_assign_binary(solver, false, a, b);
        } else {
          assert(res != INVALID_REF);
          clause *c = kissat_dereference_clause(solver, res);
          kissat_assign_reference(solver, a, res, c);
        }
      }
    }
  }

#if !defined(NDEBUG) || !defined(NPROOFS)
  if (solver->clause_satisfied || solver->clause_trivial) {
#ifndef NDEBUG
    if (checking > 1) {
      kissat_remove_checker_external(solver, esize, elits);
    }
#endif
#ifndef NPROOFS
    if (proving) {
      if (esize == 1) {
        LOG("skipping deleting unit from proof");
      } else {
        kissat_delete_external_from_proof(solver, esize, elits);
      }
    }
#endif
  } else if (!solver->inconsistent &&
        (solver->clause_shrink || solver->clause_elits_substituted)) {
#ifndef NDEBUG
    if (checking > 1) {
      kissat_check_and_add_internal(solver, isize, ilits);
      kissat_remove_checker_external(solver, esize, elits);
    }
#endif
#ifndef NPROOFS
    if (proving) {
      kissat_add_lits_to_proof(solver, isize, ilits);
      kissat_delete_external_from_proof(solver, esize, elits);
    }
#endif
  }
#endif

#if !defined(NDEBUG) || !defined(NPROOFS) || defined(LOGGING)
  if (checking) {
    if (restored) {
      LOGINTS(esize, elits, "reset restored");
      RESIZE_STACK(solver->original, solver->offset_of_last_original_clause);
    } else {
      LOGINTS(esize, elits, "saved original");
      PUSH_STACK(solver->original, 0);
      solver->offset_of_last_original_clause =
            SIZE_STACK(solver->original);
    }
  } else if (logging || proving) {
    LOGINTS(esize, elits, "reset %s", kind);
    CLEAR_STACK(solver->original);
    solver->offset_of_last_original_clause = 0;
  }
#endif
  for (all_stack(unsigned, lit, solver->clause)) {
    MARK(lit) = MARK(NOT(lit)) = 0;
  }

  CLEAR_STACK(solver->clause);

  solver->clause_satisfied = false;
  solver->clause_trivial = false;
  solver->clause_shrink = false;
  solver->clause_elits_substituted = false;
}

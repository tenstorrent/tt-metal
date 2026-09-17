#include "analyze.h"
#include "assume.h"
#include "backtrack.h"
#include "decide.h"
#include "import.h"
#include "inline.h"
#include "inlineframes.h"
#include "internal.h"
#include "propsearch.h"
#include "protect.h"
#include "report.h"
#include "sort.h"

static inline unsigned imported_literal(kissat *solver, int elit) {
  const int erlit = kissat_external_representative(solver, elit);
  const unsigned eridx = ABS(erlit);
  const import *const import = &PEEK_STACK(solver->import, eridx);
  if (!import->imported || import->eliminated) {
    return INVALID_LIT;
  }
  unsigned ilit = import->lit;
  if (erlit < 0) {
    ilit = NOT(ilit);
  }
  return ilit;
}

void kissat_reset_assumptions(kissat *solver) {
  LOG("resetting assumptions");
  for (all_stack(int, elit, solver->assumptions)) {
    unsigned ilit = imported_literal(solver, elit);
    unsigned idx = IDX(ilit);
    flags *f = FLAGS(idx);
    f->assumed = 0;
    kissat_unprotect_variable(solver, idx);
  }
  CLEAR_STACK(solver->assumptions);
  for (all_stack(int, elit, solver->redundant_assumptions)) {
    unsigned ilit = imported_literal(solver, elit);
    if (ilit == INVALID_LIT) {
      continue;
    }
    unsigned idx = IDX(ilit);
    flags *f = FLAGS(idx);
    f->assumed = 0;
  }
  CLEAR_STACK(solver->redundant_assumptions);
#ifndef NDEBUG
  for (all_variables(idx)) {
    assert(!FLAGS(idx)->assumed);
  }
#endif
}

void kissat_assume_literal(kissat *solver, int elit, unsigned ilit) {
  assert(imported_literal(solver, elit) == ilit);
  if (solver->failed_early) {
    return;
  }

  if (!EMPTY_STACK(solver->failed)) {
    LOG("resetting previously failed assumptions");
    CLEAR_STACK(solver->failed);
  }

  if (solver->level) {
    kissat_backtrack_in_consistent_state(solver, 0);
  }

  unsigned idx = IDX(ilit);
  value assumed_value = ilit == LIT(idx) ? 1 : -1;
  flags *f = FLAGS(IDX(ilit));

  if (f->fixed) {
    if (VALUE(ilit) < 0) {
      LOG("assumption %d contradicits unit clause", elit);
      kissat_reset_assumptions(solver);
      solver->failed_early = true;
      PUSH_STACK(solver->failed, elit);
    }
    return;
  }

  if (f->assumed) {
    if (f->assumed != assumed_value) {
      int found_other = 0;
      unsigned not_ilit = NOT(ilit);
      for (all_stack(int, other_elit, solver->assumptions)) {
        unsigned other_ilit = imported_literal(solver, other_elit);
        if (other_ilit == not_ilit) {
          found_other = other_elit;
        }
      }
      assert(found_other);
      LOG("assumption %d contradicits assumption %d", elit, found_other);
      kissat_reset_assumptions(solver);
      solver->failed_early = true;
      if (found_other < elit) {
        SWAP(int, elit, found_other);
      }
      PUSH_STACK(solver->failed, elit);
      PUSH_STACK(solver->failed, found_other);
    }
    return;
  }

  f->assumed = assumed_value;
  kissat_protect_variable(solver, idx);
  PUSH_STACK(solver->assumptions, elit);
}

void analyze_failed_assumptions(kissat *solver, unsigned failing) {
  assert(LEVEL(failing));
  assert(EMPTY_STACK(solver->analyzed));
  assert(EMPTY_STACK(solver->levels));
  assert(EMPTY_STACK(solver->clause));

  assigned *all_assigned = solver->assigned;

  kissat_push_analyzed(solver, all_assigned, IDX(failing));
  unsigned unresolved = 1;

  unsigned const *t = END_ARRAY(solver->trail);
  unsigned uip = INVALID_LIT;
  assigned *a = 0;
  while (unresolved) {
    do {
      assert(t > BEGIN_ARRAY(solver->trail));
      uip = *--t;
      a = ASSIGNED(uip);
    } while (!a->analyzed);

    unresolved--;
    if (a->reason == DECISION_REASON) {
      unsigned assumption_idx = a->level - 1;
      int elit = PEEK_STACK(solver->assumptions, assumption_idx);
      LOG("found %s reason assumption %d", LOGLIT(uip), elit);
      PUSH_STACK(solver->failed, elit);
    } else if (a->binary) {
      const unsigned other = a->reason;
      LOGBINARY(uip, other, "resolving %s reason", LOGLIT(uip));
      if (LEVEL(other) && !ASSIGNED(other)->analyzed) {
        kissat_push_analyzed(solver, all_assigned, IDX(other));
        unresolved++;
      }
    } else {
      const reference ref = a->reason;
      LOGREF(ref, "resolving %s reason", LOGLIT(uip));
      clause *reason = kissat_dereference_clause(solver, ref);
      for (all_literals_in_clause(lit, reason)) {
        if (lit != uip && LEVEL(lit) && !ASSIGNED(lit)->analyzed) {
          kissat_push_analyzed(solver, all_assigned, IDX(lit));
          unresolved++;
        }
      }
    }
  }

  kissat_reset_only_analyzed_literals(solver);
}

int assign_assumption(kissat *solver) {
  while (true) {
    assert(kissat_assuming(solver));
    int elit = PEEK_STACK(solver->assumptions, solver->level);
    int erlit = kissat_external_representative(solver, elit);
    unsigned ilit = imported_literal(solver, erlit);
    value v = VALUE(ilit);

    if (v > 0) {
      LOG("redundant assumption %d", elit);
      int last_elit = POP_STACK(solver->assumptions);

      kissat_unprotect_variable(solver, IDX(ilit));
      PUSH_STACK(solver->redundant_assumptions, elit);

      if (!kissat_assuming(solver)) {
        if (!solver->unassigned) {
          return 10;
        }
        return -1;
      }
      PEEK_STACK(solver->assumptions, solver->level) = last_elit;
    } else if (v < 0) {
      LOG("failed assumption %d", elit);

      PUSH_STACK(solver->failed, elit);
      if (LEVEL(ilit)) {
        analyze_failed_assumptions(solver, ilit);

#define SMALLER(A, B) ((A) < (B))
        SORT_STACK(int, solver->failed, SMALLER);
#undef SMALLER
      }
      return 20;
    } else if (v == 0) {
      solver->level++;
      assert(solver->level != INVALID_LEVEL);
      kissat_push_frame(solver, ilit);
      assert(solver->level < SIZE_STACK(solver->frames));
      LOG("assume literal %s", LOGLIT(ilit));
      kissat_assign_decision(solver, ilit);
      return 0;
    }
  }
}

int kissat_assign_assumption(kissat *solver) {
  int res = assign_assumption(solver);
  if (res < 0) {
    kissat_decide(solver);
    return 0;
  }
  return res;
}

static void iterate(kissat *solver) {
  assert(solver->iterating);
  solver->iterating = false;
  REPORT(0, 'i');
}

int kissat_propagate_assumptions(kissat *solver) {
  if (solver->inconsistent || !EMPTY_STACK(solver->failed)) {
    return 20;
  }
  int res = 0;
  while (!res && kissat_assuming(solver)) {
    clause *conflict = kissat_search_propagate(solver);
    if (conflict) {
      res = kissat_analyze(solver, conflict);
    } else if (solver->iterating) {
      iterate(solver);
    } else {
      res = assign_assumption(solver);
    }
  }
  if (res < 0) {
    res = 0;
  }
  if (!res) {
    CLEAR_STACK(solver->assumption_implied);
    for (all_stack(unsigned, lit, solver->trail)) {
      unsigned l = LEVEL(lit);
      if (l && l <= SIZE_STACK(solver->assumptions)) {
        PUSH_STACK(solver->assumption_implied, lit);
      }
    }
  }
  return res;
}

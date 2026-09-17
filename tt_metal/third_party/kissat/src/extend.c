#include "colors.h"
#include "inline.h"
#include "import.h"

void kissat_undo_eliminated_assignment(kissat *solver) {
  size_t size_etrail = SIZE_STACK(solver->etrail);
#ifdef LOGGING
  size_t size_eliminated = SIZE_STACK(solver->eliminated);
#endif
  if (!size_etrail) {
    LOG("all %zu eliminated variables are unassigned", size_eliminated);
    return;
  }

  LOG("unassigning %zu eliminated variables %.0f%%",
        size_etrail, kissat_percent(size_etrail, size_eliminated));

  value *values = BEGIN_STACK(solver->eliminated);

  while (!EMPTY_STACK(solver->etrail)) {
    const unsigned pos = POP_STACK(solver->etrail);
    assert(pos < SIZE_STACK(solver->eliminated));
    assert(values[pos]);
    LOG2("unassigned eliminated[%u] external variable", pos);
    values[pos] = 0;
  }
}

static void extend_assign(kissat *solver, value *values, int lit) {
  assert(lit);
  assert(lit != INT_MIN);
  const unsigned idx = ABS(lit);
  import *import = &PEEK_STACK(solver->import, idx);
  assert(import->eliminated);
  assert(import->imported);
  const unsigned pos = import->lit;
  assert(pos < SIZE_STACK(solver->eliminated));
  const value value = lit < 0 ? -1 : 1;
  values[pos] = value;
  assert(kissat_value(solver, lit) == lit);
  LOG("assigned eliminated[%u] external literal %d", pos, value * idx);
  PUSH_STACK(solver->etrail, pos);
}

void kissat_extend(kissat *solver) {
  assert(!EMPTY_STACK(solver->extend));
  assert(!solver->extended);

  START(extend);

  kissat_undo_eliminated_assignment(solver);
  solver->extended = true;

  LOG("extending solution with reconstruction stack of size %zu",
        SIZE_STACK(solver->extend));

  value *evalues = BEGIN_STACK(solver->eliminated);
  value *ivalues = solver->values;

  const import *const imports = BEGIN_STACK(solver->import);

  const extension *const begin = BEGIN_STACK(solver->extend);
  extension const *p = END_STACK(solver->extend);

#ifdef LOGGING
  size_t assigned = 0;
  size_t flipped = 0;
#endif

  while (p != begin) {
    unsigned pos = UINT_MAX;
    bool satisfied = false;
    bool autarky = false;

    int eliminated = 0;
    int blocking = 0;

    size_t size = 0;
#ifndef LOGGING
    (void) size;
#endif
    do {
      size++;
      assert(begin < p);
      const extension ext = *--p;
      const int elit = ext.lit;
      if (ext.blocking) {
        blocking = elit;
      }

      if (autarky || satisfied) {
        continue;
      }

      if (ext.autarky) {
        autarky = true;
      }

      assert(elit != INT_MIN);

      int repr_elit = kissat_external_representative(solver, elit);
      const unsigned repr_eidx = ABS(repr_elit);
      const import *const import = imports + repr_eidx;
      assert(import->imported);

      if (import->eliminated) {
        const unsigned tmp = import->lit;
        assert(tmp < SIZE_STACK(solver->eliminated));
        value value = evalues[tmp];

        if (repr_elit < 0) {
          value = -value;
        }

        if (value > 0) {
          LOG2("previously assigned eliminated literal %d (= %d) "
                "satisfies clause", elit, repr_elit);
          satisfied = true;
        } else if (!value && (!eliminated || pos < tmp)) {
#ifdef LOGGING
          if (eliminated) {
            LOG2("earlier unassigned eliminated literal %d (= %d)",
                  elit, repr_elit);
          } else {
            LOG2("found unassigned eliminated literal %d (= %d)",
                  elit, repr_elit);
          }
#endif
          eliminated = repr_elit;
          pos = tmp;
        }
      } else {
        const unsigned ilit = import->lit;
        value value = ivalues[ilit];
        assert(value);

        if (repr_elit < 0) {
          value = -value;
        }

        if (value > 0) {
          LOG2("internal literal %s satisfies clause",
                LOGLIT(ilit));
          satisfied = true;
        }
      }
    } while (!blocking);

    if (autarky) {
      LOGEXT2(size, p, "autarky");
      continue;
    }

    if (satisfied) {
      LOGEXT2(size, p, "satisfied");
      continue;
    }

    if (eliminated && eliminated != blocking) {
      LOGEXT2(size, p,
            "assigning eliminated unassigned external literal %d "
            "to satisfy size %zu witness labelled clause at",
            eliminated, size);
      extend_assign(solver, evalues, eliminated);
#ifdef LOGGING
      assigned++;
#endif
      continue;
    }

#ifdef LOGGING
    const unsigned blocking_idx = ABS(blocking);
    assert(blocking_idx < SIZE_STACK(solver->import));
    assert(imports[blocking_idx].eliminated);
    assert(imports[blocking_idx].imported);
    const unsigned blocking_pos = imports[blocking_idx].lit;
    assert(blocking_pos < SIZE_STACK(solver->eliminated));
    const value blocking_value = evalues[blocking_pos];
    LOGEXT2(size, p,
          "%s blocking external literal %d "
          "to satisfy size %zu witness labelled clause at",
          blocking_value ? "flipping" : "assigning", blocking, size);
    if (blocking_value) {
      flipped++;
    } else {
      assigned++;
    }
#endif
    extend_assign(solver, evalues, blocking);
  }

#ifdef LOGGING
  size_t total = SIZE_STACK(solver->eliminated);
  LOG("assigned %zu external variables %.0f%% out of %zu eliminated",
        assigned, kissat_percent(assigned, total), total);
  LOG("flipped %zu external variables %.0f%% out of %zu assigned",
        flipped, kissat_percent(flipped, assigned), assigned);
  LOG("extended assignment complete");
#endif

  STOP(extend);
}

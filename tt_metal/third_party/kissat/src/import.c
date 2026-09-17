#include "import.h"
#include "inline.h"
#include "internal.h"
#include "logging.h"
#include "resize.h"

static void adjust_imports_for_external_literal(kissat *solver, unsigned eidx) {
  while (eidx >= SIZE_STACK(solver->import)) {
    import import;
    import.lit = 0;
    import.imported = false;
    import.eliminated = false;
    PUSH_STACK(solver->import, import);
  }
}

static void adjust_exports_for_external_literal(kissat *solver, unsigned eidx) {
  import *import = &PEEK_STACK(solver->import, eidx);
  unsigned iidx = solver->vars;
  kissat_enlarge_variables(solver, iidx + 1);
  unsigned ilit = 2 * iidx;
  assert(!import->imported);
  import->imported = true;
  assert(!import->eliminated);
  import->lit = ilit;
  LOG("importing external variable %u as internal literal %u", eidx, ilit);
  while (iidx >= SIZE_STACK(solver->export)) {
    PUSH_STACK(solver->export, 0);
  }
  POKE_STACK(solver->export, iidx, (int) eidx);
  LOG("exporting internal variable %u as external literal %u", iidx, eidx);
}

static inline unsigned import_literal(kissat *solver, int elit) {
  const unsigned eidx = ABS(elit);
  adjust_imports_for_external_literal(solver, eidx);

  const int repr_elit = kissat_external_representative(solver, elit);
  const unsigned repr_eidx = ABS(repr_elit);

  import *import = &PEEK_STACK(solver->import, repr_eidx);
  if (import->eliminated) {
    assert(import->imported);
    if (!GET_OPTION(incremental)) {
      return INVALID_LIT;
    }
    LOG("restoring external literal %d", elit);
    import->imported = false;
    import->eliminated = false;

    solver->restore = true;
  }
  unsigned ilit;
  if (!import->imported) {
    adjust_exports_for_external_literal(solver, repr_eidx);
  }
  assert(import->imported);
  ilit = import->lit;
  if (repr_elit < 0) {
    ilit = NOT(ilit);
  }
  assert(VALID_INTERNAL_LITERAL(ilit));
  return ilit;
}

unsigned kissat_import_literal(kissat *solver, int elit) {
  assert(VALID_EXTERNAL_LITERAL(elit));
  if (GET_OPTION(tumble)) {
    return import_literal(solver, elit);
  }
  const unsigned eidx = ABS(elit);
  assert(SIZE_STACK(solver->import) <= UINT_MAX);
  unsigned other = SIZE_STACK(solver->import);
  if (eidx < other) {
    return import_literal(solver, elit);
  }
  if (!other) {
    adjust_imports_for_external_literal(solver, other++);
  }

  unsigned ilit = 0;
  do {
    assert(VALID_EXTERNAL_LITERAL((int) other));
    ilit = import_literal(solver, other);
  } while (other++ < eidx);

  if (elit < 0 && ilit != INVALID_LIT) {
    ilit = NOT(ilit);
  }

  return ilit;
}

int kissat_external_representative(kissat *solver, int elit) {
  int initial_elit = elit;
  unsigned steps = 0;
  int target_elit = 0;
  do {
    const unsigned eidx = ABS(elit);
    const import *const current_import = &PEEK_STACK(solver->import, eidx);

    if (current_import->imported || !current_import->eliminated) {
      target_elit = elit;
      break;
    }

    steps++;

    bool flip = elit < 0;
    elit = KISSAT_GET_IMPORT_ELIT(current_import);
    if (flip) {
      elit = -elit;
    }
  } while (true);

  for (unsigned i = 1; i < steps; i++) {
    elit = initial_elit;
    unsigned eidx = ABS(elit);
    import *current_import = &PEEK_STACK(solver->import, eidx);

    assert(!(current_import->imported || !current_import->eliminated));

    bool flip = elit < 0;
    elit = KISSAT_GET_IMPORT_ELIT(current_import);
    if (flip) {
      elit = -elit;
    }
    int updated_elit = flip ? -target_elit : target_elit;
    KISSAT_SET_IMPORT_ELIT(current_import, updated_elit);
  }

  return target_elit;
}

#include "add.h"
#include "import.h"
#include "internal.h"
#include "logging.h"
#include "restore.h"

void kissat_restore_clauses(kissat *solver) {
  assert(GET_OPTION(incremental));
  assert(solver->restore);

  kissat_undo_eliminated_assignment(solver);

  extension *const begin = BEGIN_STACK(solver->extend);
  extension *const end = END_STACK(solver->extend);

  extension *write = begin;
  extension *read = begin;

  unsigned next_pos = 0;

  while (read != end) {
    const extension ext = *read++;
    assert(ext.blocking);

    if (ext.autarky) {
      extension *const autarky_begin = read - 1;

      while (read != end) {
        const extension ext = *read;
        if (ext.blocking) {
          break;
        }
        assert(ext.autarky);
        read++;
      }

      bool restore = false;

      for (extension *x = autarky_begin; x != read; x++) {
        const extension ext = *x;
        const int elit = ext.lit;
        const unsigned eidx = ABS(elit);

        import *const import = &PEEK_STACK(solver->import, eidx);

        if (import->eliminated) {
          assert(import->imported); // autarky lit should never be substituted
        } else {
          assert(import->imported);
          restore = true;
          break;
        }
      }

      if (restore) {
        LOG("restore autarky of size %u", (unsigned)(read - autarky_begin));
        for (extension *x = autarky_begin; x != read; x++) {
          const extension ext = *x;
          const int elit = ext.lit;
          LOG("autarky lit %d", elit);
          kissat_import_literal(solver, elit);
        }
      } else {
        for (extension *x = autarky_begin; x != read; x++) {
          *write++ = *x;
        }
      }

      continue;
    }

    const int elit = ext.lit;
    const unsigned eidx = ABS(elit);

    import *const import = &PEEK_STACK(solver->import, eidx);

    if (import->eliminated) {
      assert(import->imported); // the blocking lit should never be substituted

      unsigned pos = import->lit;
      if (pos >= next_pos) {
        import->lit = next_pos;
        next_pos++;
      }

      *write++ = ext;
      while (read != end) {
        const extension ext = *read;
        if (ext.blocking) {
          break;
        }
        read++;
        *write++ = ext;
      }
    } else {
      assert(import->imported);

      LOG("restore add blocking %d", elit);
      kissat_add_external_literal(solver, elit);

      while (read != end) {
        const extension ext = *read;
        if (ext.blocking) {
          break;
        }
        read++;
        LOG("restore add %d", ext.lit);
        kissat_add_external_literal(solver, ext.lit);
      }

      LOG("restore finish");
      kissat_finish_external_clause(solver, true);
    }
  }

  size_t new_size = write - begin;

  RESIZE_STACK(solver->extend, new_size);
  RESIZE_STACK(solver->eliminated, next_pos);
  solver->restore = false;
}

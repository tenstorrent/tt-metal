#include "allocate.h"
#include "backward.h"
#include "eliminate.h"
#include "inline.h"
#include "terminate.h"

static bool backward_subsume_lits(kissat *solver, reference ignore,
      unsigned size, unsigned *lits) {
  assert(size > 1);

  unsigned min_lit = INVALID_LIT;
  size_t min_occs = UINT_MAX;

  const unsigned *const end_lits = lits + size;

  for (const unsigned *p = lits; p != end_lits; p++) {
    const unsigned lit = *p;
    assert(!VALUE(lit));
    watches *watches = &WATCHES(lit);
    const size_t occs = SIZE_WATCHES(*watches);
    if (occs >= min_occs) {
      continue;
    }
    min_occs = occs;
    min_lit = lit;
  }

  assert(min_lit != INVALID_LIT);

  const unsigned occlim = solver->bounds.subsume.occurrences;
  if (min_occs > occlim) {
    return true;
  }

  LOGTMP("backward subsumption on %s with %zu occurrences in",
        LOGLIT(min_lit), min_occs);

  value *marks = solver->marks;
  const value *const values = solver->values;
  ward *const arena = BEGIN_STACK(solver->arena);

  const unsigned clslim = solver->bounds.subsume.clause_size;

  const unsigned match =
        (size > 2 ? INVALID_LIT : (lits[0] ^ lits[1] ^ min_lit));
  const unsigned not_match = (size > 2 ? INVALID_LIT : NOT(match));

  watches *watches = &WATCHES(min_lit);
  watch *begin_watches = BEGIN_WATCHES(*watches), *q = begin_watches;
  const watch *const end_watches = END_WATCHES(*watches), *p = q;

  bool found = false;
  bool marked = false;
  bool unit = false;

  assert(EMPTY_STACK(solver->delayed));
  assert(!values[min_lit]);

  bool terminated = false;

  while (p != end_watches) {
    terminated = TERMINATED(backward_terminated_1);
    if (terminated) {
      break;
    }
    const watch watch = *q++ = *p++;
    if (watch.type.binary) {
      assert(!watch.binary.redundant);
      if (size > 2) {
        continue;
      }
      const unsigned other = watch.binary.lit;
      if (other == not_match) {
        LOGBINARY(min_lit, match,
              "backward hyper unary resolution"
              "on %s first antecedent", LOGLIT(match));
        LOGBINARY(min_lit, not_match,
              "backward hyper unary resolution"
              "on %s second antecedent", LOGLIT(not_match));
        unit = true;
        break;
      }
      if (other != match) {
        continue;
      }
      if (found) {
        LOGBINARY(min_lit, match, "duplicated");
        INC(duplicated);
        kissat_disconnect_binary(solver, other, min_lit);
        kissat_delete_binary(solver, false, false, false, min_lit, other);
        kissat_update_after_removing_variable(solver, IDX(other));
        q--;
      } else {
        found = true;
      }
    } else {
      const reference ref = watch.large.ref;
      if (ref == ignore) {
        assert(size > 2);
        found = true;
        continue;
      }
      assert(ref < SIZE_STACK(solver->arena));
      struct clause *c = (struct clause *)(arena + ref);
      if (c->garbage) {
        continue;
      }
      if (c->size < size) {
        continue;
      }
      if (c->size > clslim) {
        continue;
      }
      INC(backward_checks);
      INC(subsumption_checks);
      if (!marked) {
        for (const unsigned *l = lits; l != end_lits; l++) {
          const unsigned lit = *l;
          assert(!values[lit]);
          marks[lit] |= 2;
        }
        marked = true;
      }
      unsigned limit = c->size - size;
      unsigned needed = size;
      unsigned remove = INVALID_LIT;
      for (all_literals_in_clause(lit, c)) {
        assert(needed);
        const value value = values[lit];
        if (value > 0) {
          LOGCLS(c, "satisfied by %s", LOGLIT(lit));
          kissat_mark_clause_as_garbage(solver, false, c);
          kissat_update_after_removing_clause(solver, c,
                INVALID_LIT);
          break;
        }
        if (value < 0) {
          if (!limit--) {
            break;
          }
          continue;
        }
        if (marks[lit] & 2) {
          assert(needed);
          if (!--needed) {
            break;
          }
        } else {
          const unsigned not_lit = NOT(lit);
          if (marks[not_lit] & 2) {
            if (remove != INVALID_LIT) {
              break;
            }
            remove = lit;
            if (!--needed) {
              break;
            }
          }
          if (!limit--) {
            break;
          }
        }
      }
      if (c->garbage) {
        q--;
        continue;
      }
      if (needed) {
        continue;
      }
      if (remove == INVALID_LIT) {
        LOGCLS(c, "backward subsumed");
        INC(subsumed);
        INC(backward_subsumed);
        kissat_mark_clause_as_garbage(solver, false, c);
        kissat_update_after_removing_clause(solver, c, INVALID_LIT);
        q--;
        continue;
      }
      bool satisfied = false;
      unsigned old_size = 0;
      unsigned other = INVALID_LIT;
      for (all_literals_in_clause(lit, c)) {
        const value value = values[lit];
        if (value > 0) {
          LOGCLS(c, "satisfied by %s", LOGLIT(lit));
          satisfied = true;
          break;
        } else if (!value) {
          if (lit != min_lit && lit != remove) {
            other = lit;
          }
          old_size++;
        }
      }
      if (satisfied) {
        kissat_mark_clause_as_garbage(solver, false, c);
        kissat_update_after_removing_clause(solver, c, INVALID_LIT);
        q--;
        continue;
      }
      assert(old_size <= c->size);

      INC(strengthened);
      INC(backward_strengthened);
      LOGCLS(c, "backward strengthening by removing %s in",
            LOGLIT(remove));
      assert(remove != min_lit);
      if (old_size == 2) {
        LOGTMP("backward large hyper unary resolution on %s"
              "first antecedent", LOGLIT(NOT(remove)));
        LOGCLS(c, "backward large hyper unary resolution on %s"
              "second antecedent", LOGLIT(remove));
        unit = true;
        break;
      }
      SHRINK_CLAUSE_IN_PROOF(c, remove, INVALID_LIT);
      CHECK_SHRINK_CLAUSE(c, remove, INVALID_LIT);
      if (old_size > 3) {
        unsigned *lits = c->lits;
        unsigned new_size = 0;
        for (unsigned i = 0; i < c->size; i++) {
          unsigned lit = lits[i];
          if (remove == lit) {
            continue;
          }
          const value value = values[lit];
          if (value < 0) {
            continue;
          }
          assert(!value);
          lits[new_size++] = lit;
          kissat_mark_added_literal(solver, remove);
        }
        assert(new_size == old_size - 1);
        assert(new_size > 2);
        if (!c->shrunken) {
          c->shrunken = true;
          lits[c->size - 1] = INVALID_LIT;
        }
        c->size = new_size;
        c->searched = 2;
        LOGCLS(c, "backward strengthened");
        kissat_disconnect_reference(solver, remove, ref);
      } else {
        assert(old_size == 3);
        assert(other != INVALID_LIT);
        assert(other != min_lit);
        assert(other != remove);
        PUSH_STACK(solver->delayed, other);
        LOGCLS(c, "garbage");
        assert(!c->garbage);
        assert(!c->hyper);
        const size_t bytes = kissat_actual_bytes_of_clause(c);
        ADD(arena_garbage, bytes);
        c->garbage = true;
        q--;
      }
      kissat_mark_removed_literal(solver, remove);
      kissat_update_after_removing_variable(solver, IDX(remove));
    }
  }

  while (p != end_watches) {
    *q++ = *p++;
  }

  SET_END_OF_WATCHES(*watches, q);

  if (unit) {
    LOG("backward strengthened unit clause %s", LOGLIT(min_lit));
    kissat_learned_unit(solver, min_lit);
  }

  for (all_stack(unsigned, other, solver->delayed)) {
    LOGBINARY(other, min_lit, "backward strengthened");
    kissat_watch_other(solver, false, false, other, min_lit);
    kissat_watch_other(solver, false, false, min_lit, other);
    kissat_mark_added_literal(solver, other);
    kissat_mark_added_literal(solver, min_lit);
  }
  CLEAR_STACK(solver->delayed);

  if (marked)
    for (const unsigned *l = lits; l != end_lits; l++) {
      marks[*l] &= ~2;
    }

  return !terminated;
}

bool kissat_backward_subsume_temporary(kissat *solver, reference ignore) {
  assert(!solver->watching);
  assert(GET_OPTION(backward));
  START(backward);
  unsigneds *clause = &solver->clause;
  const size_t size = SIZE_STACK(*clause);
  assert(size), assert(size <= UINT_MAX);
  unsigned *lits = BEGIN_STACK(*clause);
  bool res = backward_subsume_lits(solver, ignore, size, lits);
  STOP(backward);
  return res;
}

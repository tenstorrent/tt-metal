#include "eliminate.h"
#include "gates.h"
#include "inline.h"
#include "inlinescore.h"
#include "print.h"
#include "protect.h"
#include "resolve.h"

#include <inttypes.h>
#include <string.h>

static inline unsigned occurrences_literal(kissat *solver, unsigned lit,
      bool *update) {
  assert(!solver->watching);

  watches *watches = &WATCHES(lit);
#ifdef LOGGING
  const size_t size_watches = SIZE_WATCHES(*watches);
  LOG("literal %s has %zu watches", LOGLIT(lit), size_watches);
#endif
  const unsigned clslim = solver->bounds.eliminate.clause_size;

  watch *const begin = BEGIN_WATCHES(*watches), *q = begin;
  const watch *const end = END_WATCHES(*watches), *p = q;

  const value *const values = solver->values;
  ward *const arena = BEGIN_STACK(solver->arena);

  bool failed = false;
  unsigned res = 0;

  while (p != end) {
    const watch head = *q++ = *p++;
    if (head.type.binary) {
      const unsigned other = head.binary.lit;
      const value value = values[other];
      assert(value >= 0);
      if (value > 0) {
        kissat_eliminate_binary(solver, false, lit, other);
        q--;
      } else {
        res++;
      }
    } else {
      const reference ref = head.large.ref;
      assert(ref < SIZE_STACK(solver->arena));
      clause *const c = (struct clause *)(arena + ref);
      if (c->garbage) {
        q--;
      } else if (c->size > clslim) {
        LOG("literal %s watches too long clause of size %u",
              LOGLIT(lit), c->size);
        failed = true;
        break;
      } else {
        res++;
      }
    }
  }
  while (p != end) {
    *q++ = *p++;
  }
  SET_END_OF_WATCHES(*watches, q);
  if (failed) {
    return UINT_MAX;
  }
  if (q != end) {
    *update = true;
    LOG("literal %s actually occurs only %u times", LOGLIT(lit), res);
  }
  return res;
}

static inline clause *watch_to_clause(kissat *solver, ward *const arena,
      clause *const tmp, unsigned lit, watch watch) {
  clause *res;
  if (watch.type.binary) {
    const unsigned other = watch.binary.lit;
    tmp->lits[0] = lit;
    tmp->lits[1] = other;
    res = tmp;
  } else {
    const reference ref = watch.large.ref;
    assert(ref < SIZE_STACK(solver->arena));
    res = (struct clause *)(arena + ref);
  }
#ifdef NDEBUG
  (void) solver;
#endif
  return res;
}

static bool generate_resolvents(kissat *solver, unsigned lit,
      statches *const watches0, statches *const watches1,
      uint64_t *const resolved_ptr, uint64_t limit) {
  const unsigned not_lit = NOT(lit);
  unsigned resolved = *resolved_ptr;
  bool failed = false;

  clause tmp0, tmp1;
  memset(&tmp0, 0, sizeof tmp0);
  memset(&tmp1, 0, sizeof tmp1);
  tmp0.size = tmp1.size = 2;

  ward *const arena = BEGIN_STACK(solver->arena);
  const value *const values = solver->values;
  value *const marks = solver->marks;

  const unsigned clslim = solver->bounds.eliminate.clause_size;

  for (all_stack(watch, watch0, *watches0)) {
    clause *const c = watch_to_clause(solver, arena, &tmp0, lit, watch0);

    if (c->garbage) {
      assert(c != &tmp0);
      continue;
    }

    bool first_antecedent_satisfied = false;

    for (all_literals_in_clause(other, c)) {
      if (other == lit) {
        continue;
      }
      const value value = values[other];
      if (value < 0) {
        continue;
      }
      if (value > 0) {
        first_antecedent_satisfied = true;
        if (c != &tmp0) {
          kissat_eliminate_clause(solver, false, c, other);
        }
        break;
      }
    }

    if (first_antecedent_satisfied) {
      continue;
    }

    for (all_literals_in_clause(other, c)) {
      if (other == lit) {
        continue;
      }
      assert(!marks[other]);
      marks[other] = 1;
    }

    for (all_stack(watch, watch1, *watches1)) {
      clause *const d =
            watch_to_clause(solver, arena, &tmp1, not_lit, watch1);

      if (d->garbage) {
        assert(d != &tmp1);
        continue;
      }

      LOGCLS(c, "first %s antecedent", LOGLIT(lit));
      LOGCLS(d, "second %s antecedent", LOGLIT(not_lit));

      bool resolvent_satisfied_or_tautological = false;
      const size_t saved = SIZE_STACK(solver->resolvents);

      INC(eliminate_resolutions);

      for (all_literals_in_clause(other, d)) {
        if (other == not_lit) {
          continue;
        }
        const value value = values[other];
        if (value < 0) {
          LOG2("dropping falsified literal %s", LOGLIT(other));
          continue;
        }
        if (value > 0) {
          if (d != &tmp1) {
            kissat_eliminate_clause(solver, false, d, other);
          }
          resolvent_satisfied_or_tautological = true;
          break;
        }
        if (marks[other]) {
          LOG2("dropping repeated %s literal", LOGLIT(other));
          continue;
        }
        const unsigned not_other = NOT(other);
        if (marks[not_other]) {
          LOG("resolvent tautological on %s and %s "
                "with second %s antecedent",
                LOGLIT(NOT(other)), LOGLIT(other),
                LOGLIT(not_lit));
          resolvent_satisfied_or_tautological = true;
          break;
        }
        LOG2("including unassigned literal %s", LOGLIT(other));
        PUSH_STACK(solver->resolvents, other);
      }

      if (resolvent_satisfied_or_tautological) {
        RESIZE_STACK(solver->resolvents, saved);
        continue;
      }

      if (++resolved > limit) {
        LOG("limit of %" PRIu64 " resolvent exceeded", limit);
        failed = true;
        break;
      }

      for (all_literals_in_clause(other, c)) {
        if (other == lit) {
          continue;
        }
        const value value = values[other];
        assert(value <= 0);
        if (value < 0) {
          LOG2("dropping falsified literal %s", LOGLIT(other));
          continue;
        }
        PUSH_STACK(solver->resolvents, other);
      }

      size_t size_resolvent = SIZE_STACK(solver->resolvents) - saved;
      LOGLITS(size_resolvent,
            BEGIN_STACK(solver->resolvents) + saved, "resolvent");

      if (!size_resolvent) {
        assert(!solver->inconsistent);
        solver->inconsistent = true;
        LOG("resolved empty clause");
        CHECK_AND_ADD_EMPTY();
        ADD_EMPTY_TO_PROOF();
        failed = true;
        break;
      }

      if (size_resolvent == 1) {
        const unsigned unit = PEEK_STACK(solver->resolvents, saved);
        INC(eliminate_units);
        kissat_learned_unit(solver, unit);
        RESIZE_STACK(solver->resolvents, saved);
        if (marks[unit] <= 0) {
          continue;
        }
        LOGCLS(c, "first antecedent becomes satisfied");
        first_antecedent_satisfied = true;
        break;
      }

      if (size_resolvent > clslim) {
        LOG("resolvent size limit exceeded");
        failed = true;
        break;
      }

      PUSH_STACK(solver->resolvents, INVALID_LIT);
    }

    for (all_literals_in_clause(other, c)) {
      if (other == lit) {
        continue;
      }
      assert(marks[other] == 1);
      marks[other] = 0;
    }

    if (failed) {
      break;
    }
  }

  *resolved_ptr = resolved;

  return !failed;
}

bool kissat_generate_resolvents(kissat *solver, unsigned idx,
      unsigned *lit_ptr) {
  unsigned lit = LIT(idx);
  unsigned not_lit = NOT(lit);

  bool update = false;
  bool pure = false;
  bool swapped = false;
  bool protected = PROTECT(idx);
  uint64_t limit;

  solver->found_equivalence = INVALID_LIT;

  {
    unsigned pos_count = occurrences_literal(solver, lit, &update);
    unsigned neg_count = occurrences_literal(solver, not_lit, &update);

    if (pos_count > neg_count) {
      SWAP(unsigned, lit, not_lit);
      SWAP(size_t, pos_count, neg_count);
      swapped = true;
    }

    const unsigned occlim = solver->bounds.eliminate.occurrences;
    if (pos_count && neg_count > occlim) {
      LOG("no elimination of variable %u "
            "since its literal %s has %u > %u occurrences",
            idx, LOGLIT(not_lit), neg_count, occlim);
      return false;
    }

    limit = pos_count + (uint64_t) neg_count;
    if (pos_count) {
      const uint64_t bound = solver->bounds.eliminate.additional_clauses;
      limit += bound;
      LOG("trying to eliminate %s "
            "limit %" PRIu64 " bound %" PRIu64, LOGVAR(idx), limit, bound);
    } else {
      LOG("eliminating pure literal %s thus its variable %u",
            LOGLIT(lit), idx);
      pure = true;
    }
  }

  *lit_ptr = lit;

  INC(eliminate_attempted);
  if (pure) {
    return !protected;
  }

  const bool gates = !pure && kissat_find_gates(solver, lit, protected);

  if (swapped && solver->found_equivalence != INVALID_LIT) {
    solver->found_equivalence = NOT(solver->found_equivalence);
  }

  statches *const gates0 = &solver->gates[0];
  statches *const gates1 = &solver->gates[1];

  if (solver->values[lit]) {
    kissat_extremely_verbose(solver, "definition produced unit");
    CLEAR_STACK(*gates0);
    CLEAR_STACK(*gates1);
    return false;
  }

  bool failed = false;
  uint64_t resolved = 0;

  kissat_get_antecedents(solver, lit);
  statches *const antecedents0 = &solver->antecedents[0];
  statches *const antecedents1 = &solver->antecedents[1];

  if (gates) {
    LOG("resolving gates[0] against antecedents[1] clauses");
    if (!generate_resolvents(solver, lit,
                gates0, antecedents1, &resolved, limit)) {
      failed = true;
    } else {
      LOG("resolving gates[1] against antecedents[0] clauses");
      if (!generate_resolvents(solver, not_lit,
                  gates1, antecedents0, &resolved, limit)) {
        failed = true;
      } else if (solver->resolve_gate) {
        LOG("need to resolved gates[0] against gates[1] too");
        if (!generate_resolvents(solver, lit,
                    gates0, gates1, &resolved, limit)) {
          failed = true;
        }
      }
    }
  } else if (protected) {
    failed = true;
  } else {
    LOG("no gate extracted thus resolving all clauses");
    if (!generate_resolvents(solver, lit,
                antecedents0, antecedents1, &resolved, limit)) {
      failed = true;
    }
  }

  CLEAR_STACK(*antecedents0);
  CLEAR_STACK(*antecedents1);

  if (failed) {
    const unsigned idx = IDX(lit);
    LOG("elimination of %s failed", LOGVAR(idx));
    CLEAR_STACK(solver->resolvents);
    if (update) {
      kissat_update_variable_score(solver, &solver->schedule, idx);
    }
  }

  LOG("resolved %" PRIu64 " resolvents", resolved);

  CLEAR_STACK(*gates0);
  CLEAR_STACK(*gates1);

  return !failed;
}

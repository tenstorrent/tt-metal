#include "internal.h"
#include "logging.h"

static inline value move_smallest_literal_to_front(kissat *solver,
      const value *const values,
      const assigned *const assigned,
      bool satisfied_is_enough,
      unsigned start, unsigned size, unsigned *lits) {
  assert(1 < size);
  assert(start < size);

  unsigned a = lits[start];

  value u = values[a];
  if (!u || (u > 0 && satisfied_is_enough)) {
    return u;
  }

  unsigned pos = 0, best = a;

  {
    const unsigned i = IDX(a);
    unsigned k = (u ? assigned[i].level : UINT_MAX);

    assert(start < UINT_MAX);
    for (unsigned i = start + 1; i < size; i++) {
      const unsigned b = lits[i];
      const value v = values[b];

      if (!v || (v > 0 && satisfied_is_enough)) {
        best = b;
        pos = i;
        u = v;
        break;
      }

      const unsigned j = IDX(b);
      const unsigned l = (v ? assigned[j].level : UINT_MAX);

      bool better;

      if (u < 0 && v > 0) {
        better = true;
      } else if (u > 0 && v < 0) {
        better = false;
      } else if (u < 0) {
        assert(v < 0);
        better = (k < l);
      } else {
        assert(u > 0);
        assert(v > 0);
        assert(!satisfied_is_enough);
        better = (k > l);
      }

      if (!better) {
        continue;
      }

      best = b;
      pos = i;
      u = v;
      k = l;
    }
  }

  if (!pos) {
    return u;
  }

  lits[start] = best;
  lits[pos] = a;

  LOG("new smallest literal %s at %u swapped with %s at %u",
        LOGLIT(best), pos, LOGLIT(a), start);
#ifndef LOGGING
  (void) solver;
#endif
  return u;
}

#ifdef INLINE_SORT
static inline
#endif
void
kissat_sort_literals(kissat *solver,
#ifdef INLINE_SORT
      const value *const values, const assigned *assigned,
#endif
      unsigned size, unsigned *lits) {
#ifndef INLINE_SORT
  const value *const values = solver->values;
  const assigned *const assigned = solver->assigned;
#endif
  value u = move_smallest_literal_to_front(solver, values, assigned,
              false, 0, size, lits);
  if (size > 2)
    move_smallest_literal_to_front(solver, values, assigned,
          (u >= 0), 1, size, lits);
}

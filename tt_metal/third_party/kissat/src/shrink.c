#include "allocate.h"
#include "inline.h"
#include "inlinereap.h"
#include "minimize.h"
#include "shrink.h"

static void reset_shrinkable(kissat *solver) {
  size_t reset = 0;
  while (!EMPTY_STACK(solver->shrinkable)) {
    const unsigned idx = POP_STACK(solver->shrinkable);
    assigned *a = solver->assigned + idx;
    assert(a->shrinkable);
    a->shrinkable = false;
    reset++;
  }
  LOG("resetting %zu shrinkable variables", reset);
}

static void mark_shrinkable_as_removable(kissat *solver) {
  size_t marked = 0, reset = 0;
  struct assigned *assigned = solver->assigned;
  while (!EMPTY_STACK(solver->shrinkable)) {
    const unsigned idx = POP_STACK(solver->shrinkable);
    struct assigned *a = assigned + idx;
    assert(a->shrinkable);
    a->shrinkable = false;
    assert(!a->poisoned);
    reset++;
    if (a->removable) {
      continue;
    }
    kissat_push_removable(solver, assigned, idx), marked++;
  }
  LOG("resetting %zu shrinkable variables", reset);
  LOG("marked %zu removable variables", marked);
}

static inline int shrink_literal(kissat *solver, assigned *assigned, reap *reap,
      unsigned level, unsigned max_trail, unsigned lit) {
  assert(solver->assigned == assigned);
  assert(!reap || &solver->reap == reap);
  assert(VALUE(lit) < 0);

  const unsigned idx = IDX(lit);
  struct assigned *a = assigned + idx;
  assert(a->level <= level);
  if (!a->level) {
    LOG2("skipping root level assigned %s", LOGLIT(lit));
    return 0;
  }
  if (a->shrinkable) {
    LOG2("skipping already shrinkable literal %s", LOGLIT(lit));
    return 0;
  }
  if (a->level < level) {
    if (a->removable) {
      LOG2("skipping removable thus shrinkable %s", LOGLIT(lit));
      return 0;
    }
    const bool always_minimize_on_lower_level = (GET_OPTION(shrink) > 2);
    if (always_minimize_on_lower_level &&
          kissat_minimize_literal(solver, lit, false)) {
      LOG2("minimized thus shrinkable %s", LOGLIT(lit));
      return 0;
    }
    LOG("literal %s on lower level %u < %u not removable/shrinkable",
          LOGLIT(lit), a->level, level);
    return -1;
  }
  LOG2("marking %s as shrinkable", LOGLIT(lit));
  a->shrinkable = true;
  PUSH_STACK(solver->shrinkable, idx);
  if (reap) {
    assert(max_trail >= a->trail);
    const unsigned dist = max_trail - a->trail;
    kissat_push_reap(solver, reap, dist);
  }
  return 1;
}

static inline unsigned shrunken_block(kissat *solver, unsigned level,
      unsigned *begin_block, unsigned *end_block, unsigned uip) {
  assert(uip != INVALID_LIT);
  const unsigned not_uip = NOT(uip);
  LOG("found unique implication point %s on level %u", LOGLIT(uip), level);

  assert(begin_block < end_block);
#if defined (LOGGING) || !defined (NDEBUG)
  const size_t tmp = end_block - begin_block;
  LOG("shrinking %zu literals on level %u to single literal %s",
        tmp, level, LOGLIT(not_uip));
  assert(tmp > 1);
#endif

#ifdef LOGGING
  bool not_uip_was_in_clause = false;
#endif
  unsigned block_shrunken = 0;

  for (unsigned *p = begin_block; p != end_block; p++) {
    const unsigned lit = *p;
    if (lit == INVALID_LIT) {
      continue;
    }
#ifdef LOGGING
    if (lit == not_uip) {
      not_uip_was_in_clause = true;
    } else {
      LOG("shrunken literal %s", LOGLIT(lit));
    }
#endif
    *p = INVALID_LIT;
    block_shrunken++;
  }
  *begin_block = not_uip;
  assert(block_shrunken);
  block_shrunken--;
#ifdef LOGGING
  if (not_uip_was_in_clause) {
    LOG("keeping single literal %s on level %u", LOGLIT(not_uip), level);
  } else
    LOG("shrunken all literals on level %u and added %s instead",
          level, LOGLIT(not_uip));
#endif
  const unsigned uip_idx = IDX(uip);
  assigned *assigned = solver->assigned;
  struct assigned *a = assigned + uip_idx;
  if (!a->analyzed) {
    kissat_push_analyzed(solver, assigned, uip_idx);
  }

  mark_shrinkable_as_removable(solver);
#ifndef LOGGING
  (void) level;
#endif
  return block_shrunken;
}

static inline void push_literals_of_block(kissat *solver, assigned *assigned,
      reap *reap,
      unsigned *begin_block, unsigned *end_block,
      unsigned level, unsigned max_trail) {
  assert(!reap || reap == &solver->reap);
  assert(!reap || kissat_empty_reap(reap));

  assert(assigned == solver->assigned);

  for (const unsigned *p = begin_block; p != end_block; p++) {
    const unsigned lit = *p;
    if (lit == INVALID_LIT) {
      continue;
    }
#ifndef NDEBUG
    int tmp =
#endif
          shrink_literal(solver, assigned, reap,
                level, max_trail, lit);
    assert(tmp > 0);
  }
}

static inline unsigned shrink_next(kissat *solver, reap *reap,
      const unsigned *const begin_trail,
      const unsigned *const end_trail, unsigned max_trail) {
  assert(!kissat_empty_reap(reap));
  const unsigned dist = kissat_pop_reap(solver, reap);
  const unsigned pos = max_trail - dist;
  const unsigned *const t = begin_trail + pos;
  assert(begin_trail <= t), assert(t < end_trail);
  const unsigned uip = *t;
  assert(VALUE(uip) > 0);
  LOG("trying to shrink literal %s at trail[%u]", LOGLIT(uip), pos);
#ifdef NDEBUG
  (void) end_trail;
#endif
  return uip;
}

static inline unsigned shrink_along_binary(kissat *solver, assigned *assigned,
      reap *reap,
      unsigned level, unsigned max_trail,
      unsigned uip, unsigned other) {
  assert(VALUE(other) < 0);
  LOGBINARY2(uip, other, "shrinking along %s reason", LOGLIT(uip));
  int tmp = shrink_literal(solver, assigned, reap,
              level, max_trail, other);
#ifndef LOGGING
  (void) uip;
#endif
  return tmp > 0;
}

static inline unsigned shrink_along_large(kissat *solver, assigned *assigned,
      reap *reap,
      unsigned level, unsigned max_trail,
      unsigned uip, reference ref, bool *failed_ptr) {
  unsigned open = 0;
  LOGREF2(ref, "shrinking along %s reason", LOGLIT(uip));
  clause *c = kissat_dereference_clause(solver, ref);
  if (GET_OPTION(minimizeticks)) {
    INC(search_ticks);
  }
  for (all_literals_in_clause(other, c)) {
    if (other == uip) {
      continue;
    }
    assert(VALUE(other) < 0);
    int tmp = shrink_literal(solver, assigned, reap,
                level, max_trail, other);
    if (tmp < 0) {
      *failed_ptr = true;
      break;
    }
    if (tmp > 0) {
      open++;
    }
  }
  return open;
}

static inline unsigned shrink_along_reason(kissat *solver, assigned *assigned,
      reap *reap,
      unsigned level, unsigned uip, unsigned max_trail,
      bool resolve_large_clauses, bool *failed_ptr) {
  unsigned open = 0;
  const unsigned uip_idx = IDX(uip);
  struct assigned *a = assigned + uip_idx;
  assert(a->shrinkable);
  assert(a->level == level);
  assert(a->reason != DECISION_REASON);
  if (a->binary) {
    const unsigned other = a->reason;
    open = shrink_along_binary(solver, assigned, reap, level, max_trail,
                uip, other);
  } else {
    reference ref = a->reason;
    if (resolve_large_clauses)
      open = shrink_along_large(solver, assigned, reap, level, max_trail,
                  uip, ref, failed_ptr);
    else {
      LOGREF(ref, "not shrinking %s reason", LOGLIT(uip));
      *failed_ptr = true;
    }
  }
  return open;
}

static inline unsigned shrink_block(kissat *solver,
      unsigned *begin_block, unsigned *end_block,
      unsigned level, unsigned max_trail) {
  assert(level < solver->level);

  unsigned open = end_block - begin_block;

  LOG("trying to shrink %u literals on level %u", open, level);
  LOG("maximum trail position %u on level %u", max_trail, level);

  assigned *assigned = solver->assigned;
  reap *reap = GET_OPTION(reap) ? &solver->reap : 0;

  push_literals_of_block(solver, assigned, reap,
        begin_block, end_block, level, max_trail);

  assert(!reap || kissat_size_reap(reap) == open);
  assert(SIZE_STACK(solver->shrinkable) == open);

  const unsigned *const begin_trail = BEGIN_ARRAY(solver->trail);
  const unsigned *const end_trail = END_ARRAY(solver->trail);

  const bool resolve_large_clauses = (GET_OPTION(shrink) > 1);
  unsigned uip = INVALID_LIT;
  bool failed = false;

  const unsigned *t = reap ? 0 : begin_trail + max_trail;

  while (!failed) {
    if (reap) {
      uip = shrink_next(solver, reap, begin_trail, end_trail, max_trail);
    } else {
      do {
        assert(begin_trail <= t), uip = *t--;
      } while (!assigned[IDX(uip)].shrinkable);
    }
    if (open == 1) {
      break;
    }
    open += shrink_along_reason(solver, assigned, reap,
                level, uip, max_trail,
                resolve_large_clauses, &failed);
    assert(open > 1);
    open--;
  }

  unsigned block_shrunken = 0;
  if (failed) {
    reset_shrinkable(solver);
  } else
    block_shrunken =
          shrunken_block(solver, level, begin_block, end_block, uip);

  if (reap) {
    kissat_clear_reap(solver, reap);
  }

  return block_shrunken;
}

static unsigned *next_block(kissat *solver, unsigned *begin_lits,
      unsigned *end_block,
      unsigned *level_ptr, unsigned *max_trail_ptr) {
  assigned *assigned = solver->assigned;

  unsigned level = INVALID_LEVEL;
  unsigned max_trail = 0;

  unsigned *begin_block = end_block;

  while (begin_lits < begin_block) {
    const unsigned lit = begin_block[-1];
    assert(lit != INVALID_LIT);
    const unsigned idx = IDX(lit);
    struct assigned *a = assigned + idx;
    unsigned lit_level = a->level;
    if (level == INVALID_LEVEL) {
      level = lit_level;
      LOG("starting to shrink level %u", level);
    } else {
      assert(lit_level >= level);
      if (lit_level > level) {
        break;
      }
    }
    begin_block--;
    const unsigned trail = a->trail;
    if (trail > max_trail) {
      max_trail = trail;
    }
  }

  *level_ptr = level;
  *max_trail_ptr = max_trail;

  return begin_block;
}

static unsigned minimize_block(kissat *solver, unsigned *begin_block,
      unsigned *end_block) {
  if (!GET_OPTION(shrinkminimize)) {
    return 0;
  }

  unsigned minimized = 0;

  for (unsigned *p = begin_block; p != end_block; p++) {
    const unsigned lit = *p;
    assert(lit != INVALID_LIT);
    if (!kissat_minimize_literal(solver, lit, true)) {
      continue;
    }
    LOG("minimize-shrunken literal %s", LOGLIT(lit));
    *p = INVALID_LIT;
    minimized++;
  }

  return minimized;
}

static inline unsigned *minimize_and_shrink_block(kissat *solver,
      unsigned *begin_lits, unsigned *end_block,
      unsigned *total_shrunken,
      unsigned *total_minimized) {
  assert(EMPTY_STACK(solver->shrinkable));

  unsigned level, max_trail;

  unsigned *begin_block = next_block(solver, begin_lits, end_block,
              &level, &max_trail);

  unsigned open = end_block - begin_block;
  assert(open > 0);

  unsigned block_shrunken = 0;
  unsigned block_minimized = 0;

  if (open < 2) {
    LOG("only one literal on level %u", level);
  } else {
    block_shrunken = shrink_block(solver, begin_block, end_block,
                level, max_trail);
    if (!block_shrunken) {
      block_minimized = minimize_block(solver, begin_block, end_block);
    }
  }

  block_shrunken += block_minimized;
  LOG("shrunken %u literals on level %u (including %u minimized)",
        block_shrunken, level, block_minimized);

  *total_minimized += block_minimized;
  *total_shrunken += block_shrunken;

  return begin_block;
}

void kissat_shrink_clause(kissat *solver) {
  assert(GET_OPTION(minimize) > 0);
  assert(GET_OPTION(shrink) > 0);
  assert(!EMPTY_STACK(solver->clause));

  START(shrink);

  unsigned total_shrunken = 0;
  unsigned total_minimized = 0;

  unsigned *begin_lits = BEGIN_STACK(solver->clause);
  unsigned *end_lits = END_STACK(solver->clause);

  unsigned *end_block = END_STACK(solver->clause);

  while (end_block != begin_lits)
    end_block = minimize_and_shrink_block(solver, begin_lits, end_block,
                &total_shrunken, &total_minimized);
  unsigned *q = begin_lits;
  for (const unsigned *p = q; p != end_lits; p++) {
    const unsigned lit = *p;
    if (lit != INVALID_LIT) {
      *q++ = lit;
    }
  }
  LOG("clause shrunken by %u literals (including %u minimized)",
        total_shrunken, total_minimized);
  assert(q + total_shrunken == end_lits);
  SET_END_OF_STACK(solver->clause, q);
  ADD(literals_shrunken, total_shrunken);
  ADD(literals_minimize_shrunken, total_minimized);

  LOGTMP("shrunken learned");
  kissat_reset_poisoned(solver);

  STOP(shrink);
}

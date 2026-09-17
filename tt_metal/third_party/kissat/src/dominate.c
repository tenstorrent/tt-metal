#include "dominate.h"
#include "inline.h"

unsigned kissat_find_dominator(kissat *solver, unsigned lit, clause *c) {
  assert(solver->watching);
  assert(solver->level == 1);

  LOGCLS(c, "starting to find dominator of %s from", LOGLIT(lit));

  ward *const arena = BEGIN_STACK(solver->arena);
  assigned *assigned = solver->assigned;

  unsigned count = 0;
  unsigned leaf = INVALID_LIT;

  for (all_literals_in_clause(other, c)) {
    if (other == lit) {
      continue;
    }
    assert(VALUE(other) < 0);
    const unsigned other_idx = IDX(other);
    if (!assigned[other_idx].level) {
      continue;
    }
    if (!count++) {
      leaf = other;
    }
  }
  assert(count);
  if (count < 2) {
    LOGCLS(c, "essentially binary");
    return INVALID_LIT;
  }

  START(dominate);

  unsigneds *analyzed = &solver->analyzed;

  assert(EMPTY_STACK(*analyzed));
  assert(leaf != INVALID_LIT);
  const unsigned leaf_idx = IDX(leaf);
  struct assigned *a = assigned + leaf_idx;
  a->analyzed = true;

  PUSH_STACK(*analyzed, leaf);

  LOG("starting to mark implication chain at %s", LOGLIT(leaf));

  unsigned root = leaf;

  for (;;) {
    assert(a == ASSIGNED(root));
    if (a->reason == DECISION_REASON) {
      break;
    }
    unsigned prev = INVALID_LIT;
    if (a->binary) {
      prev = a->reason;
      LOGBINARY(root, prev, "following %s reason", LOGLIT(root));
    } else {
      const reference ref = a->reason;
      LOGREF(ref, "following %s reason", LOGLIT(root));
      clause *reason = (clause *)(arena + ref);
      assert(kissat_clause_in_arena(solver, reason));
      for (all_literals_in_clause(other, reason)) {
        if (other == NOT(root)) {
          continue;
        }
        assert(VALUE(other) < 0);
        const unsigned other_idx = IDX(other);
        if (!assigned[other_idx].level) {
          continue;
        }
        if (prev != INVALID_LIT) {
          assert(!solver->probing);
          LOGCLS(reason, "early abort due to");
          prev = INVALID_LIT;
          break;
        }
        prev = other;
      }
    }
    if (prev == INVALID_LIT) {
      break;
    }
    assert(VALUE(prev) < 0);
    LOG("marking implied %s", LOGLIT(prev));
    const unsigned prev_idx = IDX(prev);
    a = assigned + prev_idx;
    assert(!a->analyzed);
    a->analyzed = true;
    PUSH_STACK(*analyzed, prev);
    root = prev;
  }
  LOG("root %s of implication chain", LOGLIT(root));

  unsigned unmarked = 0;
  for (all_literals_in_clause(start, c)) {
    if (start == lit) {
      continue;
    }
    if (start == leaf) {
      continue;
    }
    assert(VALUE(start) < 0);
    const unsigned start_idx = IDX(start);
    if (!assigned[start_idx].level) {
      continue;
    }
    LOG("starting next common dominator search at %s", LOGLIT(start));
    unsigned dom = start;
    const unsigned dom_idx = IDX(dom);
    a = assigned + dom_idx;
    while (!a->analyzed) {
      assert(a == ASSIGNED(dom));
      if (a->reason == DECISION_REASON) {
        break;
      }
      unsigned prev = INVALID_LIT;
      if (a->binary) {
        prev = a->reason;
        LOGBINARY(root, prev, "following %s reason", LOGLIT(root));
      } else {
        const reference ref = a->reason;
        LOGREF(ref, "following %s reason", LOGLIT(root));
        clause *reason = kissat_dereference_clause(solver, ref);
        for (all_literals_in_clause(other, reason)) {
          if (other == NOT(dom)) {
            continue;
          }
          assert(VALUE(other) < 0);
          const unsigned other_idx = IDX(other);
          if (!assigned[other_idx].level) {
            continue;
          }
          if (prev != INVALID_LIT) {
            LOGCLS(reason, "early abort due to");
            prev = INVALID_LIT;
            break;
          }
          prev = other;
        }
      }
      if (prev == INVALID_LIT) {
        break;
      }
      assert(VALUE(prev) < 0);
      const unsigned prev_idx = IDX(prev);
      a = assigned + prev_idx;
      dom = prev;
    }
    LOG("new common dominator %s of %s", LOGLIT(dom), LOGLIT(start));
    while (unmarked < SIZE_STACK(*analyzed)) {
      const unsigned other = PEEK_STACK(*analyzed, unmarked);
      if (other == dom) {
        break;
      }
      const unsigned other_idx = IDX(other);
      a = assigned + other_idx;
      assert(a->analyzed);
      a->analyzed = false;
      unmarked++;
    }
    if (unmarked == SIZE_STACK(*analyzed)) {
      LOG("all analyzed literals unmarked due to early abort");
      break;
    }
  }

  unsigned res = INVALID_LIT;
  while (unmarked < SIZE_STACK(*analyzed)) {
    const unsigned other = PEEK_STACK(*analyzed, unmarked);
    if (res == INVALID_LIT) {
      res = other;
    }
    const unsigned other_idx = IDX(other);
    a = assigned + other_idx;
    assert(a->analyzed);
    a->analyzed = false;
    unmarked++;
  }
  CLEAR_STACK(*analyzed);

#ifdef LOGGING
  if (res == INVALID_LIT) {
    LOG("no dominator found");
  } else {
    LOG("found dominator %s", LOGLIT(res));
  }
#endif
  STOP(dominate);

  return res;
}

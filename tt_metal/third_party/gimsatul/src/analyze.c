#include "analyze.h"

#include <string.h>

#include "assign.h"
#include "backtrack.h"
#include "bump.h"
#include "export.h"
#include "macros.h"
#include "minimize.h"
#include "promote.h"
#include "reduce.h"
#include "ring.h"
#include "sort.h"
#include "trace.h"
#include "utilities.h"

static void bump_reason (struct ring *ring, struct watcher *watcher) {
  assert (watcher->redundant);
  watcher->used = MAX_USED;
  unsigned new_glue = recompute_glue (ring, watcher);
  if (new_glue < watcher->glue)
    promote_watcher (ring, watcher, new_glue);
  else
    new_glue = watcher->glue;
  assert (watcher->glue);
  assert (watcher->glue <= MAX_GLUE);
  unsigned stable = ring->stable;
  ring->statistics.usage[stable].glue[new_glue]++;
  ring->statistics.usage[stable].bumped++;
  ring->statistics.bumped++;
}

static bool analyze_reason_side_literal (struct ring *ring, unsigned lit) {
  unsigned idx = IDX (lit);
  struct variable *v = ring->variables + idx;
  if (!v->level)
    return false;
  if (v->seen)
    return false;
  v->seen = true;
  PUSH (ring->analyzed, idx);
  return true;
}

static void analyze_reason_side_literals (struct ring *ring) {
  if (!ring->options.bump_reasons)
    return;

  uint64_t *count = &ring->delay.bump_reason.count;
  if (*count) {
    *count -= 1;
    return;
  }

  if (ring->averages[ring->stable].decisions.value > 10)
    return;

  size_t original = SIZE (ring->analyzed);
  size_t limit = 10 * original;
  uint64_t ticks = 0;

  for (all_elements_on_stack (unsigned, lit, ring->clause)) {
    struct variable *v = VAR (lit);
    if (!v->level)
      continue;
    struct watch *reason = v->reason;
    if (!reason)
      continue;
    assert (v->seen || v->shrinkable);
    if (is_binary_pointer (reason)) {
      assert (NOT (lit) == lit_pointer (reason));
      if (analyze_reason_side_literal (ring, other_pointer (reason)) &&
          SIZE (ring->analyzed) > limit)
        break;
    } else {
      const unsigned not_lit = NOT (lit);
      struct watcher *watcher = get_watcher (ring, reason);
      ticks++;
      for (all_watcher_literals (other, watcher))
        if (other != not_lit && analyze_reason_side_literal (ring, other) &&
            SIZE (ring->analyzed) > limit)
          break;
    }
  }

  ring->statistics.contexts[ring->context].ticks += ticks;

  uint64_t *current = &ring->delay.bump_reason.current;

  if (SIZE (ring->analyzed) > limit) {
    while (SIZE (ring->analyzed) > original)
      ring->variables[POP (ring->analyzed)].seen = false;
    *current += 1;
  } else if (*current)
    *current /= 2;

  *count = *current;
}

static bool larger_trail_position (unsigned *pos, unsigned a, unsigned b) {
  unsigned i = IDX (a);
  unsigned j = IDX (b);
  return pos[i] > pos[j];
}

#define LARGER_TRAIL_POS(A, B) larger_trail_position (pos, (A), (B))

static void sort_deduced_clause (struct ring *ring) {
  LOGTMP ("clause before sorting");
  unsigned *pos = ring->trail.pos;
  SORT_STACK (unsigned, ring->clause, LARGER_TRAIL_POS);
  LOGTMP ("clause after sorting");
}

void clear_analyzed (struct ring *ring) {
  struct unsigneds *analyzed = &ring->analyzed;
  struct variable *variables = ring->variables;
  for (all_elements_on_stack (unsigned, idx, *analyzed)) {
    struct variable *v = variables + idx;
    assert (v->seen);
    v->seen = false;
  }
  CLEAR (*analyzed);

  struct unsigneds *levels = &ring->levels;
  unsigned char *used = ring->used;
  for (all_elements_on_stack (unsigned, used_level, *levels))
    used[used_level] = 0;
  CLEAR (*levels);
}

static void update_decision_rate (struct ring *ring) {
  uint64_t current = SEARCH_DECISIONS;
  uint64_t previous = ring->last.decisions;
  assert (current >= previous);
  uint64_t delta = current - previous;
  struct averages *a = ring->averages + ring->stable;
  update_average (ring, &a->decisions, "decision rate", SLOW_ALPHA, delta);
  ring->last.decisions = current;
}

static void update_tier_limits (struct ring *ring) {
  if (!ring->intervals.tiers)
    ring->intervals.tiers = 4;
  else if (ring->intervals.tiers < (1u << 16))
    ring->intervals.tiers *= 2;
  recalculate_tier_limits (ring);
  ring->limits.tiers = SEARCH_CONFLICTS + ring->intervals.tiers;
}

static void flush_last_learned (struct ring *ring) {
  unsigned *q = ring->last_learned, *p = q;
  unsigned *end = q + ring->options.eagerly_subsume;
  while (p != end) {
    unsigned idx = *p++;
    if (idx != INVALID)
      *q++ = idx;
  }
  while (q != end)
    *q++ = INVALID;
}

static void eagerly_subsume_last_learned (struct ring *ring) {
  signed char *marks = ring->marks;
  for (all_elements_on_stack (unsigned, lit, ring->clause)) {
    assert (!marks[lit]);
    marks[lit] = 1;
  }
  unsigned clause_size = SIZE (ring->clause);
  unsigned *p = ring->last_learned;
  unsigned *end = p + ring->options.eagerly_subsume;
  while (p != end) {
    unsigned idx = *p++;
    if (idx == INVALID)
      continue;
    struct watcher *watcher = index_to_watcher (ring, idx);
    if (watcher->garbage || watcher->clause->garbage) {
      p[-1] = INVALID;
      continue;
    }
    unsigned watcher_size = watcher->size;
    if (!watcher_size)
      watcher_size = watcher->clause->size;
    if (watcher_size <= clause_size)
      continue;
    LOGCLAUSE (watcher->clause, "trying to eagerly subsume");
    unsigned needed = clause_size;
    unsigned remain = watcher_size;
    for (all_watcher_literals (lit, watcher)) {
      if (marks[lit] && !--needed)
        break;
      else if (--remain < needed)
        break;
    }
    if (needed)
      continue;
    LOGCLAUSE (watcher->clause, "eagerly subsumed");
    mark_garbage_watcher (ring, watcher);
    p[-1] = INVALID;
    ring->statistics.eagerly_subsumed++;
  }
  for (all_elements_on_stack (unsigned, lit, ring->clause)) {
    assert (marks[lit]);
    marks[lit] = 0;
  }
  flush_last_learned (ring);
}

static void insert_last_learned (struct ring *ring, struct watch *learned) {
  unsigned prev = index_pointer (learned);
  unsigned *end = ring->last_learned + ring->options.eagerly_subsume;
  for (unsigned *p = ring->last_learned; p != end; p++) {
    unsigned tmp = *p;
    *p = prev;
    prev = tmp;
  }
}

#define RESOLVE_LITERAL(OTHER) \
  do { \
    if (OTHER == uip) \
      break; \
    assert (ring->values[OTHER] < 0); \
    unsigned OTHER_IDX = IDX (OTHER); \
    struct variable *V = variables + OTHER_IDX; \
    unsigned OTHER_LEVEL = V->level; \
    assert (OTHER_LEVEL <= conflict_level); \
    if (!OTHER_LEVEL) \
      break; \
    if (V->seen) \
      break; \
    V->seen = true; \
    PUSH (*analyzed, OTHER_IDX); \
    if (OTHER_LEVEL == conflict_level) { \
      open++; \
      break; \
    } \
    PUSH (*ring_clause, OTHER); \
    if (!used[OTHER_LEVEL]) { \
      glue++; \
      used[OTHER_LEVEL] = 1; \
      PUSH (*levels, OTHER_LEVEL); \
      if (OTHER_LEVEL > jump) \
        jump = OTHER_LEVEL; \
    } \
  } while (0)

#define CONFLICT_LITERAL(LIT_ARG) \
  do { \
    unsigned LIT = (LIT_ARG); \
    unsigned LIT_IDX = IDX (LIT); \
    struct variable *V = ring->variables + LIT_IDX; \
    unsigned LIT_LEVEL = V->level; \
    if (forced_literal == INVALID_LIT || LIT_LEVEL > conflict_level) { \
      conflict_level = LIT_LEVEL; \
      literals_on_conflict_level = 1; \
      forced_literal = LIT; \
    } else if (LIT_LEVEL == conflict_level) \
      literals_on_conflict_level++; \
  } while (0)

bool analyze (struct ring *ring, struct watch *reason) {
  assert (!ring->inconsistent);
  if (!ring->level) {
    set_inconsistent (ring, "conflict on root-level produces empty clause");
    return false;
  }
  unsigned conflict_level = 0;
  unsigned literals_on_conflict_level = 0;
  unsigned forced_literal = INVALID_LIT;
  assert (reason);
  if (is_binary_pointer (reason)) {
    unsigned lit = lit_pointer (reason);
    unsigned other = other_pointer (reason);
    CONFLICT_LITERAL (lit);
    CONFLICT_LITERAL (other);
  } else {
    struct watcher *watcher = get_watcher (ring, reason);
    for (all_watcher_literals (lit, watcher))
      CONFLICT_LITERAL (lit);
  }
  assert (conflict_level <= ring->level);
  if (conflict_level < ring->level) {
    LOG ("forced to backtrack to conflict level %u", conflict_level);
    backtrack (ring, conflict_level);
  } else
    LOG ("conflict level %u matches decision level", conflict_level);
  if (!conflict_level) {
    set_inconsistent (ring, "conflict on root-level produces empty clause");
    return false;
  }
  if (literals_on_conflict_level == 1) {
    LOG ("only literal %s on conflict level", LOGLIT (forced_literal));
    backtrack (ring, conflict_level - 1);
    LOGWATCH (reason, "forcing %s through", LOGLIT (forced_literal));
    if (is_binary_pointer (reason)) {
      unsigned lit = lit_pointer (reason);
      unsigned other = other_pointer (reason);
      assert (lit == forced_literal || other == forced_literal);
      other = lit ^ other ^ forced_literal;
      assert (other != forced_literal);
      bool redundant = redundant_pointer (reason);
      reason = tag_binary (redundant, forced_literal, other);
    }
    assign_with_reason (ring, forced_literal, reason);
    return true;
  } else
    LOG ("conflict has %u literals on conflict level",
         literals_on_conflict_level);
  struct unsigneds *ring_clause = &ring->clause;
  struct unsigneds *analyzed = &ring->analyzed;
  struct unsigneds *levels = &ring->levels;
  assert (EMPTY (*ring_clause));
  assert (EMPTY (*analyzed));
  assert (EMPTY (*levels));
  unsigned char *used = ring->used;
  struct variable *variables = ring->variables;
  struct ring_trail *trail = &ring->trail;
  unsigned *t = trail->end;
  PUSH (*ring_clause, INVALID);
  const unsigned level = ring->level;
  unsigned uip = INVALID, jump = 0, glue = 0, open = 0;
  for (;;) {
    assert (reason);
    LOGWATCH (reason, "analyzing");
    if (is_binary_pointer (reason)) {
      unsigned lit = lit_pointer (reason);
      unsigned other = other_pointer (reason);
      RESOLVE_LITERAL (lit);
      RESOLVE_LITERAL (other);
    } else {
      struct watcher *watcher = get_watcher (ring, reason);
      if (watcher->redundant)
        bump_reason (ring, watcher);
      for (all_watcher_literals (lit, watcher))
        RESOLVE_LITERAL (lit);
    }
    struct variable *v;
    do {
      assert (t > ring->trail.begin);
      uip = *--t;
      unsigned uip_idx = IDX (uip);
      v = ring->variables + uip_idx;
    } while (!v->seen || v->level != conflict_level);
    if (!--open)
      break;
    reason = variables[IDX (uip)].reason;
    assert (reason);
  }
  LOG ("back jump level %u", jump);
  struct averages *a = ring->averages + ring->stable;
  update_average (ring, &a->level, "level", SLOW_ALPHA, jump);
  LOG ("glucose level (LBD) %u", glue);
  update_average (ring, &a->glue.slow, "slow glue", SLOW_ALPHA, glue);
  update_average (ring, &a->glue.fast, "fast glue", FAST_ALPHA, glue);
  unsigned assigned = SIZE (ring->trail);
  double filled = percent (assigned, ring->size);
  LOG ("assigned %u variables %.0f%% filled", assigned, filled);
  update_average (ring, &a->trail, "trail", SLOW_ALPHA, filled);
  update_decision_rate (ring);
  unsigned *literals = ring_clause->begin;
  const unsigned not_uip = NOT (uip);
  literals[0] = not_uip;
  LOGTMP ("first UIP %s", LOGLIT (uip));
  shrink_or_minimize_clause (ring, glue);
  analyze_reason_side_literals (ring);
  bump_variables (ring);
  unsigned back = level - 1;
  backtrack (ring, back);
  update_best_and_target_phases (ring);
  if (jump != back) {
    if (!ring->options.chronological ||
        back < ring->options.backjump_limit ||
        back - ring->options.backjump_limit <= jump)
      backtrack (ring, jump);
    else {
      LOG ("chronological backtracking only (staying at %u not %u)", back,
           jump);
      ring->statistics.contexts[ring->context].chronological++;
    }
  }
  unsigned size = SIZE (*ring_clause);
  update_average (ring, &a->size, "size", SLOW_ALPHA, size);
  assert (size);
  if (size == 1) {
    trace_add_unit (&ring->trace, not_uip);
    assign_ring_unit (ring, not_uip);
    ring->iterating = 1;
  } else {
    unsigned other = literals[1];
    struct watch *learned;
    if (size == 2) {
      assert (VAR (other)->level == jump);
      learned = new_local_binary_clause (ring, true, not_uip, other);
      trace_add_binary (&ring->trace, not_uip, other);
      if (ring->options.eagerly_subsume)
        eagerly_subsume_last_learned (ring);
      export_binary_clause (ring, learned);
    } else {
      if (ring->options.sort_deduced)
        sort_deduced_clause (ring);
      else if (VAR (other)->level != jump) {
        unsigned *p = literals + 2, replacement;
        while (assert (p != ring_clause->end),
               VAR (replacement = *p)->level != jump)
          p++;
        literals[1] = replacement;
        *p = other;
      }
      struct clause *learned_clause =
          new_large_clause (size, literals, true, glue);
      learned_clause->origin = ring->id;
      LOGCLAUSE (learned_clause, "new");
      learned =
          watch_first_two_literals_in_large_clause (ring, learned_clause);
      assert (!is_binary_pointer (learned));
      trace_add_clause (&ring->trace, learned_clause);
      if (ring->options.eagerly_subsume) {
        eagerly_subsume_last_learned (ring);
        insert_last_learned (ring, learned);
      }
      export_large_clause (ring, learned_clause);
    }
    assign_with_reason (ring, not_uip, learned);
  }
  CLEAR (*ring_clause);
  clear_analyzed (ring);
  if (SEARCH_CONFLICTS > ring->limits.tiers)
    update_tier_limits (ring);
  return true;
}

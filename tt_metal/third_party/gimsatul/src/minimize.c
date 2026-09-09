#include "minimize.h"
#include "macros.h"
#include "ring.h"

static bool minimize_literal (struct ring *ring, unsigned lit,
                              unsigned depth) {
  assert (ring->values[lit] < 0);
  if (depth >= ring->options.minimize_depth)
    return false;
  unsigned idx = IDX (lit);
  struct variable *v = ring->variables + idx;
  if (!v->level)
    return true;
  if (!ring->used[v->level])
    return false;
  if (depth && (v->seen))
    return true;
  if (v->minimize)
    return true;
  if (v->poison)
    return false;
  struct watch *reason = v->reason;
  if (!reason)
    return false;
  depth++;
  bool res = true;
  const unsigned not_lit = NOT (lit);
  if (is_binary_pointer (reason)) {
    assert (lit_pointer (reason) == not_lit);
    unsigned other = other_pointer (reason);
    res = minimize_literal (ring, other, depth);
  } else {
    assert (ring->context == SEARCH_CONTEXT);
    ring->statistics.contexts[SEARCH_CONTEXT].ticks++;
    struct watcher *watcher = get_watcher (ring, reason);
    for (all_watcher_literals (other, watcher))
      if (other != not_lit && !minimize_literal (ring, other, depth)) {
        res = false;
        break;
      }
  }
  if (!v->shrinkable)
    PUSH (ring->minimize, idx);
  if (res)
    v->minimize = true;
  else
    v->poison = true;
  return res;
}

#define SHRINK_LITERAL(OTHER) \
  do { \
    if (OTHER == uip) \
      break; \
    assert (ring->values[OTHER] < 0); \
    unsigned OTHER_IDX = IDX (OTHER); \
    struct variable *V = variables + OTHER_IDX; \
    unsigned OTHER_LEVEL = V->level; \
    assert (OTHER_LEVEL <= level); \
    if (!OTHER_LEVEL) \
      break; \
    if (OTHER_LEVEL != level) { \
      LOG ("shrinking failed due to %s", LOGLIT (OTHER)); \
      return 0; \
    } \
    if (V->shrinkable) \
      break; \
    if (!v->poison && !v->minimize) \
      PUSH (*minimize, OTHER_IDX); \
    V->shrinkable = true; \
    open++; \
  } while (0)

static size_t shrink_clause (struct ring *ring) {
  LOGTMP ("trying to shrink");

  struct variable *variables = ring->variables;
  struct unsigneds *minimize = &ring->minimize;
  struct ring_trail *trail = &ring->trail;

  struct unsigneds *clause = &ring->clause;
  unsigned *begin = clause->begin;
  unsigned *end = clause->end;

  unsigned max_pos = 0, open = 0;
  unsigned level = INVALID;

  size_t shrunken = 0;

  for (unsigned *p = begin + 1; p != end; p++) {
    unsigned lit = *p;
    unsigned idx = IDX (lit);
    struct variable *v = variables + idx;
    assert (v->level < ring->level);
    if (!v->level)
      continue;
    if (level == INVALID)
      level = v->level;
    else
      assert (v->level == level);
    if (!v->shrinkable && !v->poison && !v->minimize)
      PUSH (*minimize, idx);
    v->shrinkable = true;
    unsigned pos = trail->pos[idx];
    if (pos > max_pos)
      max_pos = pos;
    open++;
  }
  LOG ("maximum trail position %u of level %u block of size %u", max_pos,
       level, open);

  assert (max_pos > 0), assert (open > 1);
  assert (level), assert (level != INVALID);

  unsigned *t = trail->begin + max_pos, uip = INVALID;

  while (open) {
    uip = *t--;
    unsigned idx = IDX (uip);
    struct variable *v = variables + idx;
    if (v->level != level)
      continue;
    if (!v->shrinkable)
      continue;
    struct watch *reason = v->reason;
    if (is_binary_pointer (reason)) {
      unsigned other = other_pointer (reason);
      SHRINK_LITERAL (other);
    } else if (reason) {
      struct watcher *watcher = get_watcher (ring, reason);
      for (all_watcher_literals (other, watcher))
        SHRINK_LITERAL (other);
    }
    assert (open);
    open--;
  }

  assert (uip != INVALID);
  LOGTMP ("shrinking succeeded with first UIP %s1 of", LOGLIT (uip));
  unsigned not_uip = NOT (uip);
  clause->begin[1] = not_uip;
  size_t deduced = end - begin;
  clause->end = clause->begin + 2;
  shrunken = deduced - 2;
  assert (shrunken);

  return shrunken;
}

static size_t minimize_clause (struct ring *ring) {
  LOGTMP ("trying to minimize clause");
  struct unsigneds *clause = &ring->clause;
  unsigned *begin = clause->begin, *q = begin + 1;
  unsigned *end = clause->end;
  size_t minimized = 0;
  for (unsigned *p = q; p != end; p++) {
    unsigned lit = *q++ = *p;
    if (!minimize_literal (ring, lit, 0))
      continue;
    LOG ("minimized literal %s", LOGLIT (lit));
    minimized++;
    q--;
  }
  clause->end = q;
  return minimized;
}

void shrink_or_minimize_clause (struct ring *ring, unsigned glue) {
  struct unsigneds *minimize = &ring->minimize;
  assert (EMPTY (*minimize));

  size_t deduced = SIZE (ring->clause);

  size_t minimized = 0;
  size_t shrunken = 0;

  if (ring->options.shrink)
    if (glue == 1 && deduced > 2)
      shrunken = shrink_clause (ring);

  if (ring->options.minimize)
    if (glue && !shrunken && deduced > 2)
      minimized = minimize_clause (ring);

  size_t learned = SIZE (ring->clause);
  assert (learned + minimized + shrunken == deduced);
  (void) minimized;

  INC_CLAUSE_STATISTICS (learned, glue, learned);
  ring->statistics.literals.learned += learned;
#ifdef METRICS
  ring->statistics.literals.minimized += minimized;
  ring->statistics.literals.shrunken += shrunken;
  ring->statistics.literals.deduced += deduced;
#endif

  struct variable *variables = ring->variables;
  for (all_elements_on_stack (unsigned, idx, *minimize)) {
    struct variable *v = variables + idx;
    assert (v->poison || v->minimize || v->shrinkable);
    v->poison = v->minimize = v->shrinkable = false;
  }
  CLEAR (*minimize);

  LOG ("minimized %zu literals out of %zu", minimized, deduced);
  LOG ("shrunken %zu literals out of %zu", shrunken, deduced);
}

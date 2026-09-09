#include "watches.h"
#include "clause.h"
#include "message.h"
#include "ring.h"
#include "tagging.h"
#include "trace.h"
#include "utilities.h"
#include "vivify.h"

#include <string.h>

void release_references (struct ring *ring) {
  if (ring->references)
    for (all_ring_literals (lit))
      RELEASE (REFERENCES (lit));
}

void disconnect_references (struct ring *ring, struct watches *saved) {
#ifndef QUIET
  size_t disconnected = 0;
#endif
  for (all_ring_literals (lit)) {
    struct references *watches = &REFERENCES (lit);
    for (all_watches (watch, *watches))
      if (is_binary_pointer (watch)) {
        assert (redundant_pointer (watch));
        assert (lit_pointer (watch) == lit);
        unsigned other = other_pointer (watch);
        if (other < lit)
          PUSH (*saved, watch);
      }
#ifndef QUIET
    disconnected += SIZE (*watches);
#endif
    RELEASE (*watches);
  }
  very_verbose (ring, "disconnected %zu clauses", disconnected);
}

void reconnect_watches (struct ring *ring, struct watches *saved) {
#ifndef QUIET
  size_t reconnected = 0;
#endif
  for (all_watchers (watcher)) {
    unsigned *literals = watcher->clause->literals;
    watcher->sum = literals[0] ^ literals[1];
    watch_literal (ring, literals[0], literals[1], watcher);
    watch_literal (ring, literals[1], literals[0], watcher);
#ifndef QUIET
    reconnected++;
#endif
  }
  for (all_watches (lit_watch, *saved)) {
    assert (is_binary_pointer (lit_watch));
    assert (redundant_pointer (lit_watch));
    unsigned lit = lit_pointer (lit_watch);
    unsigned other = other_pointer (lit_watch);
    struct watch *other_watch = tag_binary (true, other, lit);
    push_watch (ring, lit, lit_watch);
    push_watch (ring, other, other_watch);
  }
  very_verbose (ring, "reconnected %zu clauses", reconnected);
  ring->trail.propagate = ring->trail.begin;
}

struct watch *watch_literals_in_large_clause (struct ring *ring,
                                              struct clause *clause,
                                              unsigned first,
                                              unsigned second) {
  assert (clause->size > 2);
  assert (!clause->garbage);
  assert (!clause->dirty);
#ifndef NDEBUG
  assert (first != second);
  bool found_first = false;
  for (all_literals_in_clause (lit, clause))
    found_first |= (lit == first);
  assert (found_first);
  bool found_second = false;
  for (all_literals_in_clause (lit, clause))
    found_second |= (lit == second);
  assert (found_second);
#endif
  size_t size_watchers = SIZE (ring->watchers);
  if (size_watchers >= MAX_WATCHER_INDEX)
    fatal_error ("more than %zu watched clauses in ring[%u]",
                 (size_t) MAX_WATCHER_INDEX, ring->id);
  unsigned idx = size_watchers;

  if (FULL (ring->watchers))
    ENLARGE (ring->watchers);
  struct watcher *watcher = ring->watchers.end++;
  assert (ring->watchers.end <= ring->watchers.allocated);

  unsigned size = clause->size;
  unsigned glue = clause->glue;

  if (clause->origin != ring->id) {
    unsigned increase = ring->options.increase_imported_glue;
    if (increase == 2)
      glue = MAX_GLUE;
    else if (increase && glue < MAX_GLUE)
      glue++;
  }

  bool redundant = clause->redundant;

  if (size > SIZE_WATCHER_LITERALS)
    size = 0;

  unsigned used = MAX_USED;

  assert (size < (1 << (8 * sizeof watcher->size)));
  assert (glue < (1 << (8 * sizeof watcher->glue)));
  assert (used < (1 << (8 * sizeof watcher->used)));

  watcher->size = size;
  watcher->glue = glue;
  watcher->used = used;

  watcher->garbage = false;
  watcher->reason = false;
  watcher->redundant = redundant;
  watcher->vivify = false;

  watcher->sum = first ^ second;
  watcher->clause = clause;

  if (size)
    memcpy (watcher->aux, clause->literals, size * sizeof (unsigned));
  else
    watcher->aux[0] = 2;

  inc_clauses (ring, redundant);

  struct watch *first_watch = tag_index (redundant, idx, second);
  struct watch *second_watch = tag_index (redundant, idx, first);
  push_watch (ring, first, first_watch);
  push_watch (ring, second, second_watch);

  return tag_index (true, idx, INVALID_LIT);
}

struct watch *
watch_first_two_literals_in_large_clause (struct ring *ring,
                                          struct clause *clause) {
  unsigned *lits = clause->literals;
  return watch_literals_in_large_clause (ring, clause, lits[0], lits[1]);
}

struct watch *new_local_binary_clause (struct ring *ring, bool redundant,
                                       unsigned lit, unsigned other) {
  inc_clauses (ring, redundant);
  struct watch *watch_lit = tag_binary (redundant, lit, other);
  struct watch *watch_other = tag_binary (redundant, other, lit);
  push_watch (ring, lit, watch_lit);
  push_watch (ring, other, watch_other);
  LOGBINARY (redundant, lit, other, "new");
  return watch_lit;
}

unsigned *flush_watchers (struct ring *ring, unsigned start) {
  assert (start);
  struct watchers *watchers = &ring->watchers;
  assert (!EMPTY (*watchers));
  assert (!watchers->begin[0].sum);

  struct watcher *begin = watchers->begin + start;
  struct watcher *end = watchers->end;
  struct watcher *q = begin;

  size_t size = end - begin;
  unsigned *map = allocate_and_clear_array (size, sizeof *map);

  unsigned src = 0;
  unsigned dst = start;

  unsigned redundant = 0;
#ifndef QUIET
  unsigned flushed = 0;
  unsigned deleted = 0;
  unsigned mapped = 0;
#endif
  if (start >= ring->redundant) {
    assert (ring->redundant);
    redundant = ring->redundant;
  }

  for (struct watcher *p = begin; p != end; p++, src++) {
    if (p->garbage && !p->reason) {
      struct clause *clause = p->clause;
#ifndef QUIET
      deleted += dereference_clause (ring, clause);
      flushed++;
#else
      (void) dereference_clause (ring, clause);
#endif
    } else {
      *q++ = *p;

      if (!redundant && p->redundant)
        redundant = dst;

      assert (src < size);
      assert (dst < MAX_WATCHER_INDEX);
      map[src] = dst++;
#ifndef QUIET
      mapped++;
#endif
    }
  }
  watchers->end = q;

  verbose (ring, "mapped %u non-garbage watchers %.0f%%", mapped,
           percent (mapped, size));
  verbose (ring, "flushed %u garbage watched and deleted %u clauses %.0f%%",
           flushed, deleted, percent (deleted, flushed));

  if (redundant) {
    very_verbose (ring, "redundant clauses start at watcher index %u",
                  redundant);
    ring->redundant = redundant;
  } else {
    very_verbose (ring, "no redundant clauses watched");
    ring->redundant = SIZE (*watchers);
  }

  assert (ring->redundant);

  return map;
}

void mark_garbage_watcher (struct ring *ring, struct watcher *watcher) {
  LOGCLAUSE (watcher->clause, "marking garbage watcher to");
  assert (!watcher->garbage);
  watcher->garbage = true;
  dec_clauses (ring, watcher->redundant);
}

static void sort_by_size (struct ring *ring, size_t size_indices,
                          unsigned *indices) {
  size_t size_count = 256, count[size_count];
  size_t bytes = size_indices * sizeof *indices;
  unsigned *end = indices + size_indices;
  for (unsigned shift = 0; shift != 32; shift += 8) {
    memset (count, 0, sizeof count);
    for (unsigned *p = indices; p != end; p++) {
      unsigned idx = *p;
      struct watcher *watcher = index_to_watcher (ring, idx);
      assert (watcher->redundant);
      unsigned size = watcher->size ? watcher->size : watcher->clause->size;
      unsigned byte = (size >> shift) & 255;
      count[byte]++;
    }
    {
      size_t pos = 0, *c = count + size_count, size;
      while (c-- != count)
        size = *c, *c = pos, pos += size;
    }
    unsigned *tmp = sorter_block (ring, size_indices);
    for (unsigned *p = indices; p != end; p++) {
      unsigned idx = *p;
      struct watcher *watcher = index_to_watcher (ring, idx);
      unsigned size = watcher->size ? watcher->size : watcher->clause->size;
      unsigned byte = (size >> shift) & 255;
      tmp[count[byte]++] = idx;
    }
    memcpy (indices, tmp, bytes);
  }
}

static void sort_by_glue (struct ring *ring, size_t size_indices,
                          unsigned *indices) {
  size_t size_count = MAX_GLUE + 1, count[size_count];
  memset (count, 0, sizeof count);
  unsigned *end = indices + size_indices;
  for (unsigned *p = indices; p != end; p++) {
    unsigned idx = *p;
    struct watcher *watcher = index_to_watcher (ring, idx);
    assert (watcher->glue <= MAX_GLUE);
    assert (watcher->redundant);
    count[watcher->glue]++;
  }
  {
    size_t pos = 0, *c = count + size_count, size;
    while (c-- != count)
      size = *c, *c = pos, pos += size;
  }
  unsigned *tmp = sorter_block (ring, size_indices);
  for (unsigned *p = indices; p != end; p++) {
    unsigned idx = *p;
    struct watcher *watcher = index_to_watcher (ring, idx);
    tmp[count[watcher->glue]++] = idx;
  }
  size_t bytes = size_indices * sizeof *indices;
  memcpy (indices, tmp, bytes);
}

void sort_redundant_watcher_indices (struct ring *ring, size_t size_indices,
                                     unsigned *indices) {
  if (size_indices < 2)
    return;
  sort_by_size (ring, size_indices, indices);
  sort_by_glue (ring, size_indices, indices);
}

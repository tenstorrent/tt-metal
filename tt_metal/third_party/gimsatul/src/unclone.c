#include "unclone.h"
#include "message.h"
#include "ruler.h"

static void save_ring_binaries (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
  if (!ring->id) {
    assert (!ruler->occurrences);
    assert (ruler->compact == ring->size);
    ruler->occurrences = allocate_and_clear_array (
        2 * ring->size, sizeof *ruler->occurrences);
  }

  struct saved_watchers *saved = &ring->saved;
  assert (EMPTY (*saved));
  size_t irredundant = 0;

  for (all_ring_literals (lit)) {
    struct references *references = &REFERENCES (lit);
    for (all_watches (watch, *references)) {
      if (!is_binary_pointer (watch))
        continue;
      assert (redundant_pointer (watch));
      unsigned other = other_pointer (watch);
      if (other >= lit)
        continue;
      struct saved_watcher sw;
      sw.used = 0;
      sw.vivify = 0;
      sw.clause = (struct clause *) watch;
      PUSH (*saved, sw);
    }
    RELEASE (*references);
    if (ring->id)
      continue;
    unsigned *binaries = references->binaries;
    if (!binaries)
      continue;
    struct clauses *occurrenes = &OCCURRENCES (lit);
    for (unsigned *p = binaries, other; (other = *p) != INVALID; p++) {
      struct clause *clause = tag_binary (false, lit, other);
      PUSH (*occurrenes, clause);
      if (lit < other)
        irredundant++;
    }
    free (binaries);
  }

  size_t redundant = SIZE (*saved);

  if (ring->id)
    irredundant = ruler->statistics.binaries;
  else
    assert (irredundant == ruler->statistics.binaries);

  very_verbose (ring, "saved %zu binary redundant watches", redundant);
  very_verbose (ring, "flushed %zu binary irredundant watches",
                irredundant);

  assert (ring->statistics.irredundant >= irredundant);
  ring->statistics.irredundant -= irredundant;

  assert (ring->statistics.redundant >= redundant);
  ring->statistics.redundant -= redundant;
}

static void save_large_watched_clauses (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
  struct clauses *clauses = &ruler->clauses;
  assert (ring->id || EMPTY (*clauses));
  struct saved_watchers *save = &ring->saved;
#ifndef QUIET
  size_t collected = 0, saved = 0;
#endif
#if !defined(QUIET) || !defined(NDEBUG)
  size_t transferred = 0, flushed = 0;
#endif
  for (all_watchers (watcher)) {
    struct clause *clause = watcher->clause;
    if (watcher->garbage) {
      dereference_clause (ring, clause);
#ifndef QUIET
      collected++;
#endif
    } else {
      if (watcher->redundant) {
        struct saved_watcher sw = saved_watcher_from_watcher (watcher);
        PUSH (*save, sw);
#ifndef QUIET
        saved++;
#endif
      } else if (ring->id) {
        dereference_clause (ring, clause);
#if !defined(QUIET) || !defined(NDEBUG)
        flushed++;
#endif
      } else {
        PUSH (*clauses, clause);
#if !defined(QUIET) || !defined(NDEBUG)
        transferred++;
#endif
      }
      dec_clauses (ring, watcher->redundant);
    }
  }
  RESIZE (ring->watchers, 1);
  very_verbose (ring, "saved %zu redundant large watches", saved);
  very_verbose (ring, "collected %zu large watches", collected);
  if (ring->id) {
    assert (!transferred);
    very_verbose (ring, "flushed %zu irredundant large watches", flushed);
  } else {
    assert (!flushed);
    very_verbose (ring, "transferred %zu irredundant large clauses",
                  transferred);
  }
}

void unclone_ring (struct ring *ring) {
  save_ring_binaries (ring);
  save_large_watched_clauses (ring);
  reset_last_learned (ring);
  assert (SIZE (ring->watchers) == 1);
  release_ring (ring, true);
  assert (SIZE (ring->watchers) == 1);
}

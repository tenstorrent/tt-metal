#include "compact.h"
#include "message.h"
#include "ruler.h"
#include "simplify.h"
#include "utilities.h"

#include <string.h>

#include "cover.h"

static unsigned map_literal (unsigned *map, unsigned original_lit) {
  unsigned original_idx = IDX (original_lit);
  unsigned mapped_idx = map[original_idx];
  if (mapped_idx == INVALID)
    return INVALID;
  unsigned mapped_lit = LIT (mapped_idx);
  if (SGN (original_lit))
    mapped_lit = NOT (mapped_lit);
  return mapped_lit;
}

static void map_occurrences (struct ruler *ruler, unsigned *map,
                             unsigned src) {
  unsigned dst = map_literal (map, src);
  struct clauses *src_occurrences = &OCCURRENCES (src);
  struct clauses *dst_occurrences = &OCCURRENCES (dst);
  *dst_occurrences = *src_occurrences;
  struct clause **begin = dst_occurrences->begin;
  struct clause **end = dst_occurrences->end;
  struct clause **q = begin;
  for (struct clause **p = begin; p != end; p++) {
    struct clause *src_clause = *p;
    if (!is_binary_pointer (src_clause))
      continue;
    assert (lit_pointer (src_clause) == src);
    unsigned src_other = other_pointer (src_clause);
    unsigned dst_other = map_literal (map, src_other);
    assert (!redundant_pointer (src_clause));
    struct clause *dst_clause = tag_binary (false, dst, dst_other);
    *q++ = dst_clause;
  }
  dst_occurrences->end = q;
}

static void map_large_clause (unsigned *map, struct clause *clause) {
  assert (!is_binary_pointer (clause));
  assert (!clause->redundant);
  unsigned *literals = clause->literals;
  unsigned *end = literals + clause->size;
  for (unsigned *p = literals; p != end; p++)
    *p = map_literal (map, *p);
}

static void map_clauses (struct ruler *ruler, unsigned *map) {
  struct clauses *clauses = &ruler->clauses;
  struct clause **begin = clauses->begin;
  struct clause **end = clauses->end;
  for (struct clause **p = begin; p != end; p++)
    map_large_clause (map, *p);
}

/*------------------------------------------------------------------------*/

static void clean_ring (struct ring *ring, struct clauses *cleaned) {
  struct saved_watchers *saved = &ring->saved;
  signed char *values = ring->values;
  struct saved_watcher *begin = saved->begin;
  struct saved_watcher *end = saved->end;
  struct saved_watcher *q = begin;
  struct saved_watcher *p = q;
  struct unsigneds delete;
  struct unsigneds add;
  INIT (delete);
  INIT (add);
  while (p != end) {
    struct saved_watcher sw = *p++;
    struct clause *clause = sw.clause;
    if (is_binary_pointer (clause))
      *q++ = sw;
    else if (clause->garbage)
      dereference_clause (ring, clause);
    else if (clause->cleaned)
      *q++ = sw;
    else {
      assert (clause->redundant);
      bool satisfied = false, falsified = false;
      for (all_literals_in_clause (lit, clause)) {
        signed char value = values[lit];
        if (value > 0) {
          satisfied = true;
          break;
        }
        if (value < 0)
          falsified = true;
      }
      if (satisfied) {
        clause->garbage = true;
        LOGCLAUSE (clause, "satisfied");
        dereference_clause (ring, clause);
      } else if (!falsified) {
        LOGCLAUSE (clause, "already clean");
        clause->cleaned = true;
        PUSH (*cleaned, clause);
        *q++ = sw;
      } else {
        LOGCLAUSE (clause, "cleaning");
        assert (EMPTY (add));
        assert (EMPTY (delete));
        for (all_literals_in_clause (lit, clause)) {
          PUSH (delete, lit);
          signed char value = values[lit];
          assert (value <= 0);
          if (!value)
            PUSH (add, lit);
        }
        unsigned new_size = SIZE (add);
        unsigned old_size = SIZE (delete);
        assert (old_size == clause->size);
        trace_add_literals (&ring->trace, new_size, add.begin, INVALID);
#if 1
        if (new_size <= 1) {
          printf ("c ring %u compact new_size %u old_size %u\n", ring->id,
                  new_size, old_size);
          fflush (stdout);
        }
#endif
        assert (new_size > 1);
        if (new_size == 2) {
          unsigned lit = add.begin[0];
          unsigned other = add.begin[1];
          if (lit > other)
            SWAP (unsigned, lit, other);
          LOGBINARY (true, lit, other, "cleaned");
          struct clause *binary = tag_binary (true, lit, other);
          dereference_clause (ring, clause);
          sw.used = 0;
          sw.vivify = 0;
          sw.clause = binary;
          *q++ = sw;
        } else {
          trace_delete_literals (&ring->trace, old_size, delete.begin);
          assert (new_size > 2);
          unsigned old_glue = clause->glue;
          unsigned new_glue = old_glue;
          if (new_glue >= new_size - 1) {
            new_glue = new_size - 1;
            clause->glue = new_glue;
          }
          memcpy (clause->literals, add.begin,
                  new_size * sizeof (unsigned));
          clause->size = new_size;
          LOGCLAUSE (clause, "cleaned");
          clause->cleaned = true;
          PUSH (*cleaned, clause);
          *q++ = sw;
        }
        CLEAR (delete);
        CLEAR (add);
      }
    }
  }
  saved->end = q;
  RELEASE (delete);
  RELEASE (add);
}

static void clean_rings (struct ruler *ruler) {
  struct clauses cleaned_clauses;
  INIT (cleaned_clauses);

  for (all_rings (ring))
    clean_ring (ring, &cleaned_clauses);

  for (all_clauses (clause, cleaned_clauses))
    assert (clause->cleaned), clause->cleaned = false;

  very_verbose (0, "cleaned %zu clauses in total", SIZE (cleaned_clauses));
  RELEASE (cleaned_clauses);
}

/*------------------------------------------------------------------------*/

static void compact_phases (struct ring *ring, unsigned old_size,
                            unsigned new_size, unsigned *map) {
  struct phases *old_phases = ring->phases;
  struct phases *new_phases = ring->phases =
      allocate_array (new_size, sizeof *new_phases);
  struct phases *old_phase = old_phases;
  struct phases *new_phase = new_phases;
  unsigned *end = map + old_size;
  for (unsigned *mapped = map; mapped != end; mapped++, old_phase++) {
    unsigned new_idx = *mapped;
    if (new_idx == INVALID)
      continue;
    assert (new_phases + new_idx == new_phase);
    *new_phase++ = *old_phase;
  }
  assert (old_phase == old_phases + old_size);
  assert (new_phase == new_phases + new_size);
  free (old_phases);
}

static void compact_heap (struct ring *ring, struct heap *heap,
                          unsigned old_size, unsigned new_size,
                          unsigned *map) {
  struct node *old_nodes = heap->nodes;
  struct node *new_nodes = heap->nodes =
      allocate_and_clear_array (new_size, sizeof *new_nodes);
  heap->root = 0;
  struct node *new_node = new_nodes, *old_node = old_nodes;
  unsigned *end = map + old_size;
  for (unsigned *mapped = map; mapped != end; mapped++, old_node++) {
    unsigned new_idx = *mapped;
    if (new_idx == INVALID)
      continue;
    assert (new_nodes + new_idx == new_node);
    new_node->score = old_node->score;
    push_heap (heap, new_node);
    new_node++;
  }
  assert (old_node == old_nodes + old_size);
  assert (new_node == new_nodes + new_size);
  free (old_nodes);
}

static void compact_queue (struct ring *ring, struct queue *queue,
                           unsigned old_size, unsigned new_size,
                           unsigned *map) {
  struct link *old_links = queue->links;
  struct link *new_links = queue->links =
      allocate_and_clear_array (new_size, sizeof *new_links);
  struct link *first = queue->first;
  queue->first = queue->last = 0;
  queue->stamp = 0;
  for (struct link *old_link = first; old_link; old_link = old_link->next) {
    unsigned old_idx = old_link - old_links;
    unsigned new_idx = map[old_idx];
    if (new_idx == INVALID)
      continue;
    struct link *new_link = new_links + new_idx;
    enqueue (queue, new_link, false);
  }
  assert (queue->stamp == new_size);
  reset_queue_search (queue);
  free (old_links);
}

/*------------------------------------------------------------------------*/

static void compact_saved (struct ring *ring, unsigned *map,
                           struct clauses *mapped) {
#ifdef LOGGING
  assert (!ignore_values_and_levels_during_logging);
  ignore_values_and_levels_during_logging = true;
  unsigned *unmap = ring->ruler->unmap;
#endif
  struct saved_watchers *saved = &ring->saved;
  struct saved_watcher *begin = saved->begin;
  struct saved_watcher *end = saved->end;
  struct saved_watcher *q = begin;
  struct saved_watcher *p = q;
  while (p != end) {
    struct saved_watcher src_watcher = *p++;
    struct clause *src_clause = src_watcher.clause;
    if (is_binary_pointer (src_clause)) {
      assert (redundant_pointer (src_clause));
      unsigned src_lit = lit_pointer (src_clause);
      unsigned src_other = other_pointer (src_clause);
      unsigned dst_lit = map_literal (map, src_lit);
      unsigned dst_other = map_literal (map, src_other);
      if (dst_lit != INVALID && dst_other != INVALID) {
        struct clause *dst_clause;
        LOGBINARY (true, src_lit, src_other, "mapping");
        if (dst_lit < dst_other)
          dst_clause = tag_binary (true, dst_lit, dst_other);
        else
          dst_clause = tag_binary (true, dst_other, dst_lit);
        assert (dst_clause);
        LOG ("mapped redundant binary clause %u(%d) %u(%d)", dst_lit,
             unmap_and_export_literal (unmap, src_lit), dst_other,
             unmap_and_export_literal (unmap, src_other));
        *q++ = saved_watcher_from_binary (dst_clause);
      } else {
#ifdef LOGGING
        if (dst_lit == INVALID)
          LOG ("cannot map literal %s", LOGLIT (src_lit));
        else if (dst_other == INVALID)
          LOG ("cannot map literal %s", LOGLIT (src_other));
        LOGBINARY (true, src_lit, src_other, "cannot map");
#endif
      }
    } else if (src_clause->garbage)
      dereference_clause (ring, src_clause);
    else if (src_clause->mapped)
      *q++ = src_watcher;
    else {
      struct clause *dst_clause = src_clause;
      for (all_literals_in_clause (src_lit, src_clause))
        if (map_literal (map, src_lit) == INVALID) {
          LOG ("cannot map literal %s", LOGLIT (src_lit));
          dst_clause = 0;
          break;
        }
      if (dst_clause) {
        LOGCLAUSE (src_clause, "mapping");
        assert (src_clause == dst_clause);
        unsigned *literals = dst_clause->literals;
        unsigned *end = literals + dst_clause->size;
#ifdef LOGGING
        if (verbosity == INT_MAX) {
          assert (src_clause->redundant);
          LOGPREFIX ("mapped redundant glue %u size %u clause[%" PRIu64 "]",
                     src_clause->glue, src_clause->size, src_clause->id);
        }
#endif
        for (unsigned *p = literals; p != end; p++) {
          unsigned src_lit = *p;
          unsigned dst_lit = map_literal (map, src_lit);
          assert (dst_lit != INVALID);
          *p = dst_lit;
#ifdef LOGGING
          if (verbosity == INT_MAX)
            printf (" %u(%d)", dst_lit,
                    unmap_and_export_literal (unmap, src_lit));
#endif
        }
#ifdef LOGGING
        if (verbosity == INT_MAX) {
          LOGSUFFIX ();
        }
#endif
        dst_clause->mapped = true;
        PUSH (*mapped, dst_clause);
        *q++ = src_watcher;
      } else {
        src_clause->garbage = true;
        LOGCLAUSE (src_clause, "cannot map");
        dereference_clause (ring, src_clause);
      }
    }
  }
#ifdef LOGGING
  assert (ignore_values_and_levels_during_logging);
  ignore_values_and_levels_during_logging = false;
#endif
#ifndef QUIET
  size_t flushed = end - q;
  size_t kept = q - begin;
  verbose (ring, "flushed %zu clauses during compaction", flushed);
  verbose (ring, "kept %zu clauses during compaction", kept);
#endif
  saved->end = q;
}

/*------------------------------------------------------------------------*/

static void compact_ring (struct ring *ring, unsigned *map,
                          struct clauses *mapped) {
  struct ruler *ruler = ring->ruler;
  unsigned old_size = ring->size;
  unsigned new_size = ruler->compact;
  assert (new_size <= old_size);
  (void) old_size, (void) new_size;

  ring->best = 0;
  assert (ring->context == SEARCH_CONTEXT);
  assert (!ring->level);
  ring->probe = ring->id * (new_size / SIZE (ruler->rings));
  ring->size = new_size;
  ring->target = 0;
  ring->unassigned = new_size;

  init_ring (ring);

  compact_phases (ring, old_size, new_size, map);
  compact_heap (ring, &ring->heap, old_size, new_size, map);
  compact_queue (ring, &ring->queue, old_size, new_size, map);

  assert (SIZE (ring->watchers) == 1);
  compact_saved (ring, map, mapped);
  ring->size = new_size;
  ring->statistics.active = new_size;

  ring->ruler_units = ruler->units.end;
}

static void compact_rings (struct ruler *ruler, unsigned *map) {
  struct clauses mapped_clauses;
  INIT (mapped_clauses);

  for (all_rings (ring))
    compact_ring (ring, map, &mapped_clauses);

  for (all_clauses (clause, mapped_clauses))
    assert (clause->mapped), clause->mapped = false;

  very_verbose (0, "mapped %zu clauses in total", SIZE (mapped_clauses));
  RELEASE (mapped_clauses);
}

/*------------------------------------------------------------------------*/

static void compact_bool_array (bool **array_ptr, unsigned old_size,
                                unsigned new_size, unsigned *map) {
  bool *old_array = *array_ptr;
  bool *new_array = *array_ptr = allocate_block (new_size);
  bool *p = old_array, *n = new_array, *end = old_array + old_size;
  unsigned old_idx = 0;
  while (p != end) {
    assert (p == old_array + old_idx);
    bool value = *p++;
    unsigned new_idx = map[old_idx++];
    if (new_idx == INVALID)
      continue;
    assert (n == new_array + new_idx);
    *n++ = value;
  }
  assert (n == new_array + new_size);
  free (old_array);
}

void compact_ruler (struct simplifier *simplifier, bool initially) {
  struct ruler *ruler = simplifier->ruler;
  if (ruler->inconsistent || ruler->terminate)
    return;

  if (!initially)
    clean_rings (ruler);

  signed char *values = (signed char *) ruler->values;
  bool *eliminated = simplifier->eliminated;
  unsigned new_compact = 0;
  for (all_ruler_indices (idx)) {
    if (eliminated[idx])
      continue;
    unsigned lit = LIT (idx);
    if (values[lit])
      continue;
    new_compact++;
  }

  unsigned *unmap = allocate_array (new_compact, sizeof *unmap);
  unsigned *old_unmap = ruler->unmap;

  unsigned old_compact = ruler->compact;
  unsigned *map = allocate_array (old_compact, sizeof *map);
  unsigned mapped = 0;

  ROG ("reducing compact size from %u to %u (original %u)", old_compact,
       new_compact, ruler->size);

  for (all_ruler_indices (idx)) {
    unsigned lit = LIT (idx);
    if (eliminated[idx]) {
      map[idx] = INVALID;
#ifdef LOGGING
      if (old_unmap)
        ROG ("skipping eliminated variable %u (literal %u) "
             "which was original variable %u (literal %u)",
             idx, lit, old_unmap[idx], LIT (old_unmap[idx]));
      else
        ROG ("skipping eliminated original variable %u (literal %u)", idx,
             lit);
#endif
      continue;
    }
    if (values[lit]) {
      map[idx] = INVALID;
#ifdef LOGGING
      if (old_unmap)
        ROG ("skipping assigned variable %u (literal %u) "
             "which was original variable %u (literal %u)",
             idx, lit, old_unmap[idx], LIT (old_unmap[idx]));
      else
        ROG ("skipping assigned original variable %u (literal %u)", idx,
             lit);
#endif
      continue;
    }
    unsigned old_idx = old_unmap ? old_unmap[idx] : idx;
    unmap[mapped] = old_idx;
    map[idx] = mapped;
#ifdef LOGGING
    if (old_unmap)
      ROG ("mapping variable %u (literal %u) which was originally "
           "variable %u (literal %u) to variable %u (literal %u)",
           idx, lit, old_idx, LIT (old_idx), mapped, LIT (mapped));
    else
      ROG ("mapping original variable %u (literal %u) "
           "to variable %u (literal %u)",
           idx, lit, mapped, LIT (mapped));
#endif
    mapped++;
  }
  SHRINK_STACK (ruler->extension[0]);
  for (all_ruler_indices (idx)) {
    unsigned lit = LIT (idx);
    unsigned not_lit = NOT (lit);
    if (!eliminated[idx] && !values[lit]) {
      assert (map[idx] <= idx);
      map_occurrences (ruler, map, lit);
      map_occurrences (ruler, map, not_lit);
    } else {
      assert (map[idx] == INVALID);
      RELEASE (OCCURRENCES (lit));
      RELEASE (OCCURRENCES (not_lit));
    }
  }
  assert (new_compact == mapped);
  ruler->compact = new_compact;

  map_clauses (ruler, map);

  compact_bool_array (&ruler->eliminate, old_compact, new_compact, map);
  compact_bool_array (&ruler->subsume, old_compact, new_compact, map);

  free (ruler->units.begin);
  ruler->units.begin = allocate_array (new_compact, sizeof (unsigned));
  ruler->units.propagate = ruler->units.end = ruler->units.begin;

  if (!initially)
    compact_rings (ruler, map);

  free (map);
  if (old_unmap)
    free (old_unmap);

  ruler->unmap = unmap;
  ruler->trace.unmap = unmap;
  for (all_rings (ring))
    ring->trace.unmap = unmap;

  free ((void *) ruler->values);
  ruler->values = allocate_and_clear_block (2 * new_compact);

  verbose (0, "mapped %u variables to %u variables", ruler->size, mapped);
}

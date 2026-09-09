#include "witness.h"
#include "decide.h"
#include "message.h"
#include "ruler.h"
#include "utilities.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

signed char *extend_witness (struct ring *ring) {
  struct ruler *ruler = ring->ruler;
  LOG ("extending witness from %u to %u variables", ring->size,
       ruler->size);
  signed char *witness = allocate_array (2 * ruler->size, sizeof *witness);
  signed char *values = ring->values;
  assert (ring->size == ruler->compact);
  signed char phase = initial_phase (ring);
  LOG ("setting all %u original variables to %d", ruler->size, phase);
  for (unsigned idx = 0; idx != ruler->size; idx++) {
    unsigned lit = LIT (idx);
    unsigned not_lit = NOT (lit);
    witness[lit] = phase;
    witness[not_lit] = -phase;
    LOG ("setting original literal %u(%d)=%d to default phase", lit,
         only_export_literal (lit), (int) phase);
  }
  unsigned *unmap = ruler->unmap;
  ruler->unmap = 0;
  LOG ("unmapping and assigning %u ring variables", ring->size);
  for (unsigned ring_idx = 0; ring_idx != ring->size; ring_idx++) {
    unsigned ring_lit = LIT (ring_idx);
    signed char value = values[ring_lit];
    unsigned ruler_idx = unmap[ring_idx];
    unsigned ruler_lit = LIT (ruler_idx);
    unsigned not_ruler_lit = NOT (ruler_lit);
    witness[ruler_lit] = value;
    witness[not_ruler_lit] = -value;
#ifdef LOGGING
    int exported = only_export_literal (ruler_lit);
    LOG ("assigning original literal %u(%d)=%d "
         "to value of ring literal %u(%d)=%d",
         ruler_lit, exported, (int) value, ring_lit, exported, (int) value);
#endif
  }
  free (unmap);
  LOG ("forcing %zu saved units", SIZE (ruler->extension[1]));
  for (all_elements_on_stack (unsigned, lit, ruler->extension[1])) {
    unsigned not_lit = NOT (lit);
    witness[lit] = 1;
    witness[not_lit] = -1;
    LOG ("forcing original literal %u(%d)=1 as saved unit", lit,
         only_export_literal (lit));
  }
#ifndef QUIET
  size_t flipped = 0;
#endif
  struct unsigneds *extension = &ruler->extension[0];
  unsigned *begin = extension->begin;
  unsigned *p = extension->end;
  unsigned pivot = INVALID;
  bool satisfied = false;
  LOG ("going through extension stack of size %zu", (size_t) (p - begin));
#ifdef LOGGING
  size_t clauses = 0;
  if (verbosity == INT_MAX) {
    {
      for (unsigned *q = begin; q != p; q++)
        if (*q == INVALID)
          clauses++;
    }
    {
      LOG ("printing extension stack of size %zu with %zu clauses",
           SIZE (*extension), clauses);
      unsigned *q = p;
      size_t clause_idx = clauses;
      while (q != begin) {
        unsigned *next = q;
        while (*--next != INVALID)
          ;
        LOGPREFIX ("extension clause weakened[%zu]", clause_idx);
        for (unsigned *c = next + 1; c != q; c++) {
          unsigned lit = *c;
          int value = witness[lit];
          printf (" %u(%d)=%d", lit, only_export_literal (lit), value);
        }
        LOGSUFFIX ();
        clause_idx--;
        q = next;
      }
    }
  }
#endif
  while (p != begin) {
    unsigned lit = *--p;
    if (lit == INVALID) {
      if (!satisfied) {
#ifdef LOGGING
        int exported = only_export_literal (pivot);
        LOG ("flipping %u(%d)=-1 to %u(%d)=1 due to clause[%zu]", pivot,
             exported, pivot, exported, clauses);
#endif
        assert (pivot != INVALID);
        unsigned not_pivot = NOT (pivot);
        assert (witness[pivot] < 0);
        assert (witness[not_pivot] > 0);
        witness[pivot] = 1;
        witness[not_pivot] = -1;
#ifndef QUIET
        flipped++;
#endif
      }
      satisfied = false;
#ifdef LOGGING
      clauses--;
#endif
    } else if (!satisfied) {
      signed char value = witness[lit];
      if (value > 0)
        satisfied = true;
    }
    pivot = lit;
  }
  verbose (ring, "flipped %zu literals", flipped);
  return witness;
}

#ifndef NDEBUG

void check_witness (signed char *values, struct unsigneds *original) {
  size_t clauses = 0;
  for (unsigned *c = original->begin, *p; c != original->end; c = p + 1) {
    bool satisfied = false;
    for (p = c; assert (p != original->end), *p != INVALID; p++)
      if (values[*p] > 0)
        satisfied = true;
    clauses++;
    if (satisfied)
      continue;
    acquire_message_lock ();
    fprintf (stderr, "gimsatul: error: unsatisfied clause[%zu]", clauses);
    for (unsigned *q = c; q != p; q++)
      fprintf (stderr, " %d", only_export_literal (*q));
    fputs (" 0\n", stderr);
    release_message_lock ();
    abort ();
  }
}

#endif

struct line {
  char buffer[80];
  size_t size;
};

static void flush_line (struct line *line) {
  fwrite (line->buffer, 1, line->size, stdout);
  fputc ('\n', stdout);
  line->size = 0;
}

static void print_signed_literal (struct line *line, int lit) {
  char buffer[32];
  sprintf (buffer, " %d", lit);
  size_t len = strlen (buffer);
  if (line->size + len >= sizeof line->buffer)
    flush_line (line);
  if (!line->size)
    line->buffer[line->size++] = 'v';
  memcpy (line->buffer + line->size, buffer, len);
  line->size += len;
}

static void print_unsigned_literal (struct line *line, signed char *values,
                                    unsigned unsigned_lit) {
  assert (unsigned_lit < (unsigned) INT_MAX);
  int signed_lit = IDX (unsigned_lit) + 1;
  signed_lit *= values[unsigned_lit];
  print_signed_literal (line, signed_lit);
}

void print_witness (unsigned size, signed char *values) {
  struct line line;
  line.size = 0;
  for (unsigned idx = 0; idx != size; idx++)
    print_unsigned_literal (&line, values, LIT (idx));
  print_signed_literal (&line, 0);
  if (line.size)
    flush_line (&line);
}

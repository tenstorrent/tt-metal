#include "trace.h"
#include "message.h"
#include "stack.h"
#include "utilities.h"

#include <assert.h>
#include <inttypes.h>

static void binary_proof_line (struct trace *trace, size_t size,
                               unsigned *literals, unsigned except) {
  const unsigned *end = literals + size;
  unsigned *unmap = trace->unmap;
  for (const unsigned *p = literals; p != end; p++) {
    unsigned lit = *p;
    if (lit == except)
      continue;
    unsigned unmapped = unmap_literal (unmap, lit);
    unsigned tmp = unmapped + 2;
    while (tmp & ~127u) {
      unsigned char ch = (tmp & 0x7f) | 128;
      PUSH (trace->buffer, ch);
      tmp >>= 7;
    }
    unsigned char ch = tmp;
    PUSH (trace->buffer, ch);
  }
  PUSH (trace->buffer, 0);
}

static void ascii_proof_line (struct trace *trace, size_t size,
                              unsigned *literals, unsigned except) {
  const unsigned *end = literals + size;
  unsigned *unmap = trace->unmap;
  char tmp[32];
  for (const unsigned *p = literals; p != end; p++) {
    unsigned lit = *p;
    if (lit == except)
      continue;
    int unmapped = unmap_and_export_literal (unmap, lit);
    sprintf (tmp, "%d", unmapped);
    for (char *q = tmp, ch; (ch = *q); q++)
      PUSH (trace->buffer, ch);
    PUSH (trace->buffer, ' ');
  }
  PUSH (trace->buffer, '0');
  PUSH (trace->buffer, '\n');
}

void trace_add_literals (struct trace *trace, size_t size,
                         unsigned *literals, unsigned except) {
  if (!trace->file)
    return;
  assert (EMPTY (trace->buffer));
  if (trace->binary) {
    PUSH (trace->buffer, 'a');
    binary_proof_line (trace, size, literals, except);
  } else
    ascii_proof_line (trace, size, literals, except);
  write_buffer (&trace->buffer, trace->file);
}

void trace_add_empty (struct trace *trace) {
  if (!trace->file)
    return;
  trace_add_literals (trace, 0, 0, INVALID);
}

void trace_add_unit (struct trace *trace, unsigned unit) {
  if (!trace->file)
    return;
  trace_add_literals (trace, 1, &unit, INVALID);
}

void trace_add_binary (struct trace *trace, unsigned lit, unsigned other) {
  if (!trace->file)
    return;
  unsigned literals[2] = {lit, other};
  trace_add_literals (trace, 2, literals, INVALID);
}

void trace_delete_literals (struct trace *trace, size_t size,
                            unsigned *literals) {
  if (!trace->file)
    return;
  assert (EMPTY (trace->buffer));
  PUSH (trace->buffer, 'd');
  if (trace->binary)
    binary_proof_line (trace, size, literals, INVALID);
  else {
    PUSH (trace->buffer, ' ');
    ascii_proof_line (trace, size, literals, INVALID);
  }
  write_buffer (&trace->buffer, trace->file);
}

void trace_delete_binary (struct trace *trace, unsigned lit,
                          unsigned other) {
  if (!trace->file)
    return;
  unsigned literals[2] = {lit, other};
  trace_delete_literals (trace, 2, literals);
}

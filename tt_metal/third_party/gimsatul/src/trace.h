#ifndef _trace_h_INCLUDED
#define _trace_h_INCLUDED

#include "stack.h"

#include <stdbool.h>

struct file;

struct trace {
  bool binary;
  struct file *file;
  struct buffer buffer;
  unsigned *unmap;
};

void trace_add_empty (struct trace *);
void trace_add_unit (struct trace *, unsigned unit);
void trace_add_binary (struct trace *, unsigned, unsigned);
void trace_add_literals (struct trace *, size_t, unsigned *,
                         unsigned except);

void trace_delete_literals (struct trace *, size_t, unsigned *);
void trace_delete_binary (struct trace *, unsigned, unsigned);

struct clause;
void trace_add_clause (struct trace *, struct clause *);
void trace_delete_clause (struct trace *, struct clause *);

#endif

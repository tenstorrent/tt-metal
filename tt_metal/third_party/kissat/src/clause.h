#ifndef _clause_h_INCLUDED
#define _clause_h_INCLUDED

#include "arena.h"
#include "literal.h"
#include "reference.h"
#include "utilities.h"

#include <stdbool.h>

typedef struct clause clause;

#define LD_MAX_GLUE 21u
#define MAX_GLUE ((1u<<LD_MAX_GLUE)-1)

struct clause {
unsigned glue:
  LD_MAX_GLUE;

  bool garbage: 1;
  bool hyper: 1;
  bool keep: 1;
  bool reason: 1;
  bool redundant: 1;
  bool shrunken: 1;
  bool subsume: 1;
  bool sweeped: 1;
  bool vivify: 1;

  unsigned used: 2;

  unsigned searched;
  unsigned size;

  unsigned lits[3];
};

#define SIZE_OF_CLAUSE_HEADER ((size_t) &((clause*)0)->searched)

#define BEGIN_LITS(C) ((C)->lits)
#define END_LITS(C) (BEGIN_LITS (C) + (C)->size)

#define all_literals_in_clause(LIT,C) \
  unsigned LIT, * LIT ## _PTR = BEGIN_LITS (C), \
                * const LIT ## _END = END_LITS (C); \
  LIT ## _PTR != LIT ## _END && ((LIT = *LIT ## _PTR), true); \
  ++LIT ## _PTR

static inline size_t kissat_bytes_of_clause(unsigned size) {
  const size_t res = sizeof(clause) + (size - 3) * sizeof(unsigned);
  return kissat_align_ward(res);
}

static inline size_t kissat_actual_bytes_of_clause(clause *c) {
  unsigned const *p = END_LITS(c);
  if (c->shrunken)
    while (*p++ != INVALID_LIT)
      ;
  return kissat_align_ward((char *) p - (char *) c);
}

static inline clause *kissat_next_clause(clause *c) {
  word bytes = kissat_actual_bytes_of_clause(c);
  return (clause *)((char *) c + bytes);
}

struct kissat;

void kissat_new_binary_clause(struct kissat *,
      bool redundant, unsigned, unsigned);

reference kissat_new_original_clause(struct kissat *);
reference kissat_new_irredundant_clause(struct kissat *);
reference kissat_new_redundant_clause(struct kissat *, unsigned glue);

#ifndef INLINE_SORT
void kissat_sort_literals(struct kissat *, unsigned size, unsigned *lits);
#endif

void kissat_connect_clause(struct kissat *, clause *);

clause *kissat_delete_clause(struct kissat *, clause *);
void kissat_delete_binary(struct kissat *,
      bool redundant, bool hyper, bool weakened, unsigned, unsigned);

void kissat_mark_clause_as_garbage(struct kissat *, bool weakened, clause *);

#endif

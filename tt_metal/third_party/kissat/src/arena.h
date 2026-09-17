#ifndef _arena_h_INCLUDED
#define _arena_h_INCLUDED

#include "reference.h"
#include "stack.h"
#include "utilities.h"

#ifdef COMPACT
typedef word ward;
#else
typedef w2rd ward;
#endif

#define LD_MAX_ARENA_32 (29 -  (unsigned) sizeof (ward)/4)

#define LD_MAX_ARENA \
  ((sizeof (word) == 4) ? LD_MAX_ARENA_32 : LD_MAX_REF)

#define MAX_ARENA ((size_t)1 << LD_MAX_ARENA)

// *INDENT-OFF*

typedef STACK (ward) arena;

// *INDENT-ON*

struct clause;
struct kissat;

reference kissat_allocate_clause(struct kissat *, size_t size);
void kissat_shrink_arena(struct kissat *);

#if !defined(NDEBUG) || defined(LOGGING)

bool kissat_clause_in_arena(const struct kissat *, const struct clause *);

#endif

static inline word kissat_align_ward(word w) {
#ifdef COMPACT
  return kissat_align_word(w);
#else
  return kissat_align_w2rd(w);
#endif
}

#endif

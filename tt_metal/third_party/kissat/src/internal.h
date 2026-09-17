#ifndef _internal_h_INCLUDED
#define _internal_h_INCLUDED

#include "arena.h"
#include "array.h"
#include "assign.h"
#include "averages.h"
#include "cache.h"
#include "check.h"
#include "clause.h"
#include "clueue.h"
#include "cover.h"
#include "extend.h"
#include "smooth.h"
#include "flags.h"
#include "format.h"
#include "frames.h"
#include "heap.h"
#include "kissat.h"
#include "limits.h"
#include "literal.h"
#include "mode.h"
#include "nonces.h"
#include "options.h"
#include "phases.h"
#include "profile.h"
#include "proof.h"
#include "queue.h"
#include "random.h"
#include "reap.h"
#include "reluctant.h"
#include "rephase.h"
#include "stack.h"
#include "statistics.h"
#include "literal.h"
#include "value.h"
#include "vector.h"
#include "watch.h"

typedef struct datarank datarank;

struct datarank {
  unsigned data;
  unsigned rank;
};

typedef struct import import;

struct import {
  unsigned lit: 30;
  bool imported: 1;
  bool eliminated: 1;
};

#define KISSAT_GET_IMPORT_ELIT(IMPORT) \
  (((int)((IMPORT)->lit << 2)) >> 2)
#define KISSAT_SET_IMPORT_ELIT(IMPORT, ELIT) \
  (IMPORT)->lit = (assert(ABS(ELIT) < (1 << 29)), \
    (((unsigned)(ELIT)) & ((1u << 30) - 1)))

typedef struct termination termination;

struct termination {
#ifdef COVERAGE
  volatile uint64_t flagged;
#else
  volatile bool flagged;
#endif
  volatile void *state;
  int (*volatile terminate)(void *);
};

typedef struct search_start search_start;

struct search_start {
#define SAVE(X) uint64_t X;
  SAVE_AT_SEARCH_START
#undef SAVE
};

// *INDENT-OFF*

typedef STACK (value) eliminated;
typedef STACK (import) imports;
typedef STACK (datarank) dataranks;
typedef STACK (watch) statches;
typedef STACK (watch *) patches;

// *INDENT-ON*

struct kitten;

struct kissat {
#if !defined(NDEBUG) || defined(METRICS)
  bool backbone_computing;
#endif
#ifdef LOGGING
  bool compacting;
#endif
  bool extended;
#if !defined(NDEBUG) || defined(METRICS)
  bool failed_probing;
#endif
  bool inconsistent;
  bool iterating;
  bool probing;
#ifndef QUIET
  bool sectioned;
#endif
  bool stable;
#if !defined(NDEBUG) || defined(METRICS)
  bool transitive_reducing;
  bool vivifying;
#endif
  bool watching;

  bool large_clauses_watched_after_binary_clauses;

  termination termination;

  unsigned vars;
  unsigned size;
  unsigned active;

  ints export;
  ints units;
  imports import;
  extensions extend;
  unsigneds witness;
  bool restore;

  assigned *assigned;
  flags *flags;
  unsigned *protect;

  mark *marks;

  value *values;
  phases phases;
  cache cache;
  nonces nonces;

  eliminated eliminated;
  unsigneds etrail;

  links *links;
  queue queue;

  rephased rephased;

  heap scores;
  double scinc;

  heap schedule;

  unsigned level;
  frames frames;

  unsigned_array trail;
  unsigned *propagate;

  ints assumptions;
  ints redundant_assumptions;
  ints failed;
  bool failed_early;
  unsigneds assumption_implied;

  unsigned best_assigned;
  unsigned target_assigned;
  unsigned unflushed;
  unsigned unassigned;

  unsigneds delayed;

#if defined(LOGGING) || !defined(NDEBUG)
  unsigneds resolvent;
#endif
  unsigned resolvent_size;
  unsigned antecedent_size;

  dataranks ranks;

  unsigneds analyzed;
  unsigneds levels;
  unsigneds minimize;
  unsigneds poisoned;
  unsigneds promote;
  unsigneds removable;
  unsigneds shrinkable;

  reap reap;

  clause conflict;

  bool clause_satisfied;
  bool clause_shrink;
  bool clause_trivial;
  bool clause_elits_substituted;

  unsigneds clause;
  unsigneds shadow;

  arena arena;
  clueue clueue;
  vectors vectors;
  reference first_reducible;
  reference last_irredundant;
  watches *watches;

  sizes sorter;

  generator random;
  averages averages[2];
  reluctant reluctant;

  bounds bounds;
  delays delays;
  enabled enabled;
  effort last;
  limited limited;
  limits limits;
  waiting waiting;
  unsigned walked;

  statistics statistics;
  search_start search_start;
  mode mode;

  uint64_t ticks;

  format format;

  statches antecedents[2];
  statches gates[2];
  patches xorted[2];
  unsigneds resolvents;
  bool resolve_gate;
  unsigned found_equivalence;

  struct kitten *kitten;
#ifdef METRICS
  uint64_t *gate_eliminated;
#else
  bool gate_eliminated;
#endif

#if !defined(NDEBUG) || !defined(NPROOFS)
  unsigneds added;
  unsigneds removed;
#endif

#if !defined(NDEBUG) || !defined(NPROOFS) || defined(LOGGING)
  ints original;
  size_t offset_of_last_original_clause;
#endif

#ifndef QUIET
  profiles profiles;
#endif

#ifndef NOPTIONS
  options options;
#endif

#ifndef NDEBUG
  checker *checker;
#endif

#ifndef NPROOFS
  proof *proof;
#endif
};

#define VARS (solver->vars)
#define LITS (2*solver->vars)

static inline unsigned kissat_assigned(kissat *solver) {
  assert(VARS >= solver->unassigned);
  return VARS - solver->unassigned;
}

#define all_variables(IDX) \
  unsigned IDX = 0, IDX ## _END = solver->vars; \
  IDX != IDX ## _END; \
  ++IDX

#define all_literals(LIT) \
  unsigned LIT = 0, LIT ## _END = LITS; \
  LIT != LIT ## _END; \
  ++LIT

#define all_clauses(C) \
  clause *       C         = (clause*) BEGIN_STACK (solver->arena), \
         * const C ## _END = (clause*) END_STACK (solver->arena), \
	 * C ## _NEXT; \
  C != C ## _END && (C ## _NEXT = kissat_next_clause (C), true); \
  C = C ## _NEXT

#endif

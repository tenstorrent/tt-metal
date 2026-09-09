#ifndef _ring_h_INCLUDED
#define _ring_h_INCLUDED

#include "average.h"
#include "clause.h"
#include "heap.h"
#include "logging.h"
#include "macros.h"
#include "options.h"
#include "profile.h"
#include "queue.h"
#include "stack.h"
#include "statistics.h"
#include "tagging.h"
#include "trace.h"
#include "variable.h"
#include "watches.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

struct ruler;

struct reluctant {
  uint64_t u, v;
};

struct ring_limits {
  uint64_t mode;
  uint64_t randec;
  uint64_t reduce;
  uint64_t rephase;
  uint64_t restart;
  uint64_t simplify;
  uint64_t tiers;
  long long conflicts;
  struct {
    uint64_t conflicts;
    uint64_t reductions;
  } probe;
};

struct intervals {
  uint64_t mode;
  uint64_t tiers;
};

struct averages {
  struct {
    struct average fast;
    struct average slow;
  } glue;
  struct average level;
  struct average trail;
  struct average decisions;
  struct average size;
};

struct ring_last {
  uint64_t decisions;
  unsigned fixed;
  uint64_t probing;
  uint64_t walk;
#ifndef QUIET
  struct mode {
    uint64_t conflicts;
    uint64_t ticks;
    double time;
  } mode;
#endif
};

struct ring_delay {
  struct {
    uint64_t count, current;
  } bump_reason;
};

struct ring_trail {
  unsigned *begin, *end;
  unsigned *pos, *propagate;
};

struct ring_units {
  unsigned *begin, *end;
  unsigned *iterate, *export;
};

#define MAX_REDUNDANCY (~(uint64_t) 0)

#ifdef LOGGING
#define LOG_REDUNDANCY(R) (unsigned) ((R) >> 32), (unsigned) (R)
#endif

#define BINARY_BUCKET 0
#define SIZE_POOL 8

struct bucket {
  uint64_t redundancy;
  atomic_uintptr_t shared;
};

struct pool {
  struct bucket bucket[SIZE_POOL];
};

struct ring;

struct rings {
  struct ring **begin, **end, **allocated;
};

struct ring {
  unsigned id;
  unsigned threads;
  struct pool *pool;
  unsigned *ruler_units;
  struct ruler *ruler;

  volatile int status;

  bool import_after_propagation_and_conflict;
  bool inconsistent;
  bool stable;

  signed char iterating;

  unsigned best;
  unsigned context;
  unsigned level;
  unsigned probe;
  unsigned size;
  unsigned target;
  unsigned unassigned;

  unsigned tier1_glue_limit[2];
  unsigned tier2_glue_limit[2];

  signed char *marks;
  signed char *values;

  bool *inactive;
  unsigned char *used;

  struct unsigneds analyzed;
  struct unsigneds clause;
  struct unsigneds levels;
  struct unsigneds minimize;
  struct unsigneds sorter;
  struct unsigneds outoforder;
  struct unsigneds promote;
  struct rings exports;

  struct references *references;
  struct ring_trail trail;
  struct ring_units ring_units;
  struct variable *variables;

  struct heap heap;
  struct phases *phases;
  struct queue queue;

  unsigned redundant;
  struct watchers watchers;
  unsigned last_learned[4];
  struct saved_watchers saved;

  struct trace trace;

  struct averages averages[2];
  struct intervals intervals;
  struct ring_last last;
  struct ring_delay delay;
  struct ring_limits limits;
  struct options options;
  struct reluctant reluctant;
  struct ring_profiles profiles;
  struct ring_statistics statistics;

  unsigned randec;
  uint64_t random;
};

/*------------------------------------------------------------------------*/

#define VAR(LIT) (ring->variables + IDX (LIT))
#define REFERENCES(LIT) (ring->references[LIT])

/*------------------------------------------------------------------------*/

#define all_ring_indices(IDX) \
  unsigned IDX = 0, END_##IDX = ring->size; \
  IDX != END_##IDX; \
  ++IDX

#define all_ring_literals(LIT) \
  unsigned LIT = 0, END_##LIT = 2 * ring->size; \
  LIT != END_##LIT; \
  ++LIT

#define all_phases(PHASE) \
  struct phases *PHASE = ring->phases, *END_##PHASE = PHASE + ring->size; \
  (PHASE != END_##PHASE); \
  ++PHASE

#define all_averages(AVG) \
  struct average *AVG = (struct average *) &ring->averages, \
                 *END_##AVG = (struct average *) ((char *) AVG + \
                                                  sizeof ring->averages); \
  AVG != END_##AVG; \
  ++AVG

#define all_watchers(WATCHER) \
  struct watcher *WATCHER = ring->watchers.begin + 1, \
                 *END_##WATCHER = ring->watchers.end; \
  (WATCHER != END_##WATCHER); \
  ++WATCHER

#define all_redundant_watchers(WATCHER) \
  struct watcher *WATCHER = ring->watchers.begin + ring->redundant, \
                 *END_##WATCHER = ring->watchers.end; \
  (WATCHER != END_##WATCHER); \
  ++WATCHER

#define capacity_last_learned \
  (sizeof ring->last_learned / sizeof *ring->last_learned)

#define real_end_last_learned (ring->last_learned + capacity_last_learned)

#define really_all_last_learned(IDX_PTR) \
  unsigned *IDX_PTR = ring->last_learned, \
           *IDX_PTR##_END = real_end_last_learned; \
  IDX_PTR != IDX_PTR##_END; \
  IDX_PTR++

/*------------------------------------------------------------------------*/

void init_ring (struct ring *);
void release_ring (struct ring *, bool keep_values);

struct ring *new_ring (struct ruler *);
void delete_ring (struct ring *);

void init_pool (struct ring *, unsigned threads);

void reset_last_learned (struct ring *);

void mark_satisfied_watchers_as_garbage (struct ring *);

void inc_clauses (struct ring *ring, bool redundant);
void dec_clauses (struct ring *ring, bool redundant);

void set_inconsistent (struct ring *, const char *msg);
void set_satisfied (struct ring *);

void print_ring_profiles (struct ring *);

unsigned *sorter_block (struct ring *, size_t size);

struct ring *random_other_ring (struct ring *);

/*------------------------------------------------------------------------*/

static inline unsigned watcher_to_index (struct ring *ring,
                                         struct watcher *watcher) {
  assert (ring->watchers.begin <= watcher);
  assert (watcher < ring->watchers.end);
  size_t idx = watcher - ring->watchers.begin;
  assert (idx <= MAX_WATCHER_INDEX);
  assert (idx <= UINT_MAX);
  return idx;
}

static inline struct watcher *index_to_watcher (struct ring *ring,
                                                unsigned idx) {
  return &PEEK (ring->watchers, idx);
}

static inline struct watcher *get_watcher (struct ring *ring,
                                           struct watch *watch) {
  assert (!is_binary_pointer (watch));
  size_t idx = index_pointer (watch);
  return &PEEK (ring->watchers, idx);
}

static inline struct clause *get_clause (struct ring *ring,
                                         struct watch *watch) {
  assert (watch);
  return get_watcher (ring, watch)->clause;
}

/*------------------------------------------------------------------------*/

static inline void push_watch (struct ring *ring, unsigned lit,
                               struct watch *watch) {
  LOGWATCH (watch, "watching %s in", LOGLIT (lit));
  PUSH (REFERENCES (lit), watch);
}

static inline void watch_literal (struct ring *ring, unsigned lit,
                                  unsigned other, struct watcher *watcher) {
  unsigned idx = watcher_to_index (ring, watcher);
  struct watch *watch = tag_index (watcher->redundant, idx, other);
  push_watch (ring, lit, watch);
}

/*------------------------------------------------------------------------*/

#endif

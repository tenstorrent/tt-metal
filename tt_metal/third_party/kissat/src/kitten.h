#ifndef _kitten_h_INCLUDED
#define _kitten_h_INCLUDED

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct kitten kitten;

kitten *kitten_init(void);
void kitten_clear(kitten *);
void kitten_release(kitten *);

void kitten_track_antecedents(kitten *);
void kitten_shuffle_clauses(kitten *);
void kitten_randomize_phases(kitten *);
void kitten_flip_phases(kitten *);

void kitten_assume(kitten *, unsigned lit);

void kitten_clause(kitten *, size_t size, unsigned *);
void kitten_unit(kitten *, unsigned);
void kitten_binary(kitten *, unsigned, unsigned);

void kitten_clause_with_id_and_exception(kitten *, unsigned id,
      size_t size, const unsigned *,
      unsigned except);


void kitten_set_ticks_limit(kitten *, uint64_t);

int kitten_solve(kitten *);

signed char kitten_value(kitten *, unsigned);
bool kitten_failed(kitten *, unsigned);

unsigned kitten_compute_clausal_core(kitten *, uint64_t *learned);
void kitten_shrink_to_clausal_core(kitten *);

void kitten_traverse_core_ids(kitten *, void *state,
      void (*traverse)(void *state, unsigned id));

void kitten_traverse_core_clauses(kitten *, void *state,
      void (*traverse)(void *state,
            bool learned, size_t,
            const unsigned *));
struct kissat;
kitten *kitten_embedded(struct kissat *);

#endif

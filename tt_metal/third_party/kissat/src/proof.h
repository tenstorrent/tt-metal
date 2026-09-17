#ifndef _proof_h_INCLUDED
#define _proof_h_INCLUDED

#ifndef NPROOFS

#include <stdbool.h>
#include <stdlib.h>

typedef struct kissat kissat;
typedef struct proof proof;

struct clause;
struct file;

void kissat_init_proof(struct kissat *, struct file *, bool binary);
void kissat_release_proof(struct kissat *);

#ifndef QUIET
void kissat_print_proof_statistics(struct kissat *, bool verbose);
#endif

void kissat_add_binary_to_proof(struct kissat *, unsigned, unsigned);
void kissat_add_clause_to_proof(struct kissat *, const struct clause *c);
void kissat_add_empty_to_proof(struct kissat *);
void kissat_add_lits_to_proof(struct kissat *, size_t, const unsigned *);
void kissat_add_unit_to_proof(struct kissat *, unsigned);

void kissat_shrink_clause_in_proof(struct kissat *, const struct clause *,
      unsigned remove, unsigned keep);

void kissat_delete_binary_from_proof(struct kissat *, unsigned, unsigned);
void kissat_delete_clause_from_proof(struct kissat *,
      const struct clause *c);
void kissat_delete_external_from_proof(struct kissat *, size_t, const int *);
void kissat_delete_internal_from_proof(struct kissat *, size_t,
      const unsigned *);

#define ADD_BINARY_TO_PROOF(A,B) \
do { \
  if (solver->proof) \
    kissat_add_binary_to_proof (solver, (A), (B)); \
} while (0)

#define ADD_CLAUSE_TO_PROOF(CLAUSE) \
do { \
  if (solver->proof) \
    kissat_add_clause_to_proof (solver, (CLAUSE)); \
} while (0)

#define ADD_EMPTY_TO_PROOF() \
do { \
  if (solver->proof) \
    kissat_add_empty_to_proof (solver); \
} while (0)

#define ADD_LITS_TO_PROOF(SIZE,LITS) \
do { \
  if (solver->proof) \
    kissat_add_lits_to_proof (solver, (SIZE), (LITS)); \
} while (0)

#define ADD_STACK_TO_PROOF(S) \
  ADD_LITS_TO_PROOF (SIZE_STACK (S), BEGIN_STACK (S))

#define ADD_UNIT_TO_PROOF(A) \
do { \
  if (solver->proof) \
    kissat_add_unit_to_proof (solver, (A)); \
} while (0)

#define SHRINK_CLAUSE_IN_PROOF(C,REMOVE,KEEP) \
do { \
  if (solver->proof) \
    kissat_shrink_clause_in_proof (solver, (C), (REMOVE), (KEEP)); \
} while (0)

#define DELETE_BINARY_FROM_PROOF(A,B) \
do { \
  if (solver->proof) \
    kissat_delete_binary_from_proof (solver, (A), (B)); \
} while (0)

#define DELETE_CLAUSE_FROM_PROOF(CLAUSE) \
do { \
  if (solver->proof) \
    kissat_delete_clause_from_proof (solver, (CLAUSE)); \
} while (0)

#define DELETE_LITS_FROM_PROOF(SIZE,LITS)\
do { \
  if (solver->proof) \
    kissat_delete_internal_from_proof (solver, (SIZE), (LITS)); \
} while (0)

#define DELETE_STACK_FROM_PROOF(S)\
do { \
  if (solver->proof) \
    kissat_delete_internal_from_proof (solver, \
                                       SIZE_STACK (S), BEGIN_STACK (S)); \
} while (0)

#else

#define ADD_BINARY_TO_PROOF(...) do { } while (0)
#define ADD_CLAUSE_TO_PROOF(...) do { } while (0)
#define ADD_LITS_TO_PROOF(...) do { } while (0)
#define ADD_EMPTY_TO_PROOF(...) do { } while (0)
#define ADD_STACK_TO_PROOF(...) do { } while (0)
#define ADD_UNIT_TO_PROOF(...) do { } while (0)

#define SHRINK_CLAUSE_IN_PROOF(...) do { } while (0)

#define DELETE_BINARY_FROM_PROOF(...) do { } while (0)
#define DELETE_CLAUSE_FROM_PROOF(...) do { } while (0)
#define DELETE_LITS_FROM_PROOF(...) do { } while (0)
#define DELETE_STACK_FROM_PROOF(...) do { } while (0)

#endif

#endif

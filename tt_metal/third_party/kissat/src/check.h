#ifndef _check_h_INCLUDED
#define _check_h_INCLUDED

#ifndef NDEBUG

#include <stdbool.h>
#include <stdlib.h>

struct kissat;

void kissat_check_satisfying_assignment(struct kissat *);

typedef struct checker checker;

struct clause;

void kissat_init_checker(struct kissat *);
void kissat_release_checker(struct kissat *);

#ifndef QUIET
void kissat_print_checker_statistics(struct kissat *, bool verbose);
#endif

void kissat_add_unchecked_external(struct kissat *, size_t, const int *);

void kissat_check_and_add_binary(struct kissat *, unsigned, unsigned);
void kissat_check_and_add_clause(struct kissat *, struct clause *c);
void kissat_check_and_add_empty(struct kissat *);
void kissat_check_and_add_internal(struct kissat *,
      size_t, const unsigned *);
void kissat_check_and_add_unit(struct kissat *, unsigned);

void kissat_check_shrink_clause(struct kissat *, struct clause *,
      unsigned remove, unsigned keep);

void kissat_remove_checker_binary(struct kissat *, unsigned, unsigned);
void kissat_remove_checker_clause(struct kissat *, struct clause *c);
void kissat_remove_checker_external(struct kissat *, size_t, const int *);

void kissat_remove_checker_internal(struct kissat *,
      size_t, const unsigned *);

#define ADD_UNCHECKED_EXTERNAL(SIZE,LITS) \
do { \
  if (GET_OPTION (check) > 1) \
    kissat_add_unchecked_external (solver, (SIZE), (LITS)); \
} while (0)

#define CHECK_AND_ADD_BINARY(A,B) \
do { \
  if (GET_OPTION (check) > 1) \
    kissat_check_and_add_binary (solver, (A), (B)); \
} while (0)

#define CHECK_AND_ADD_CLAUSE(CLAUSE) \
do { \
  if (GET_OPTION (check) > 1) \
    kissat_check_and_add_clause (solver, (CLAUSE)); \
} while (0)

#define CHECK_AND_ADD_EMPTY() \
do { \
  if (GET_OPTION (check) > 1) \
    kissat_check_and_add_empty (solver); \
} while (0)

#define CHECK_AND_ADD_LITS(SIZE,LITS) \
do { \
  if (GET_OPTION (check) > 1) \
    kissat_check_and_add_internal (solver, (SIZE), (LITS)); \
} while (0)

#define CHECK_AND_ADD_STACK(S) \
  CHECK_AND_ADD_LITS (SIZE_STACK (S), BEGIN_STACK (S))

#define CHECK_AND_ADD_UNIT(A) \
do { \
  if (GET_OPTION (check) > 1) \
    kissat_check_and_add_unit (solver, (A)); \
} while (0)

#define CHECK_SHRINK_CLAUSE(C,REMOVE,KEEP) \
do { \
  if (GET_OPTION (check) > 1) \
    kissat_check_shrink_clause (solver, (C), (REMOVE), (KEEP)); \
} while (0)

#define REMOVE_CHECKER_BINARY(A,B) \
do { \
  if (GET_OPTION (check) > 1) \
    kissat_remove_checker_binary (solver, (A), (B)); \
} while (0)

#define REMOVE_CHECKER_CLAUSE(CLAUSE) \
do { \
  if (GET_OPTION (check) > 1) \
    kissat_remove_checker_clause (solver, (CLAUSE)); \
} while (0)

#define REMOVE_CHECKER_LITS(SIZE,LITS) \
do { \
  if (GET_OPTION (check) > 1) \
    kissat_remove_checker_internal (solver, (SIZE), (LITS)); \
} while (0)

#define REMOVE_CHECKER_STACK(S) \
do { \
  if (GET_OPTION (check) > 1) \
    kissat_remove_checker_internal (solver, \
				    SIZE_STACK (S), BEGIN_STACK (S)); \
} while (0)

#else

#define ADD_UNCHECKED_EXTERNAL(...) do { } while (0)

#define CHECK_AND_ADD_BINARY(...) do { } while (0)
#define CHECK_AND_ADD_CLAUSE(...) do { } while (0)
#define CHECK_AND_ADD_EMPTY(...) do { } while (0)
#define CHECK_AND_ADD_LITS(...) do { } while (0)
#define CHECK_AND_ADD_STACK(...) do { } while (0)
#define CHECK_AND_ADD_UNIT(...) do { } while (0)

#define CHECK_SHRINK_CLAUSE(...) do { } while (0)

#define REMOVE_CHECKER_BINARY(...) do { } while (0)
#define REMOVE_CHECKER_CLAUSE(...) do { } while (0)
#define REMOVE_CHECKER_LITS(...) do { } while (0)
#define REMOVE_CHECKER_STACK(...) do { } while (0)

#endif
#endif

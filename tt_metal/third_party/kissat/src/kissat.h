#ifndef _kissat_h_INCLUDED
#define _kissat_h_INCLUDED

#define KISSAT_EXTRAS

typedef struct kissat kissat;

// Default (partial) IPASIR interface.

const char *kissat_signature(void);
kissat *kissat_init(void);
void kissat_add(kissat *solver, int lit);
void kissat_assume(kissat *solver, int lit);
int kissat_solve(kissat *solver);
int kissat_value(kissat *solver, int lit);
int kissat_failed(kissat *solver, int lit);
void kissat_release(kissat *solver);

void kissat_set_terminate(kissat *solver,
      void *state, int (*terminate)(void *state));

// Additional API functions.

void kissat_protect(kissat *solver, int var);
void kissat_unprotect(kissat *solver, int var);

void kissat_terminate(kissat *solver);
void kissat_reserve(kissat *solver, int max_var);

const char *kissat_id(void);
const char *kissat_version(void);
const char *kissat_compiler(void);

const char *kissat_copyright(void);
void kissat_build(const char *line_prefix);
void kissat_banner(const char *line_prefix, const char *name_of_app);

int kissat_get_option(kissat *solver, const char *name);
int kissat_set_option(kissat *solver, const char *name, int new_value);

int kissat_has_configuration(const char *name);
int kissat_set_configuration(kissat *solver, const char *name);

void kissat_set_conflict_limit(kissat *solver, unsigned);
void kissat_set_decision_limit(kissat *solver, unsigned);

void kissat_print_statistics(kissat *solver);

#endif

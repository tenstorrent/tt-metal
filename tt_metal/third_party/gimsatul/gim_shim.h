/* gim_shim.h
 *
 * Stable C API wrapping gimsatul v1.1.3 as an in-process static library.
 * Replaces the historical subprocess + DIMACS-file integration.
 *
 * Usage:
 *   gim_solver* s = gim_new(num_vars, threads);
 *   gim_add_clause(s, lits, n);   // lits are +/- 1-based DIMACS ints
 *   int st = gim_solve(s);        // 10 SAT / 20 UNSAT / 0 unknown
 *   if (st == 10) int v = gim_val(s, var);  // +var true, -var false
 *   gim_free(s);
 *
 * All gimsatul objects are compiled with -DNDEBUG (ABI requirement: struct
 * ruler has an `#ifndef NDEBUG` field). The shim silences stdout by setting
 * the gimsatul global `verbosity = -1`.
 */
#ifndef GIM_SHIM_H
#define GIM_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gim_solver gim_solver;

/* Create a solver for `num_vars` variables (1-based DIMACS numbering) using
 * `threads` worker rings (>=1). Returns NULL on allocation failure. */
gim_solver* gim_new(int num_vars, int threads);

/* Add one clause given as `n` signed 1-based DIMACS literals. Handles the
 * trivial (x and -x) / duplicate-literal / unit-conflict cases exactly as
 * gimsatul's DIMACS parser does. Safe to call repeatedly before gim_solve. */
void gim_add_clause(gim_solver* s, const int* dimacs_lits, int n);

/* Solve the accumulated formula. Returns 10 (SAT), 20 (UNSAT), or 0 (unknown).
 * On SAT, the witness is stashed for gim_val. Not incremental: call once. */
int gim_solve(gim_solver* s);

/* For the last SAT solve, return +var if `var` (1-based) is true, -var if
 * false. Returns 0 if var is out of range or no SAT model is available. */
int gim_val(gim_solver* s, int var);

/* Release all resources. */
void gim_free(gim_solver* s);

#ifdef __cplusplus
}
#endif

#endif /* GIM_SHIM_H */

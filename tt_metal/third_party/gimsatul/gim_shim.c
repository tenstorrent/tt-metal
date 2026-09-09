/* gim_shim.c
 *
 * In-process C API over gimsatul v1.1.3. See gim_shim.h.
 *
 * The clause-add dispatch mirrors parse_dimacs_body() in parse.c (trivial /
 * duplicate-literal handling + unit / binary / large clause primitives) and the
 * solve flow mirrors main() in gimsatul.c:
 *   new_ruler -> add clauses -> simplify_ruler -> clone_rings ->
 *   solve_rings -> winner->status -> extend_witness.
 *
 * MUST be compiled with -DNDEBUG -O3 (same flags as the gimsatul objects):
 * struct ruler has an `#ifndef NDEBUG original` field, so a mismatched NDEBUG
 * changes sizeof(struct ruler) and corrupts the ABI.
 */

#include "ruler.h"
#include "options.h"
#include "solve.h"
#include "witness.h"
#include "simplify.h"
#include "clone.h"
#include "detach.h"
#include "clause.h"
#include "stack.h"
#include "macros.h"
#include "trace.h"

#include "gim_shim.h"

#include <stdlib.h>

/* Globals defined in gimsatul objects but not declared in a public header. */
extern int verbosity;                       /* message.c: -1 silences 'c ...' */
extern void initialize_options(struct options*); /* options.c: memset 0 + defaults */

struct gim_solver {
    struct ruler* ruler;
    struct options options;
    signed char* marked;   /* scratch for duplicate/trivial detection, len nvars */
    int* model;            /* per-var (0-based) +1 true / -1 false after SAT */
    int nvars;
    int threads;
    int have_model;
    int solved;
};

gim_solver* gim_new(int num_vars, int threads) {
    if (num_vars < 0) {
        num_vars = 0;
    }
    if (threads < 1) {
        threads = 1;
    }
    verbosity = -1;  /* global: silence all stdout from the solver */

    gim_solver* s = (gim_solver*)calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }
    s->nvars = num_vars;
    s->threads = threads;

    initialize_options(&s->options);
    s->options.threads = threads;
    s->options.witness = 0;  /* we read the witness ourselves */

    s->ruler = new_ruler((size_t)num_vars, &s->options);
    s->marked = (signed char*)calloc((size_t)(num_vars > 0 ? num_vars : 1), sizeof(signed char));
    if (!s->ruler || !s->marked) {
        gim_free(s);
        return NULL;
    }
    return s;
}

void gim_add_clause(gim_solver* s, const int* dimacs_lits, int n) {
    if (!s || !s->ruler || s->solved) {
        return;
    }
    struct ruler* ruler = s->ruler;
    signed char* marked = s->marked;

    struct unsigneds clause;
    INIT(clause);
    int trivial = 0;

    for (int i = 0; i < n; i++) {
        int sl = dimacs_lits[i];
        if (sl == 0) {
            continue;  /* defensive: DIMACS 0 terminator should not appear here */
        }
        unsigned idx = (unsigned)abs(sl) - 1;
        signed char sign = (sl < 0) ? -1 : 1;
        unsigned ulit = 2u * idx + (sign < 0);  /* gimsatul literal encoding */
        signed char mark = marked[idx];
        if (mark == -sign) {
            trivial = 1;  /* x and -x in same clause */
        } else if (!mark) {
            PUSH(clause, ulit);
            marked[idx] = sign;
        }  /* else duplicate literal: drop */
    }

    if (!ruler->inconsistent && !trivial) {
        const size_t size = SIZE(clause);
        unsigned* literals = clause.begin;
        if (!size) {
            ruler->inconsistent = true;  /* empty clause */
        } else if (size == 1) {
            const unsigned unit = *clause.begin;
            const signed char value = ruler->values[unit];
            if (value < 0) {
                ruler->inconsistent = true;  /* conflicting unit */
                trace_add_empty(&ruler->trace);
            } else if (!value) {
                assign_ruler_unit(ruler, unit);
            }
        } else if (size == 2) {
            new_ruler_binary_clause(ruler, literals[0], literals[1]);
        } else {
            struct clause* large = new_large_clause(size, literals, false, 0);
            PUSH(ruler->clauses, large);
        }
    }

    for (all_elements_on_stack(unsigned, ul, clause)) {
        marked[IDX(ul)] = 0;
    }
    RELEASE(clause);
}

int gim_solve(gim_solver* s) {
    if (!s || !s->ruler || s->solved) {
        return 0;
    }
    s->solved = 1;
    s->have_model = 0;

    struct ruler* ruler = s->ruler;
    simplify_ruler(ruler);
    clone_rings(ruler);
    struct ring* winner = solve_rings(ruler);
    int status = winner ? winner->status : 0;

    if (status == 10) {
        signed char* witness = extend_witness(winner);
        s->model = (int*)calloc((size_t)(s->nvars > 0 ? s->nvars : 1), sizeof(int));
        if (s->model) {
            for (int i = 0; i < s->nvars; i++) {
                s->model[i] = (witness[LIT((unsigned)i)] > 0) ? 1 : -1;
            }
            s->have_model = 1;
        }
        free(witness);
    }
    return status;
}

int gim_val(gim_solver* s, int var) {
    if (!s || !s->have_model || !s->model || var < 1 || var > s->nvars) {
        return 0;
    }
    return (s->model[var - 1] > 0) ? var : -var;
}

void gim_free(gim_solver* s) {
    if (!s) {
        return;
    }
    if (s->ruler) {
        if (s->solved) {
            detach_and_delete_rings(s->ruler);
        }
        delete_ruler(s->ruler);
    }
    free(s->marked);
    free(s->model);
    free(s);
}

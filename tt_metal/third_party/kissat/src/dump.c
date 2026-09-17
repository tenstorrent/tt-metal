#ifndef NDEBUG

#include "inline.h"

#include <inttypes.h>

static void dump_literal(kissat *solver, unsigned ilit) {
  const int elit = kissat_export_literal(solver, ilit);
  printf("%u(%d)", ilit, elit);
  const int value = VALUE(ilit);
  if (value) {
    const unsigned ilit_level = LEVEL(ilit);
    printf("@%u=%d", ilit_level, value);
  }
}

static void dump_binary(kissat *solver, unsigned a, unsigned b) {
  printf("binary clause ");
  dump_literal(solver, a);
  fputc(' ', stdout);
  dump_literal(solver, b);
  fputc('\n', stdout);
}

static void dump_clause(kissat *solver, clause *c) {
  if (c->redundant) {
    printf("redundant glue %u", c->glue);
  } else {
    printf("irredundant");
  }
  const reference ref = kissat_reference_clause(solver, c);
  if (c->garbage) {
    printf(" garbage");
  }
  printf(" clause[%u]", ref);
  for (all_literals_in_clause(lit, c)) {
    fputc(' ', stdout);
    dump_literal(solver, lit);
  }
  fputc('\n', stdout);
}

static void dump_ref(kissat *solver, reference ref) {
  clause *c = kissat_dereference_clause(solver, ref);
  dump_clause(solver, c);
}


static void dump_trail(kissat *solver) {
  unsigned prev = 0;
  for (unsigned level = 0; level <= solver->level; level++) {
    frame *frame = &FRAME(level);
    unsigned next;
    if (level < solver->level) {
      next = frame[1].trail;
    } else {
      next = SIZE_ARRAY(solver->trail);
    }
    if (next == prev) {
      printf("frame[%u] has no assignments\n", level);
    } else {
      printf("frame[%u] has %u assignments\n", level, next - prev);
      if (prev < next) {
        printf("block[%u] = trail[%u..%u]\n", level, prev, next - 1);
      }
    }
    for (unsigned i = prev; i < next; i++) {
      printf("trail[%u] ", i);
      const unsigned lit = PEEK_ARRAY(solver->trail, i);
      dump_literal(solver, lit);
      const unsigned lit_level = LEVEL(lit);
      assert(lit_level <= level);
      if (lit_level < level) {
        printf(" out-of-order");
      }
      assigned *a = ASSIGNED(lit);
      if (!lit_level) {
        printf(" UNIT\n");
        assert(!a->binary);
        assert(a->reason == UNIT_REASON);
      } else {
        fputc(' ', stdout);
        if (a->binary) {
          const unsigned other = a->reason;
          dump_binary(solver, lit, other);
        } else if (a->reason == DECISION_REASON) {
          printf("DECISION\n");
        } else {
          assert(a->reason != UNIT_REASON);
          const reference ref = a->reason;
          dump_ref(solver, ref);
        }
      }
    }
    prev = next;
  }
}

static void dump_values(kissat *solver) {
  for (unsigned idx = 0; idx < VARS; idx++) {
    unsigned lit = LIT(idx);
    int value = solver->values[lit];
    printf("val[%u] = ", lit);
    if (!value) {
      printf("unassigned\n");
    } else {
      printf("%d\n", value);
    }
  }
}

static void dump_queue(kissat *solver) {
  const queue *const queue = &solver->queue;
  printf("queue: first %u, last %u, stamp %u, search %u (stamp %u)\n",
        queue->first, queue->last, queue->stamp,
        queue->search.idx, queue->search.stamp);
  const links *const links = solver->links;
  for (unsigned idx = queue->first;
        !DISCONNECTED(idx); idx = links[idx].next) {
    const struct links *l = links + idx;
    printf("%u ( prev %u, next %u, stamp %u )\n",
          idx, l->prev, l->next, l->stamp);
  }
}

static void dump_scores(kissat *solver) {
  heap *heap = &solver->scores;
  printf("scores.vars = %u\n", heap->vars);
  printf("scores.size = %u\n", heap->size);
  for (unsigned i = 0; i < SIZE_STACK(heap->stack); i++) {
    printf("scores.stack[%u] = %u\n", i, PEEK_STACK(heap->stack, i));
  }
  for (unsigned i = 0; i < heap->vars; i++) {
    printf("scores.score[%u] = %g\n", i, heap->score[i]);
  }
  for (unsigned i = 0; i < heap->vars; i++) {
    printf("scores.pos[%u] = %u\n", i, heap->pos[i]);
  }
}

static void dump_export(kissat *solver) {
  const unsigned size = SIZE_STACK(solver->export);
  for (unsigned idx = 0; idx < size; idx++) {
    printf("export[%u] = %u\n", LIT(idx), PEEK_STACK(solver->export, idx));
  }
}

void dump_map(kissat *solver) {
  const unsigned size = SIZE_STACK(solver->export);
  unsigned first = INVALID_LIT;
  for (unsigned idx = 0; idx < size; idx++) {
    const unsigned ilit = LIT(idx);
    const int elit = PEEK_STACK(solver->export, idx);
    printf("map[%u] -> %d", ilit, elit);
    if (elit) {
      const unsigned eidx = ABS(elit);
      const import *const import = &PEEK_STACK(solver->import, eidx);
      if (import->eliminated && import->imported) {
        printf(" -> eliminated[%u]", import->lit);
      } else if (import->eliminated) {
        printf(" -> substituted[%d]", KISSAT_GET_IMPORT_ELIT(import));
      } else {
        unsigned mlit = import->lit;
        if (elit < 0) {
          mlit = NOT(mlit);
        }
        printf(" -> %u", mlit);
      }
    }
    if (!LEVEL(ilit) && VALUE(ilit)) {
      if (first == INVALID_LIT) {
        first = ilit;
        printf(" #");
      } else {
        printf(" *");
      }
    }
    fputc('\n', stdout);
  }
}

static void dump_import(kissat *solver) {
  const unsigned size = SIZE_STACK(solver->import);
  for (unsigned idx = 1; idx < size; idx++) {
    import *import = &PEEK_STACK(solver->import, idx);
    printf("import[%u] = ", idx);
    if (!import->imported) {
      if (import->eliminated) {
        printf("substituted[%d]\n", KISSAT_GET_IMPORT_ELIT(import));
      } else {
        printf("undefined\n");
      }
    } else if (import->eliminated) {
      unsigned pos = import->lit;
      printf("eliminated[%u]", pos);
      if (pos < SIZE_STACK(solver->eliminated)) {
        int value = PEEK_STACK(solver->eliminated, pos);
        if (value) {
          printf(" (assigned to %d)", value);
        }
      }
      fputc('\n', stdout);
    } else {
      printf("%u\n", import->lit);
    }
  }
}

static void dump_etrail(kissat *solver) {
  for (unsigned i = 0; i < SIZE_STACK(solver->etrail); i++) {
    printf("etrail[%u] = %d\n", i, (int) PEEK_STACK(solver->etrail, i));
  }
}

static void dump_extend(kissat *solver) {
  const extension *const begin = BEGIN_STACK(solver->extend);
  const extension *const end = END_STACK(solver->extend);
  for (const extension *p = begin, *q; p != end; p = q) {
    assert(p->blocking);
    printf("extend[%zu] %d", (size_t)(p - begin), p->lit);
    if (!p[1].blocking) {
      fputs(" :", stdout);
    }
    for (q = p + 1; q != end && !q->blocking; q++) {
      printf(" %d", q->lit);
    }
    fputc('\n', stdout);
  }
}

static void dump_binaries(kissat *solver) {
  for (all_literals(lit)) {
    if (solver->watching) {
      for (all_binary_blocking_watches(watch, WATCHES(lit))) {
        if (!watch.type.binary) {
          continue;
        }
        const unsigned other = watch.binary.lit;
        if (lit > other) {
          continue;
        }
        if (watch.binary.redundant) {
          printf("redundant ");
        } else {
          printf("irredundant ");
        }
        dump_binary(solver, lit, other);
      }
    } else {
      for (all_binary_large_watches(watch, WATCHES(lit))) {
        if (!watch.type.binary) {
          continue;
        }
        const unsigned other = watch.binary.lit;
        if (lit > other) {
          continue;
        }
        COVER(watch.binary.redundant);
        if (watch.binary.redundant) {
          printf("redundant ");
        } else {
          printf("irredundant ");
        }
        dump_binary(solver, lit, other);
      }
    }
  }
}

static void dump_clauses(kissat *solver) {
  for (all_clauses(c)) {
    dump_clause(solver, c);
  }
}

void dump_vectors(kissat *solver) {
  vectors *vectors = &solver->vectors;
  unsigneds *stack = &vectors->stack;
  printf("vectors.size = %zu\n", SIZE_STACK(*stack));
  printf("vectors.capacity = %zu\n", CAPACITY_STACK(*stack));
  printf("vectors.usable = %zu\n", vectors->usable);
  const unsigned *const begin = BEGIN_STACK(*stack);
  const unsigned *const end = END_STACK(*stack);
  if (begin == end) {
    return;
  }
  fputc('-', stdout);
  for (const unsigned *p = begin + 1; p != end; p++)
    if (*p == INVALID_LIT) {
      fputs(" -", stdout);
    } else {
      printf(" %u", *p);
    }
  fputc('\n', stdout);
}

int dump(kissat *solver) {
  if (!solver) {
    return 0;
  }
  printf("vars = %u\n", solver->vars);
  printf("size = %u\n", solver->size);
  printf("level = %u\n", solver->level);
  printf("active = %u\n", solver->active);
  printf("assigned = %u\n", kissat_assigned(solver));
  printf("unassigned = %u\n", solver->unassigned);
  dump_import(solver);
  dump_export(solver);
#ifdef LOGGING
  if (solver->compacting) {
    dump_map(solver);
  }
#endif
  dump_etrail(solver);
  dump_extend(solver);
  dump_trail(solver);
  if (solver->stable) {
    dump_scores(solver);
  } else {
    dump_queue(solver);
  }
  dump_values(solver);
  printf("redundant = %" PRIu64 "\n", solver->statistics.clauses_redundant);
  printf("irredundant = %" PRIu64 "\n",
        solver->statistics.clauses_irredundant);
  dump_binaries(solver);
  dump_clauses(solver);
  dump_extend(solver);
  return 0;
}

#else
int kissat_dump_dummy_to_avoid_warning;
#endif

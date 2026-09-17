#if defined(LOGGING) && !defined(QUIET)

#include "colors.h"
#include "inline.h"

#include <stdarg.h>
#include <string.h>

static void begin_logging(kissat *solver, const char *prefix,
      const char *fmt, va_list *ap) {
  TERMINAL(stdout, 1);
  assert(GET_OPTION(log));
  fputs("c ", stdout);
  COLOR(MAGENTA);
  printf("%s %u ", prefix, solver->level);
  vprintf(fmt, *ap);
}

static void end_logging(void) {
  TERMINAL(stdout, 1);
  fputc('\n', stdout);
  COLOR(NORMAL);
  fflush(stdout);
}

void kissat_log_msg(kissat *solver, const char *prefix, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  end_logging();
}

static void append_sprintf(char *str, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  const size_t len = strlen(str);
  vsprintf(str + len, fmt, ap);
  va_end(ap);
}

const char *kissat_log_lit(kissat *solver, unsigned lit) {
  assert(solver);
  char *res = kissat_next_format_string(&solver->format);
  sprintf(res, "%u", lit);
  if (!solver->compacting && GET_OPTION(log) > 1) {
    append_sprintf(res, "(%d)", kissat_export_literal(solver, lit));
    if (solver->values) {
      const value value = VALUE(lit);
      if (value) {
        append_sprintf(res, "=%d", value);
        if (solver->assigned) {
          append_sprintf(res, "@%u", LEVEL(lit));
        }
      }
    }
  }
  assert(strlen(res) < FORMAT_STRING_SIZE);
  return res;
}

const char *kissat_log_var(kissat *solver, unsigned idx) {
  assert(solver);
  char *res = kissat_next_format_string(&solver->format);
  const unsigned lit = LIT(idx);
  sprintf(res, "variable %u (literal %s)", idx, LOGLIT(lit));
  assert(strlen(res) < FORMAT_STRING_SIZE);
  return res;
}

static void log_lits(kissat *solver, size_t size, const unsigned *lits) {
  for (size_t i = 0; i < size; i++) {
    fputc(' ', stdout);
    fputs(LOGLIT(lits[i]), stdout);
  }
}

void kissat_log_lits(kissat *solver, const char *prefix, size_t size,
      const unsigned *const lits, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  printf(" size %zu clause", size);
  log_lits(solver, size, lits);
  end_logging();
}

void kissat_log_resolvent(kissat *solver, const char *prefix,
      const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  const size_t size = SIZE_STACK(solver->resolvent);
  printf(" size %zu resolvent", size);
  const unsigned *const lits = BEGIN_STACK(solver->resolvent);
  log_lits(solver, size, lits);
  end_logging();
}

void kissat_log_ints(kissat *solver, const char *prefix, size_t size,
      const int *const lits, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  printf(" size %zu external literals clause", size);
  for (size_t i = 0; i < size; i++) {
    printf(" %d", lits[i]);
  }
  end_logging();
}

void kissat_log_unsigneds(kissat *solver, const char *prefix, size_t size,
      const unsigned *const lits, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  printf(" size %zu clause", size);
  for (size_t i = 0; i < size; i++) {
    printf(" %u", lits[i]);
  }
  end_logging();
}

void kissat_log_extensions(kissat *solver, const char *prefix, size_t size,
      const extension *const exts, const char *fmt, ...) {
  assert(size > 0);
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  const extension *const begin = BEGIN_STACK(solver->extend);
  const size_t pos = exts - begin;
  printf(" extend[%zu]", pos);
  printf(" %d", exts[0].lit);
  if (size > 1) {
    fputs(" :", stdout);
  }
  for (size_t i = 1; i < size; i++) {
    printf(" %d", exts[i].lit);
  }
  end_logging();
}

static void log_clause(kissat *solver, const clause *c) {
  fputc(' ', stdout);
  if (c == &solver->conflict) {
    fputs("static ", stdout);
    fputs(c->redundant ? "redundant" : "irredundant", stdout);
    fputs(" binary conflict clause", stdout);
  } else {
    if (c->hyper) {
      assert(c->size == 3);
      fputs("hyper ", stdout);
    }
    if (c->redundant) {
      printf("redundant glue %u", c->glue);
    } else {
      fputs("irredundant", stdout);
    }
    printf(" size %u", c->size);
    if (c->reason) {
      fputs(" reason", stdout);
    }
    if (c->garbage) {
      fputs(" garbage", stdout);
    }
    fputs(" clause", stdout);
    if (kissat_clause_in_arena(solver, c)) {
      reference ref = kissat_reference_clause(solver, c);
      printf("[%u]", ref);
    }
  }
}

void kissat_log_clause(kissat *solver, const char *prefix,
      const clause *c, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  log_clause(solver, c);
  log_lits(solver, c->size, c->lits);
  end_logging();
}

static void log_binary(kissat *solver, unsigned a, unsigned b) {
  printf(" binary clause %s %s", LOGLIT(a), LOGLIT(b));
}

void kissat_log_binary(kissat *solver, const char *prefix,
      unsigned a, unsigned b, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  log_binary(solver, a, b);
  end_logging();
}

void kissat_log_unary(kissat *solver, const char *prefix, unsigned a,
      const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  printf(" unary clause %s", LOGLIT(a));
  end_logging();
}

static void log_ref(kissat *solver, reference ref) {
  clause *c = kissat_dereference_clause(solver, ref);
  log_clause(solver, c);
  log_lits(solver, c->size, c->lits);
}

void kissat_log_ref(kissat *solver, const char *prefix, reference ref,
      const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  log_ref(solver, ref);
  end_logging();
}

void kissat_log_watch(kissat *solver, const char *prefix,
      unsigned lit, watch watch, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  if (watch.type.binary) {
    log_binary(solver, lit, watch.binary.lit);
  } else {
    log_ref(solver, watch.large.ref);
  }
  end_logging();
}

void kissat_log_line(kissat *solver, const char *prefix,
      const line *line, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  printf(" cache line[%" PRIu64 "]"
        " unsatisfying %u clause over %u variables and signature 0x%016"
        PRIx64 "", line->inserted, line->unsatisfied, line->vars,
        line->signature);
  va_end(ap);
  end_logging();
}

void kissat_log_xor(kissat *solver, const char *prefix, unsigned lit,
      unsigned size, const unsigned *lits, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  begin_logging(solver, prefix, fmt, &ap);
  va_end(ap);
  printf(" size %u XOR gate ", size);
  fputs(kissat_log_lit(solver, lit), stdout);
  printf(" =");
  for (unsigned i = 0; i < size; i++) {
    if (i) {
      fputs(" ^ ", stdout);
    } else {
      fputc(' ', stdout);
    }
    fputs(kissat_log_lit(solver, lits[i]), stdout);
  }
  end_logging();
}

#else

int kissat_log_dummy_to_avoid_pedantic_warning;

#endif

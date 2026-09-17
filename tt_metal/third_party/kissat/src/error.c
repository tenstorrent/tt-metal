#include "colors.h"
#include "cover.h"
#include "error.h"

#include <stdarg.h>
#include <stdlib.h>

static void (*kissat_abort_function)(void);

void kissat_call_function_instead_of_abort(void (*f)(void)) {
  kissat_abort_function = f;
}

// *INDENT-OFF*

void
kissat_abort (void)
{
  if (kissat_abort_function)
    { FLUSH_COVERAGE (); kissat_abort_function (); } // Keep all in this line.
  else
    { FLUSH_COVERAGE (); abort (); }                 // Keep all in this line.
}

// *INDENT-ON*

static void typed_error_message_start(const char *type) {
  fflush(stdout);
  TERMINAL(stderr, 2);
  COLOR(BOLD);
  fputs("kissat: ", stderr);
  COLOR(RED);
  fputs(type, stderr);
  fputs(": ", stderr);
  COLOR(NORMAL);
}

void kissat_fatal_message_start(void) {
  typed_error_message_start("fatal error");
}

static void vprint_error(const char *type, const char *fmt, va_list *ap) {
  typed_error_message_start(type);
  vfprintf(stderr, fmt, *ap);
  fputc('\n', stderr);
  fflush(stderr);
}

void kissat_error(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vprint_error("error", fmt, &ap);
  va_end(ap);
}

void kissat_fatal(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vprint_error("fatal error", fmt, &ap);
  va_end(ap);
  kissat_abort();
}

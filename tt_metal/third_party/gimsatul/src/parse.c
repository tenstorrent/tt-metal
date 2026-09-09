#include "parse.h"
#include "geatures.h"
#include "message.h"
#include "options.h"
#include "ruler.h"

#include <assert.h>
#include <ctype.h>
#include <inttypes.h>
#include <stdarg.h>
#include <string.h>

static void parse_error (struct file *dimacs, const char *, ...)
    __attribute__ ((format (printf, 2, 3)));

static void parse_error (struct file *dimacs, const char *fmt, ...) {
  fprintf (stderr, "gimsatul: parse error: at line %" PRIu64 " in '%s': ",
           dimacs->lines, dimacs->path);
  va_list ap;
  va_start (ap, fmt);
  vfprintf (stderr, fmt, ap);
  va_end (ap);
  fputc ('\n', stderr);
  exit (1);
}

/*------------------------------------------------------------------------*/

static int next_char (struct file *dimacs) {
  int res = getc (dimacs->file);
  if (res == '\r') {
    res = getc (dimacs->file);
    if (res != '\n')
      return EOF;
  }
  if (res == '\n')
    dimacs->lines++;
  return res;
}

static bool parse_int (struct file *dimacs, int *res_ptr, int prev,
                       int *next) {
  int ch = prev == EOF ? next_char (dimacs) : prev;
  int sign = 1;
  if (ch == '-') {
    sign = -1;
    ch = next_char (dimacs);
    if (!isdigit (ch) || ch == '0')
      return false;
  } else if (!isdigit (ch))
    return false;
  unsigned tmp = ch - '0';
  while (isdigit (ch = next_char (dimacs))) {
    if (!tmp && ch == '0')
      return false;
    if (UINT_MAX / 10 < tmp)
      return false;
    tmp *= 10;
    unsigned digit = ch - '0';
    if (UINT_MAX - digit < tmp)
      return false;
    tmp += digit;
  }
  int res;
  if (sign > 0) {
    if (tmp > 0x7fffffffu)
      return false;
    res = tmp;
  } else {
    assert (sign < 0);
    if (tmp > 0x80000000u)
      return false;
    if (tmp == 0x80000000u)
      res = INT_MIN;
    else
      res = -tmp;
  }
  *next = ch;
  *res_ptr = res;
  return true;
}

void parse_dimacs_header (struct options *options, int *variables_ptr,
                          int *clauses_ptr) {
  struct file *dimacs = &options->dimacs;
  if (verbosity >= 0) {
    printf ("c\nc parsing DIMACS file '%s'\n", dimacs->path);
    fflush (stdout);
  }
  int ch;
#ifndef NDEBUG
  struct buffer buffer;
  INIT (buffer);
  while ((ch = next_char (dimacs)) == 'c') {
    while ((ch = next_char (dimacs)) == ' ' || ch == '\t')
      ;
    assert (EMPTY (buffer));
    if (ch == '\n')
      continue;
    int first = ch;
    do
      if (ch == EOF)
        parse_error (dimacs, "unexpected end-of-file in header comment");
      else if (first == '-')
        PUSH (buffer, ch);
    while ((ch = next_char (dimacs)) != '\n');
    if (first == '-') {
      PUSH (buffer, 0);
      (void) parse_option_with_value (options, buffer.begin);
      CLEAR (buffer);
    }
  }
  RELEASE (buffer);
#else
  while ((ch = next_char (dimacs)) == 'c') {
    while ((ch = next_char (dimacs)) != '\n')
      if (ch == EOF)
        parse_error (dimacs, "unexpected end-of-file in header comment");
  }
#endif
  if (ch != 'p')
    parse_error (dimacs, "expected 'c' or 'p'");
  int variables = 0, clauses;
  if (next_char (dimacs) != ' ' || next_char (dimacs) != 'c' ||
      next_char (dimacs) != 'n' || next_char (dimacs) != 'f' ||
      next_char (dimacs) != ' ' ||
      !parse_int (dimacs, &variables, EOF, &ch) || variables < 0 ||
      ch != ' ' || !parse_int (dimacs, &clauses, EOF, &ch) || clauses < 0)
  INVALID_HEADER:
    parse_error (dimacs, "invalid 'p cnf ...' header line");
  if (variables > MAX_VAR)
    parse_error (dimacs, "too many variables (maximum %u)", MAX_VAR);
  while (ch == ' ' || ch == '\t')
    ch = next_char (dimacs);
  if (ch != '\n')
    goto INVALID_HEADER;
  message (0, "parsed 'p cnf %d %d' header", variables, clauses);
  *variables_ptr = variables;
  *clauses_ptr = clauses;
}

void parse_dimacs_body (struct ruler *ruler, int variables, int expected) {
#ifndef QUIET
  double start_parsing = START (ruler, parse);
#endif
#if 0
  bool force = ruler->options.force;
#endif
  struct file *dimacs = &ruler->options.dimacs;
  signed char *marked = allocate_and_clear_block (variables);
  struct unsigneds clause;
  INIT (clause);
  int signed_lit = 0, parsed = 0;
#ifndef NDEBUG
  struct unsigneds *original = ruler->original;
#endif
  bool trivial = false;
  for (;;) {
    int ch = next_char (dimacs);
    if (ch == EOF) {
    END_OF_FILE:
      if (signed_lit)
        parse_error (dimacs, "terminating zero missing");
      if (parsed != expected)
        parse_error (dimacs, "clause missing");
      break;
    }
    if (ch == ' ' || ch == '\t' || ch == '\n')
      continue;
    if (ch == 'c') {
    SKIP_BODY_COMMENT:
      while ((ch = next_char (dimacs)) != '\n')
        if (ch == EOF)
          parse_error (dimacs, "invalid end-of-file in body comment");
      continue;
    }
    if (!parse_int (dimacs, &signed_lit, ch, &ch))
      parse_error (dimacs, "failed to parse literal");
    if (signed_lit == INT_MIN || abs (signed_lit) > variables)
      parse_error (dimacs, "invalid literal %d", signed_lit);
    if (parsed == expected)
      parse_error (dimacs, "too many clauses");
    if (ch != 'c' && ch != ' ' && ch != '\t' && ch != '\n' && ch != EOF)
      parse_error (dimacs, "invalid character after '%d'", signed_lit);
    if (signed_lit) {
      unsigned idx = abs (signed_lit) - 1;
      assert (idx < (unsigned) variables);
      signed char sign = (signed_lit < 0) ? -1 : 1;
      signed char mark = marked[idx];
      unsigned unsigned_lit = 2 * idx + (sign < 0);
#ifndef NDEBUG
      PUSH (*original, unsigned_lit);
#endif
      if (mark == -sign) {
        ROG ("skipping trivial clause");
        trivial = true;
      } else if (!mark) {
        PUSH (clause, unsigned_lit);
        marked[idx] = sign;
      } else
        assert (mark == sign);
    } else {
#ifndef NDEBUG
      PUSH (*original, INVALID);
#endif
      parsed++;
      unsigned *literals = clause.begin;
      if (!ruler->inconsistent && !trivial) {
        const size_t size = SIZE (clause);
        assert (size <= ruler->size);
        if (!size) {
          assert (!ruler->inconsistent);
          very_verbose (0, "%s", "found empty original clause");
          ruler->inconsistent = true;
        } else if (size == 1) {
          const unsigned unit = *clause.begin;
          const signed char value = ruler->values[unit];
          if (value < 0) {
            assert (!ruler->inconsistent);
            very_verbose (0, "found inconsistent unit");
            ruler->inconsistent = true;
            trace_add_empty (&ruler->trace);
          } else if (!value)
            assign_ruler_unit (ruler, unit);
        } else if (size == 2)
          new_ruler_binary_clause (ruler, literals[0], literals[1]);
        else {
          struct clause *large_clause =
              new_large_clause (size, literals, false, 0);
          ROGCLAUSE (large_clause, "new");
          PUSH (ruler->clauses, large_clause);
        }
      } else
        trivial = false;
      for (all_elements_on_stack (unsigned, unsigned_lit, clause))
        marked[IDX (unsigned_lit)] = 0;
      CLEAR (clause);
    }
    if (ch == 'c')
      goto SKIP_BODY_COMMENT;
    if (ch == EOF)
      goto END_OF_FILE;
  }
  assert (parsed == expected);
  assert (dimacs->file);
  if (dimacs->close == 1)
    fclose (dimacs->file);
#ifdef GIMSATUL_HAS_COMPRESSION
  if (dimacs->close == 2)
    pclose (dimacs->file);
#endif
  RELEASE (clause);
  ruler->statistics.original = parsed;
  free (marked);
#ifndef QUIET
  double end_parsing = STOP (ruler, parse);
  message (0, "parsing took %.2f seconds", end_parsing - start_parsing);
#endif
}

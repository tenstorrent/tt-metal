#ifndef QUIET

#include "colors.h"
#include "internal.h"
#include "print.h"
#include "report.h"
#include "resources.h"

#include <inttypes.h>
#include <string.h>

#define MB \
  (kissat_current_resident_set_size ()/(double)(1<<20))

#define REMAINING_VARIABLES \
  kissat_percent (solver->active, SIZE_STACK(solver->import))

#define REPORTS \
REP("seconds", "5.2f", kissat_time (solver)) \
REP("MB", "2.0f", MB) \
REP("level", ".0f", AVERAGE (level)) \
REP("switched", "1" PRIu64, statistics->switched_modes) \
REP("reductions", "1" PRIu64, statistics->reductions) \
REP("restarts", "2" PRIu64, statistics->restarts) \
REP("rate", ".0f", AVERAGE (decision_rate)) \
REP("conflicts", "3" PRIu64, CONFLICTS) \
REP("redundant", "3" PRIu64, REDUNDANT_CLAUSES) \
REP("trail", ".0f%%", AVERAGE (trail)) \
REP("glue", ".0f", AVERAGE (slow_glue)) \
REP("irredundant", "2" PRIu64, IRREDUNDANT_CLAUSES) \
REP("variables", "2u", solver->active) \
REP("remaining", "1.0f%%" , REMAINING_VARIABLES) \

void kissat_report(kissat *solver, bool verbose, char type) {
  statistics *statistics = &solver->statistics;
  const int verbosity = kissat_verbosity(solver);
  if (verbosity < 0) {
    return;
  }
  if (verbose && verbosity < 2) {
    return;
  }
  char line[128], *p = line;
  unsigned pad[32], n = 1, pos = 0;
  pad[0] = 0;
  // *INDENT-OFF*
#define REP(NAME,FMT,VALUE) \
  do { \
    *p++ = ' ', pos++; \
    sprintf (p, "%" FMT, VALUE); \
    while (*p) \
      p++, pos++; \
    pad[n++] = pos; \
  } while (0);
  REPORTS
#undef REP
  // *INDENT-ON*
  assert(p < line + sizeof line);
  TERMINAL(stdout, 1);
  if (!(solver->limits.reports++ % 20)) {
#define ROWS 3
    unsigned last[ROWS];
    char rows[ROWS][128], *r[ROWS];
    for (unsigned j = 0; j < ROWS; j++) {
      last[j] = 0, rows[j][0] = 0, r[j] = rows[j];
    }
    unsigned row = 0, i = 1;
#define REP(NAME,FMT,VALUE) \
      do { \
	if (last[row]) \
	  *r[row]++ = ' ', last[row]++; \
	unsigned target = pad[i]; \
	const unsigned name_len = strlen (NAME); \
	const unsigned val_len = target - pad[i-1] - 1; \
	if (val_len < name_len) \
	  target += (name_len - val_len)/2; \
	while (last[row] + name_len < target) \
	  *r[row]++ = ' ', last[row]++; \
	for (const char * p = NAME; *p; p++) \
	  *r[row]++ = *p, last[row]++; \
	if (++row == ROWS) \
	  row = 0; \
	i++; \
      } while (0);
    REPORTS
#undef REP
    assert(i == n);
    for (unsigned j = 0; j < ROWS; j++) {
      assert(r[j] < rows[j] + sizeof rows[j]);
      *r[j] = 0;
    }
    if (solver->limits.reports > 1) {
      fputs("c\n", stdout);
    }
    for (unsigned j = 0; j < ROWS; j++) {
      fputs("c  ", stdout);
      COLOR(YELLOW);
      fputs(rows[j], stdout);
      COLOR(NORMAL);
      fputc('\n', stdout);
    }
    fputs("c\n", stdout);
  }
  fputc('c', stdout);
  fputc(' ', stdout);
  switch (type) {
  case '1':
  case '0':
  case '?':
  case 'i':
    COLOR(BOLD);
    break;
  case 'e':
    COLOR(BOLD GREEN);
    break;
  case '2':
  case 's':
    COLOR(GREEN);
    break;
  case '3':
  case 'f':
  case 't':
  case 'u':
  case 'v':
  case 'w':
    COLOR(BLUE);
    break;
  case 'b':
  case 'd':
    COLOR(BOLD BLUE);
    break;
  case '[':
  case ']':
    COLOR(MAGENTA);
    break;
  }
  fputc(type, stdout);
  COLOR(NORMAL);
  if (solver->stable) {
    COLOR(MAGENTA);
  }
  fputs(line, stdout);
  if (solver->stable) {
    COLOR(NORMAL);
  }
  fputc('\n', stdout);
  fflush(stdout);
}

#else

int kissat_report_dummy_to_avoid_warning;

#endif

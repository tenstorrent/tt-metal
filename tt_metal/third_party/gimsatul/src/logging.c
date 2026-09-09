#include "logging.h"
#include "ring.h"
#include "ruler.h"
#include "utilities.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

bool ignore_values_and_levels_during_logging;

static char loglitbuf[8][64];
static unsigned loglitpos;

#define loglitsize (sizeof loglitbuf / sizeof *loglitbuf)

static char *next_loglitbuf (void) {
  char *res = loglitbuf[loglitpos++];
  if (loglitpos == loglitsize)
    loglitpos = 0;
  return res;
}

const char *loglit (struct ring *ring, unsigned unsigned_lit) {
  char *res = next_loglitbuf ();
  int signed_lit =
      unmap_and_export_literal (ring->ruler->unmap, unsigned_lit);
  sprintf (res, "%u(%d)", unsigned_lit, signed_lit);
  if (!ignore_values_and_levels_during_logging) {
    signed char *values = ring->values;
    if (values) {
      signed char value = values[unsigned_lit];
      if (value) {
        sprintf (res + strlen (res), "=%d", (int) value);
        if (ring->variables) {
          struct variable *v = VAR (unsigned_lit);
          if (v->level != INVALID)
            sprintf (res + strlen (res), "@%u", v->level);
        }
      }
    }
  }
  assert (strlen (res) + 1 < sizeof *loglitbuf);
  return res;
}

const char *logvar (struct ring *ring, unsigned idx) {
  unsigned lit = LIT (idx);
  const char *tmp = loglit (ring, lit);
  char *res = next_loglitbuf ();
  sprintf (res, "variable %u(%u) (literal %s)", idx, idx + 1, tmp);
  return res;
}

const char *roglit (struct ruler *ruler, unsigned unsigned_lit) {
  char *res = next_loglitbuf ();
  int signed_lit = unmap_and_export_literal (ruler->unmap, unsigned_lit);
  sprintf (res, "%u(%d)", unsigned_lit, signed_lit);
  if (!ignore_values_and_levels_during_logging) {
    signed char *values = (signed char *) ruler->values;
    if (values) {
      signed char value = values[unsigned_lit];
      if (value)
        sprintf (res + strlen (res), "=%d", (int) value);
    }
  }
  assert (strlen (res) + 1 < sizeof *loglitbuf);
  return res;
}

const char *rogvar (struct ruler *ruler, unsigned idx) {
  unsigned lit = LIT (idx);
  const char *tmp = roglit (ruler, lit);
  char *res = next_loglitbuf ();
  sprintf (res, "variable %u(%u) (literal %s)", idx, idx + 1, tmp);
  return res;
}

#include "application.h"
#include "cover.h"
#include "handle.h"
#include "kissat.h"
#include "print.h"

#include <assert.h>
#include <stdbool.h>

static kissat *volatile solver;

// *INDENT-OFF*

static void
kissat_signal_handler (int sig)
{
  kissat_signal (solver, "caught", sig);
  kissat_print_statistics (solver);
  kissat_signal (solver, "raising", sig);
#ifdef QUIET
  (void) sig;
#endif
  FLUSH_COVERAGE (); } // Keep this '}' in the same line!

// *INDENT-ON*

static volatile bool ignore_alarm = false;

static void kissat_alarm_handler(void) {
  if (ignore_alarm) {
    return;
  }
  assert(solver);
  kissat_terminate(solver);
}

#ifndef NDEBUG
extern int dump(kissat *);
#endif

#include "random.h"
#include "error.h"
#include <strings.h>

int main(int argc, char **argv) {
  int res;
  solver = kissat_init();
  kissat_set_option(solver, "incremental", 0);
  kissat_init_alarm(kissat_alarm_handler);
  kissat_init_signal_handler(kissat_signal_handler);
  res = kissat_application(solver, argc, argv);
  kissat_reset_signal_handler();
  ignore_alarm = true;
  kissat_reset_alarm();
  kissat_release(solver);
#ifndef NDEBUG
  if (!res) {
    return dump(0);
  }
#endif
  return res;
}

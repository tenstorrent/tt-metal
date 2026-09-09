#include "catch.h"
#include "message.h"
#include "ruler.h"
#include "statistics.h"

#include <assert.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*------------------------------------------------------------------------*/

static atomic_int caught_signal;
static volatile struct ruler *one_global_ruler;
static atomic_bool catching_signals;
static atomic_bool catching_alarm;

/*------------------------------------------------------------------------*/

#define SIGNALS \
  SIGNAL (SIGABRT) \
  SIGNAL (SIGBUS) \
  SIGNAL (SIGILL) \
  SIGNAL (SIGINT) \
  SIGNAL (SIGSEGV) \
  SIGNAL (SIGTERM)

// clang-format off

// Saved previous signal handlers.

#define SIGNAL(SIG) \
static void (*saved_ ## SIG ## _handler)(int);
SIGNALS
#undef SIGNAL
static void (*saved_SIGALRM_handler)(int);

// clang-format on

static void reset_alarm_handler (void) {
  if (atomic_exchange (&catching_alarm, false))
    signal (SIGALRM, saved_SIGALRM_handler);
}

void reset_signal_handlers (void) {
  one_global_ruler = 0;
  if (atomic_exchange (&catching_signals, false)) {
    // clang-format off
#define SIGNAL(SIG) \
      signal (SIG, saved_ ## SIG ## _handler);
      SIGNALS
#undef SIGNAL
    // clang-format on
  }
  reset_alarm_handler ();
}

static void caught_message (int sig) {
  if (verbosity < 0)
    return;
  const char *name = "SIGNUNKNOWN";
#define SIGNAL(SIG) \
  if (sig == SIG) \
    name = #SIG;
  SIGNALS
#undef SIGNAL
  if (sig == SIGALRM)
    name = "SIGALRM";
  char buffer[80];
  sprintf (buffer, "c\nc caught signal %d (%s)\nc\n", sig, name);
  size_t bytes = strlen (buffer);
  if (write (1, buffer, bytes) != bytes)
    exit (2);
}

static void raising_message (int sig) {
  if (verbosity < 0)
    return;
  const char *name = "SIGNUNKNOWN";
#define SIGNAL(SIG) \
  if (sig == SIG) \
    name = #SIG;
  SIGNALS
#undef SIGNAL
  if (sig == SIGALRM)
    name = "SIGALRM";
  char buffer[80];
  sprintf (buffer,
           "c\nc raising signal %d (%s) after reporting statistics\n", sig,
           name);
  size_t bytes = strlen (buffer);
  if (write (1, buffer, bytes) != bytes)
    exit (2);
}

static void exit_message (void) {
  const char *message = "c calling 'exit (1)' as raising signal returned\n";
  size_t bytes = strlen (message);
  if (write (1, message, bytes) != bytes)
    exit (2);
}

static void catch_signal (int sig) {
  if (atomic_exchange (&caught_signal, sig))
    return;
  caught_message (sig);
#ifndef QUIET
  struct ruler *ruler = (struct ruler *) one_global_ruler;
#endif
  reset_signal_handlers ();
#ifndef QUIET
  if (ruler)
    print_ruler_statistics (ruler);
#endif
  raising_message (sig);
  raise (sig);
  exit_message ();
  exit (1);
}

static void catch_alarm (int sig) {
  assert (sig == SIGALRM);
  if (!catching_alarm)
    catch_signal (sig);
  if (atomic_exchange (&caught_signal, sig))
    return;
  if (verbosity > 0)
    caught_message (sig);
  reset_alarm_handler ();
  assert (one_global_ruler);
  set_terminate ((struct ruler *) one_global_ruler, 0);
  caught_signal = 0;
}

static void set_alarm_handler (unsigned seconds) {
  assert (seconds);
  assert (!catching_alarm);
  saved_SIGALRM_handler = signal (SIGALRM, catch_alarm);
  alarm (seconds);
  catching_alarm = true;
}

void set_signal_handlers (struct ruler *ruler) {
  unsigned seconds = ruler->options.seconds;
  one_global_ruler = ruler;
  assert (!catching_signals);
  // clang-format off
#define SIGNAL(SIG) \
  saved_ ## SIG ##_handler = signal (SIG, catch_signal);
  SIGNALS
#undef SIGNAL
  // clang-format on
  catching_signals = true;
  if (seconds)
    set_alarm_handler (seconds);
}

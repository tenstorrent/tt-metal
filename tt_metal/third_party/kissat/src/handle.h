#ifndef _handle_h_INCLUDED
#define _handle_h_INCLUDED

#include <signal.h>

void kissat_init_signal_handler(void (*handler)(int));
void kissat_reset_signal_handler(void);

void kissat_init_alarm(void (*handler)(void));
void kissat_reset_alarm(void);

#define SIGNALS \
SIGNAL(SIGABRT) \
SIGNAL(SIGBUS) \
SIGNAL(SIGINT) \
SIGNAL(SIGSEGV) \
SIGNAL(SIGTERM)

// *INDENT-OFF*

static inline const char *
kissat_signal_name (int sig)
{
#define SIGNAL(SIG) \
  if (sig == SIG) return #SIG;
  SIGNALS
#undef SIGNAL
  if (sig == SIGALRM)
    return "SIGALRM";
  return "SIGUNKNOWN";
}

// *INDENT-ON*

#endif

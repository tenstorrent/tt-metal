#ifndef _report_h_INCLUDED
#define _report_h_INCLUDED

#ifndef QUIET

struct ring;

void reset_report ();
void report (struct ring *, char);
void verbose_report (struct ring *, char, int level);

#else

#define report(...) \
  do { \
  } while (0)
#define verbose_report(...) \
  do { \
  } while (0)

#endif

#endif

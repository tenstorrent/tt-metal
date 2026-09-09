#ifndef _witness_h_INCLUDED
#define _witness_h_INCLUDED

struct ring;

signed char *extend_witness (struct ring *);
void print_witness (unsigned size, signed char *values);

#ifndef NDEBUG

struct unsigneds;
void check_witness (signed char *values, struct unsigneds *);

#else

#define check_witness(...) \
  do { \
  } while (0)

#endif

#endif

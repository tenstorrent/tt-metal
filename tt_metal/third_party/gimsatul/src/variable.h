#ifndef _variable_h_INCLUDED
#define _variable_h_INCLUDED

#include <stdbool.h>

struct watch;

struct phases {
  signed char best : 2;
  signed char saved : 2;
  signed char target : 2;
  signed char padding : 2;
};

struct variable {
  unsigned level;
  bool minimize;
  bool poison;
  bool seen;
  bool shrinkable;
  struct watch *reason;
};

#endif

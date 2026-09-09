#ifndef _scale_h_INCLUDED
#define _scale_h_INCLUDED

#include <stdint.h>

struct ring;

uint64_t scale_interval (struct ring *, const char *, uint64_t);

#endif

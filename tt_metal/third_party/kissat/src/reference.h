#ifndef _reference_h_INCLUDED
#define _reference_h_INCLUDED

#include "stack.h"

typedef unsigned reference;

#define REFERENCE_FORMAT "u"

#define LD_MAX_REF 31u
#define MAX_REF ((1u << LD_MAX_REF)-1)

#define INVALID_REF UINT_MAX

// *INDENT-OFF*
typedef STACK (reference) references;
// *INDENT-ON*

#endif

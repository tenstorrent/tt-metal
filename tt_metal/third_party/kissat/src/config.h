#ifndef NOPTIONS
#ifndef _config_h_INCLUDED
#define _config_h_INCLUDED

#define CONFIGURATIONS \
CONFIGURATION (default) \
CONFIGURATION (sat) \
CONFIGURATION (unsat) \

struct kissat;

void kissat_configuration_usage(void);

#endif
#endif

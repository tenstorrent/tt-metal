#ifndef _resources_h_INCLUDED
#define _resources_h_INCLUDED

double kissat_wall_clock_time(void);

#ifndef QUIET

#ifndef _resources_h_INLCUDED
#define _resources_h_INLCUDED

#include <stdint.h>

struct kissat;

double kissat_process_time(void);
uint64_t kissat_current_resident_set_size(void);
uint64_t kissat_maximum_resident_set_size(void);
void kissat_print_resources(struct kissat *);

#endif

#endif
#endif

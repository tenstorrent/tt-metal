#ifndef _resize_h_INCLUDED
#define _resize_h_INCLUDED

struct kissat;

void kissat_decrease_size(struct kissat *solver);
void kissat_increase_size(struct kissat *, unsigned new_size);
void kissat_enlarge_variables(struct kissat *, unsigned new_vars);

#endif

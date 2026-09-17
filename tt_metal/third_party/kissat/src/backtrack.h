#ifndef _backtrack_h_INCLUDED
#define _backtrack_h_INCLUDED

struct kissat;

void kissat_backtrack_without_updating_phases(struct kissat *, unsigned);
void kissat_backtrack_in_consistent_state(struct kissat *, unsigned);
void kissat_backtrack_after_conflict(struct kissat *, unsigned);
void kissat_backtrack_propagate_and_flush_trail(struct kissat *);

#endif

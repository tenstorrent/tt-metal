#ifndef _backtrack_h_INCLUDED
#define _backtrack_h_INCLUDED

struct ring;
void backtrack (struct ring *, unsigned level);
void update_best_and_target_phases (struct ring *);

#endif

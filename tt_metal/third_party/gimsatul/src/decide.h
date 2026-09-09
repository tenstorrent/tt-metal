#ifndef _decide_h_INCLUDED
#define _decide_h_INCLUDED

struct ring;

void decide (struct ring *);

signed char initial_phase (struct ring *);
signed char decide_phase (struct ring *, unsigned idx);
void start_random_decision_sequence (struct ring *);

#endif

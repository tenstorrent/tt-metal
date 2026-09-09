#ifndef _average_h_INCLUDED
#define _average_h_INCLUDED

struct average {
  double value, biased, exp;
};

struct ring;

void update_average (struct ring *, struct average *, const char *,
                     double alpha, double);

#endif

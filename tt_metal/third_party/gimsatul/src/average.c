#include "average.h"
#include "logging.h"
#include "message.h"

#ifdef LOGGING
#include "ring.h"
#endif

void update_average (struct ring *ring, struct average *average,
                     const char *name, double alpha, double y) {
  double old_biased = average->biased;
  double delta = y - old_biased;
  double scaled_delta = alpha * delta;
  double new_biased = old_biased + scaled_delta;
  average->biased = new_biased;
  double old_exp = average->exp;
  double new_value;
  if (old_exp) {
    double beta = 1 - alpha;
    double new_exp = old_exp * beta;
    average->exp = new_exp;
    double div = 1 - new_exp;
    new_value = new_biased / div;
  } else
    new_value = new_biased;
  LOG ("update %s average with %g to %g from %g", name, y, new_value,
       average->value);
  average->value = new_value;
  (void) ring, (void) name;
}

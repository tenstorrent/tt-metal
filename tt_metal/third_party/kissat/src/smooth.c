#include "allocate.h"
#include "internal.h"
#include "logging.h"

void kissat_init_smooth(kissat *solver, smooth *smooth, int window,
      const char *name) {
  assert(window > 0);
  const double alpha = 1.0 / window;
  LOG("initialized %s EMA alpha %g window %d", name, alpha, window);
  smooth->value = 0;
  smooth->biased = 0;
  smooth->alpha = alpha;
  smooth->beta = 1.0 - alpha;
  assert(smooth->beta > 0);
  smooth->exp = 1.0;
#ifdef LOGGING
  smooth->name = name;
  smooth->updated = 0;
#else
  (void) solver;
  (void) name;
#endif
}

void kissat_update_smooth(kissat *solver, smooth *smooth, double y) {
#ifdef LOGGING
  smooth->updated++;
  const double old_value = smooth->value;
#endif
  const double old_biased = smooth->biased;
  const double alpha = smooth->alpha;
  const double beta = smooth->beta;
  const double delta = y - old_biased;
  const double scaled_delta = alpha * delta;
  const double new_biased = old_biased + scaled_delta;
  LOG("update %" PRIu64 " of biased %s EMA %g with %g (delta %g) "
        "yields %g (scaled delta %g)",
        smooth->updated, smooth->name, old_biased, y, delta,
        new_biased, scaled_delta);
  smooth->biased = new_biased;
  double old_exp = smooth->exp;
  double new_exp, div, new_value;
  if (old_exp) {
    new_exp = old_exp * beta;
    assert(new_exp < 1);
    smooth->exp = new_exp;
    div = 1 - new_exp;
    assert(div > 0);
    new_value = new_biased / div;
  } else {
    new_value = new_biased;
#ifdef LOGGING
    new_exp = 0;
    div = 1;
#endif
  }
  smooth->value = new_value;
  LOG("update %" PRIu64 " of corrected %s EMA %g "
        "with %g (delta %g) yields %g (exponent %g, div %g)",
        smooth->updated, smooth->name, old_value, y, delta,
        new_value, new_exp, div);
#ifndef LOGGING
  (void) solver;
#endif
}

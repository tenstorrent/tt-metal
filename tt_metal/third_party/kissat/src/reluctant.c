#include "internal.h"
#include "logging.h"


void kissat_enable_reluctant(reluctant *reluctant,
      uint64_t period, uint64_t limit) {
  if (limit && period > limit) {
    period = limit;
  }
  reluctant->limited = (limit > 0);
  reluctant->trigger = false;
  reluctant->period = period;
  reluctant->wait = period;
  reluctant->u = reluctant->v = 1;
  reluctant->limit = limit;
}

void kissat_disable_reluctant(reluctant *reluctant) {
  reluctant->period = 0;
}

void kissat_tick_reluctant(reluctant *reluctant) {
  if (!reluctant->period) {
    return;
  }

  if (reluctant->trigger) {
    return;
  }

  assert(reluctant->wait > 0);
  if (--reluctant->wait) {
    return;
  }

  uint64_t u = reluctant->u;
  uint64_t v = reluctant->v;

  if ((u & -u) == v) {
    u++;
    v = 1;
  } else {
    assert(UINT64_MAX / 2 >= v);
    v *= 2;
  }

  assert(v);
  assert(UINT64_MAX / v >= reluctant->period);
  uint64_t wait = v * reluctant->period;

  if (reluctant->limited && wait > reluctant->limit) {
    u = v = 1;
    wait = reluctant->period;
  }

  reluctant->trigger = true;
  reluctant->wait = wait;
  reluctant->u = u;
  reluctant->v = v;
}

void kissat_init_reluctant(kissat *solver) {
  if (GET_OPTION(reluctant)) {
    LOG("enable reluctant doubling with period %d limit %d",
          GET_OPTION(reluctantint), GET_OPTION(reluctantlim));
    kissat_enable_reluctant(&solver->reluctant,
          GET_OPTION(reluctantint),
          GET_OPTION(reluctantlim));
  } else {
    LOG("reluctant doubling disabled and thus no stable restarts");
    kissat_disable_reluctant(&solver->reluctant);
  }
}

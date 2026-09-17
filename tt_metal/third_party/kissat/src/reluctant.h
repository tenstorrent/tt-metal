#ifndef _reluctant_h_INCLUDED
#define _reluctant_h_INCLUDED

#include <stdbool.h>
#include <stdint.h>

typedef struct reluctant reluctant;

struct reluctant {
  bool limited;
  bool trigger;
  uint64_t period;
  uint64_t wait;
  uint64_t u, v;
  uint64_t limit;
};

void kissat_enable_reluctant(reluctant *, uint64_t period, uint64_t limit);
void kissat_disable_reluctant(reluctant *);
void kissat_tick_reluctant(reluctant *);

static inline bool kissat_reluctant_triggered(reluctant *reluctant) {
  if (!reluctant->trigger) {
    return false;
  }
  reluctant->trigger = false;
  return true;
}

struct kissat;

void kissat_init_reluctant(struct kissat *);

#endif

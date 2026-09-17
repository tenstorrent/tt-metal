#ifndef _rephase_h_INCLUDED
#define _rephase_h_INCLUDED

#include <stdbool.h>
#include <stdint.h>

typedef struct rephased rephased;

struct rephased {
  uint64_t count;
  char last;
};

struct kissat;

bool kissat_rephasing(struct kissat *);
void kissat_rephase(struct kissat *);
char kissat_rephase_best(struct kissat *);

void kissat_reset_rephased(struct kissat *);
void kissat_reset_best_assigned(struct kissat *);
void kissat_reset_target_assigned(struct kissat *);

#define REPHASES \
REPHASE (best, 'B', 0) \
REPHASE (inverted, 'I', 1) \
REPHASE (original, 'O', 2) \
REPHASE (walking, 'W', 3)

#endif

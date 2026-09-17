#include "flags.h"
#include "internal.h"
#include "logging.h"
#include "protect.h"

#define PROTECT_MAX UINT_MAX

void kissat_protect_variable(kissat *solver, unsigned idx) {
  assert(GET_OPTION(incremental) || !GET(searches) || PROTECT(idx)
        || FLAGS(idx)->fixed || !FLAGS(idx)->active);

  if (FLAGS(idx)->fixed) {
    LOG("internal %s is fixed, protect does nothing", LOGVAR(idx));
    return;
  }

  kissat_activate_literal(solver, LIT(idx));

  if (PROTECT(idx) < PROTECT_MAX) {
    PROTECT(idx)++;
    LOG("protecting internal %s (counter at %u)", LOGVAR(idx), PROTECT(idx));
  } else {
    LOG("stuck counter, protecting internal %s forever", LOGVAR(idx));
  }
}

void kissat_unprotect_variable(kissat *solver, unsigned idx) {
  assert(PROTECT(idx) || FLAGS(idx)->fixed);

  if (FLAGS(idx)->fixed) {
    LOG("internal %s is fixed, unprotect does nothing", LOGVAR(idx));
  } else if (PROTECT(idx) < PROTECT_MAX) {
    PROTECT(idx)--;
    LOG("unprotecting internal %s (counter at %u)", LOGVAR(idx), PROTECT(idx));
  } else {
    LOG("stuck counter for internal %s, unprotect does noething", LOGVAR(idx));
  }
}

void kissat_move_protection(kissat *solver,
      unsigned from_idx, unsigned to_idx) {
  unsigned move = PROTECT(from_idx);
  if (!move) {
    LOG("no protection to move from %s to %s", LOGVAR(from_idx),
          LOGVAR(to_idx));
    return;
  }
  PROTECT(from_idx) = 0;
  if (FLAGS(to_idx)->fixed) {
    LOG("no protection to move from %s to fixed %s", LOGVAR(from_idx),
          LOGVAR(to_idx));
    return;
  }
  PROTECT(to_idx) += move;
  if (PROTECT(to_idx) < move || PROTECT(to_idx) > PROTECT_MAX) {
    LOG("stuck counter while moving %u protection from %s to %s",
          move, LOGVAR(from_idx), LOGVAR(to_idx));
    PROTECT(to_idx) = PROTECT_MAX;
  } else {
    LOG("moved %u protection from %s to %s",
          move, LOGVAR(from_idx), LOGVAR(to_idx));
  }
}

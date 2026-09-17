#include "internal.h"
#include "logging.h"
#include "nonces.h"

static const size_t size_nonces = 32;

void kissat_init_nonces(kissat *solver) {
  LOG("initializing %zu nonces", size_nonces);
  assert(EMPTY_STACK(solver->nonces));
  generator random = solver->random;
  for (size_t i = 0; i < size_nonces; i++) {
    uint64_t nonce = 1 | kissat_next_random64(&random);
    LOG2("nonce[%zu] = 0x%016" PRIx64, i, nonce);
    PUSH_STACK(solver->nonces, nonce);
  }
}

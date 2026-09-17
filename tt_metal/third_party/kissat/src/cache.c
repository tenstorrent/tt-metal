#include "allocate.h"
#include "internal.h"
#include "logging.h"
#include "nonces.h"
#include "print.h"
#include "random.h"

#include <math.h>

static uint64_t compute_cache_signature(kissat *solver) {
  if (EMPTY_STACK(solver->nonces)) {
    kissat_init_nonces(solver);
  }

  const uint64_t *const begin_nonces = BEGIN_STACK(solver->nonces);
  const uint64_t *const end_nonces = END_STACK(solver->nonces);
  uint64_t const *n = begin_nonces;

  const value *const saved = solver->phases.saved;
  const unsigned vars = VARS;

  assert(vars == solver->cache.vars);
  uint64_t res = 0;

  for (unsigned idx = 0; idx < vars; idx++) {
    const value value = (saved[idx] > 0 ? 1 : -1);
    const int64_t extended = value * (int64_t)(1 + idx);
    res += extended;
    res *= *n++;
    if (n == end_nonces) {
      n = begin_nonces;
    }
  }

  LOG("assignment signature 0x%016" PRIx64 " for %u variables", res, vars);

  return res;
}

static void release_cache_line(kissat *solver, line *line) {
  LOGLINE(line, "releasing");
  kissat_delete_bits(solver, line->bits, line->vars);
}

static void copy_line(kissat *solver, line *l,
      unsigned unsatisfied, uint64_t signature, uint64_t inserted) {
  l->unsatisfied = unsatisfied;
  l->signature = signature;
  l->inserted = inserted;
  const unsigned vars = VARS;
  l->vars = vars;
  bits *bits = l->bits;
  assert(!vars || bits);
  const value *const saved = solver->phases.saved;
  for (unsigned idx = 0; idx < vars; idx++) {
    const bool bit = (saved[idx] > 0);
    kissat_set_bit_explicitly(bits, vars, idx, bit);
  }
  LOGLINE(l, "copied");
}

static void reset_last_looked_up(kissat *solver, cache *cache) {
  assert(cache->looked);
  LOG("reset last looked up at cache[%zu]", cache->last_looked_up_position);
  cache->last_looked_up_position = MAX_SIZE_T;
  cache->looked = false;
#ifndef LOGGING
  (void) solver;
#endif
}

bool kissat_insert_cache(kissat *solver, unsigned unsatisfied) {
  cache *cache = &solver->cache;
  const size_t size = SIZE_STACK(cache->lines);

  if (!cache->valid) {
    if (size)
      kissat_verbose(solver,
            "need to release %zu invalid cache lines", size);
    ADD(cache_released, size);
    kissat_release_cache(solver);
    assert(cache->valid);
  }

  if (cache->looked) {
    reset_last_looked_up(solver, cache);
  }

  assert(solver->vars == VARS);
  assert(cache->vars == VARS);

  const uint64_t inserted = cache->inserted++;
  LOG("insertion attempt %" PRIu64, inserted);

  const uint64_t signature = compute_cache_signature(solver);

  const line *const end = END_STACK(cache->lines);
  line *begin = BEGIN_STACK(cache->lines);
  line *replace = 0;
  for (line *l = begin; l != end; l++) {
    assert(l->vars == VARS);

    if (l->signature == signature) {
      LOGLINE(l, "found same signature");
      return false;
    }

    if (!replace ||
          l->unsatisfied > replace->unsatisfied ||
          (l->unsatisfied == replace->unsatisfied &&
                l->inserted < replace->inserted)) {
      replace = l;
    }
  }

  if (replace && replace->unsatisfied < unsatisfied) {
    kissat_very_verbose(solver,
          "not cached assignment[%" PRIu64 "] with "
          "%u unsatisfied clauses (exceeds maximum %u)",
          inserted, unsatisfied, replace->unsatisfied);
    return false;
  }

  const unsigned limit = kissat_log2_ceiling_of_uint64(inserted + 1);

  if (replace && size >= limit) {
    kissat_very_verbose(solver, "keeping cache size %zu "
          "due to limit %u = log2 (%" PRIu64 ")",
          size, limit, inserted);
    LOGLINE(replace, "evicting");
    assert(replace->vars == VARS);
#ifndef QUIET
    uint64_t replace_inserted = replace->inserted;
    unsigned replace_unsatisfied = replace->unsatisfied;
#endif
    copy_line(solver, replace, unsatisfied, signature, inserted);
    kissat_very_verbose(solver,
          "cached assignment[%" PRIu64 "] "
          "with %u unsatisfied clauses",
          replace->inserted, replace->unsatisfied);
    kissat_very_verbose(solver,
          "replaced cached assignment[%" PRIu64 "] "
          "with %u unsatisfied clauses",
          replace_inserted, replace_unsatisfied);
  } else {
    kissat_very_verbose(solver, "increasing cache size %zu "
          "to limit %u = log2 (%" PRIu64 ")",
          size, limit, inserted);
    line line;
    line.bits = kissat_new_bits(solver, solver->vars);
    copy_line(solver, &line, unsatisfied, signature, inserted);
    PUSH_STACK(cache->lines, line);
    kissat_very_verbose(solver, "cached assignment[%" PRIu64 "] "
          "with %u unsatisfied clauses",
          line.inserted, line.unsatisfied);
  }

  INC(cache_inserted);
  return true;
}

void kissat_update_cache(kissat *solver, unsigned unsatisfied) {
  cache *cache = &solver->cache;
  assert(cache->valid);
  assert(cache->looked);
  const size_t last = cache->last_looked_up_position;
  assert(last < SIZE_STACK(cache->lines));
  line *line = &PEEK_STACK(cache->lines, last);
  if (line->unsatisfied == unsatisfied) {
    return;
  }
  kissat_very_verbose(solver,
        "updating cached assignment[%" PRIu64 "] "
        "unsatisfied clauses from %u to %u",
        line->inserted, line->unsatisfied, unsatisfied);
  line->unsatisfied = unsatisfied;
  INC(cache_updated);
}

bits *kissat_lookup_cache(kissat *solver) {
  cache *cache = &solver->cache;
  if (!cache->valid) {
    kissat_very_verbose(solver, "can not use invalid assignment cache");
    return 0;
  }

  const size_t size = SIZE_STACK(cache->lines);
  if (!size) {
    kissat_very_verbose(solver, "can not use empty assignment cache");
    return 0;
  }
#ifndef QUIET
  kissat_very_verbose(solver, "using assignment cache of size %zu", size);
  if (kissat_verbosity(solver) > 2) {
    for (size_t i = 0; i < size; i++) {
      line *line = &PEEK_STACK(cache->lines, i);
      kissat_extremely_verbose(solver,
            "cache[%zu] contains "
            "assignment[%" PRIu64 "] "
            "unsatisfied %u",
            i, line->inserted, line->unsatisfied);
    }
  }
#endif
  assert(size <= UINT_MAX);
  unsigned pos;
  if (GET_OPTION(cachesample)) {
    double sum = 0;
    for (size_t i = 0; i < size; i++) {
      line *l = &PEEK_STACK(cache->lines, i);
      const double score = 1.0 / (l->unsatisfied + 1.0);
      const double next = sum + score;
      kissat_extremely_verbose(solver,
            "cache[%zu] score %.7f of "
            "assignment[%" PRIu64 "] unsatisfied %u "
            "in range %.7f ... %.7f", i, score,
            l->inserted, l->unsatisfied, sum, next);
      sum = next;
    }
    const double fraction = kissat_pick_double(&solver->random);
    const double limit = fraction * sum;
    kissat_extremely_verbose(solver,
          "cache picking limit %g = fraction %g * sum %g",
          limit, fraction, sum);
    pos = 0;
    double tmp = 0;
    for (size_t i = 0; i < size; i++) {
      line *l = &PEEK_STACK(cache->lines, i);
      double score = 1.0 / (l->unsatisfied + 1.0);
      tmp += score;
      if (tmp >= limit) {
        pos = i;
        break;
      }
    }
  } else {
    pos = kissat_pick_random(&solver->random, 0, size);
  }

  LOG("looking up line %u in cache of size %zu", pos, size);
  line *l = &PEEK_STACK(cache->lines, pos);
  LOGLINE(l, "looked up");
  kissat_very_verbose(solver,
        "using previous cached assignment[%" PRIu64 "] "
        "with %u unsatisfied clauses",
        l->inserted, l->unsatisfied);
  INC(cache_reused);

  cache->looked = true;
  cache->last_looked_up_position = pos;

  return l->bits;
}

void kissat_release_cache(kissat *solver) {
  LOG("releasing cache of size %zu", SIZE_STACK(solver->cache.lines));
  for (all_stack(line, l, solver->cache.lines)) {
    release_cache_line(solver, &l);
  }
  RELEASE_STACK(solver->cache.lines);
  solver->cache.vars = solver->vars;
  solver->cache.valid = true;
}

#include "allocate.h"
#include "error.h"
#include "internal.h"
#include "logging.h"

#undef LOGPREFIX
#define LOGPREFIX "ALLOCATE"

#include <string.h>

#ifdef LOGGING
#include <inttypes.h>
#endif

static void inc_bytes(kissat *solver, size_t bytes) {
#ifdef METRICS
  if (!solver) {
    return;
  }
  ADD(allocated_current, bytes);
  LOG5("allocated_current = %s",
        FORMAT_BYTES(solver->statistics.allocated_current));
  if (solver->statistics.allocated_current >=
        solver->statistics.allocated_max) {
    solver->statistics.allocated_max = solver->statistics.allocated_current;
    LOG5("allocated_max = %s",
          FORMAT_BYTES(solver->statistics.allocated_max));
  }
#else
  (void) solver;
  (void) bytes;
#endif
}

static void dec_bytes(kissat *solver, size_t bytes) {
#ifdef METRICS
  if (!solver) {
    return;
  }
  SUB(allocated_current, bytes);
  LOG5("allocated_current = %s",
        FORMAT_BYTES(solver->statistics.allocated_current));
#else
  (void) solver;
  (void) bytes;
#endif
}

void *kissat_malloc(kissat *solver, size_t bytes) {
  void *res;
  if (!bytes) {
    return 0;
  }
  res = malloc(bytes);
  LOG4("malloc (%zu) = %p", bytes, res);
  if (!res) {
    kissat_fatal("out-of-memory allocating %zu bytes", bytes);
  }
  inc_bytes(solver, bytes);
  return res;
}

void kissat_free(kissat *solver, void *ptr, size_t bytes) {
  if (ptr) {
    LOG4("free (%p[%zu])", ptr, bytes);
    dec_bytes(solver, bytes);
    free(ptr);
  } else {
    assert(!bytes);
  }
}

void *kissat_nalloc(kissat *solver, size_t n, size_t size) {
  void *res;
  if (!n || !size) {
    return 0;
  }
  if (MAX_SIZE_T / size < n) {
    kissat_fatal("invalid 'kissat_nalloc (..., %zu, %zu)' call", n, size);
  }
  const size_t bytes = n * size;
  res = malloc(bytes);
  LOG4("nalloc (%zu, %zu) = %p", n, size, res);
  if (!res)
    kissat_fatal("out-of-memory allocating "
          "%zu = %zu x %zu bytes", bytes, n, size);
  inc_bytes(solver, bytes);
  return res;
}


void *kissat_calloc(kissat *solver, size_t n, size_t size) {
  void *res;
  if (!n || !size) {
    return 0;
  }
  if (MAX_SIZE_T / size < n) {
    kissat_fatal("invalid 'kissat_calloc (..., %zu, %zu)' call", n, size);
  }
  res = calloc(n, size);
  LOG4("calloc (%zu, %zu) = %p", n, size, res);
  const size_t bytes = n * size;
  if (!res)
    kissat_fatal("out-of-memory allocating "
          "%zu = %zu x %zu bytes", bytes, n, size);
  inc_bytes(solver, bytes);
  return res;
}

void kissat_dealloc(kissat *solver, void *ptr, size_t n, size_t size) {
  if (!n || !size) {
    return;
  }
  if (MAX_SIZE_T / size < n) {
    kissat_fatal("invalid 'kissat_dealloc (..., %zu, %zu)' call", n, size);
  }
  const size_t bytes = n * size;
  kissat_free(solver, ptr, bytes);
}

void *kissat_realloc(kissat *solver, void *p, size_t old_bytes,
      size_t new_bytes) {
  if (old_bytes == new_bytes) {
    return p;
  }
  if (!new_bytes) {
    kissat_free(solver, p, old_bytes);
    return 0;
  }
  dec_bytes(solver, old_bytes);
  void *res = realloc(p, new_bytes);
  LOG4("realloc (%p[%zu], %zu) = %p", p, old_bytes, new_bytes, res);
  if (new_bytes && !res)
    kissat_fatal("out-of-memory reallocating from %zu to %zu bytes",
          old_bytes, new_bytes);
  inc_bytes(solver, new_bytes);
  return res;
}

void *kissat_nrealloc(kissat *solver, void *p, size_t o, size_t n,
      size_t size) {
  if (!size) {
    assert(!p);
    assert(!o);
    return 0;
  }
  const size_t max = MAX_SIZE_T / size;
  if (max < o || max < n)
    kissat_fatal("invalid 'kissat_nrealloc (..., %zu, %zu, %zu)' call",
          o, n, size);
  return kissat_realloc(solver, p, o * size, n * size);
}

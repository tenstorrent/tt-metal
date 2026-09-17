#include "allocate.h"
#include "stack.h"
#include "utilities.h"

#include <assert.h>

void kissat_stack_enlarge(struct kissat *solver, chars *s, size_t bytes) {
  const size_t size = SIZE_STACK(*s);
  const size_t old_bytes = CAPACITY_STACK(*s);
  assert(MAX_SIZE_T / 2 >= old_bytes);
  size_t new_bytes;
  if (old_bytes) {
    new_bytes = 2 * old_bytes;
  } else {
    new_bytes = bytes;
    while (!kissat_aligned_word(new_bytes)) {
      new_bytes <<= 1;
    }
  }
  s->begin = kissat_realloc(solver, s->begin, old_bytes, new_bytes);
  s->allocated = s->begin + new_bytes;
  s->end = s->begin + size;
}

void kissat_shrink_stack(struct kissat *solver, chars *s, size_t bytes) {
  assert(bytes > 0);
  const size_t old_bytes_capacity = CAPACITY_STACK(*s);
  assert(kissat_aligned_word(old_bytes_capacity));
  assert(!(old_bytes_capacity % bytes));
  assert(kissat_is_zero_or_power_of_two(old_bytes_capacity / bytes));
  const size_t old_bytes_size = SIZE_STACK(*s);
  assert(!(old_bytes_size % bytes));
  const size_t old_size = old_bytes_size / bytes;
  size_t new_capacity;
  if (old_size) {
    const unsigned ld_old_size = kissat_log2_ceiling_of_word(old_size);
    new_capacity = ((size_t) 1) << ld_old_size;
  } else {
    new_capacity = 0;
  }
  assert(kissat_is_zero_or_power_of_two(new_capacity));
  size_t new_bytes_capacity = new_capacity * bytes;
  while (!kissat_aligned_word(new_bytes_capacity)) {
    new_bytes_capacity <<= 1;
  }
  if (new_bytes_capacity == old_bytes_capacity) {
    return;
  }
  assert(new_bytes_capacity < old_bytes_capacity);
  s->begin = kissat_realloc(solver, s->begin,
              old_bytes_capacity, new_bytes_capacity);
  s->allocated = s->begin + new_bytes_capacity;
  s->end = s->begin + old_bytes_size;
  assert(s->end <= s->allocated);
}

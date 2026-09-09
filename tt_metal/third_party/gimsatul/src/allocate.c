#include "allocate.h"
#include "geatures.h"
#include "message.h"

void *allocate_block (size_t bytes) {
  void *res = malloc (bytes);
  if (bytes && !res)
    fatal_error ("out-of-memory allocating %zu bytes", bytes);
  return res;
}

void *allocate_and_clear_block (size_t bytes) {
  void *res = calloc (1, bytes);
  if (bytes && !res)
    fatal_error ("out-of-memory allocating %zu bytes", bytes);
  return res;
}

void *allocate_array (size_t num, size_t bytes) {
  size_t actual_bytes = num * bytes;
  void *res = malloc (actual_bytes);
  if (actual_bytes && !res)
    fatal_error ("out-of-memory allocating %zu*%zu bytes", num, bytes);
  return res;
}

void *allocate_and_clear_array (size_t num, size_t bytes) {
  void *res = calloc (num, bytes);
  if (num && bytes && !res)
    fatal_error ("out-of-memory allocating %zu*%zu bytes", num, bytes);
  return res;
}

void *reallocate_block (void *ptr, size_t bytes) {
  void *res = realloc (ptr, bytes);
  if (bytes && !res)
    fatal_error ("out-of-memory reallocating %zu bytes", bytes);
  return res;
}

#include <assert.h>
#include <string.h>

#ifndef NDEBUG
#include "utilities.h"
#endif

#ifdef HAVE_MEMALIGN
#include <malloc.h>
#endif

void *allocate_aligned_array (size_t alignment, size_t num, size_t bytes) {
  assert (num);
  assert (bytes);
  assert (is_power_of_two (alignment));
  assert (alignment >= 2 * sizeof (size_t));
  size_t total = num * bytes;
  assert (bytes <= total);
  assert (num <= total);
  void *res;
#if defined(GIMSATUL_HAS_POSIX_MEMALIGN) || defined(GIMSATUL_HAVE_MEMALIGN)
#ifdef GIMSATUL_HAS_POSIX_MEMALIGN
  if (posix_memalign (&res, alignment, total))
    res = 0;
#else
  res = memalign (alignment, total);
#endif
  if (!res)
    fatal_error ("can not allocate %zu aligned %zu = %zu * %zu bytes",
                 alignment, total, num, bytes);
#else
  /*
          8       payxxxx   4       payxxxx
  0 1 2 3 4 5 6 7 8 9 a b   0 1 2 3 4 5 6 7 8 9 a b
  0 1 2 3 4 5 6 7 8 9 a b   0 1 2 3 4 5 6 7 8 9 a b

        7       payxxxx     but still need 7 here!
  1 2 3 4 5 6 7 8 9 a b c   (would only save one word)
  0 1 2 3 4 5 6 7 8 9 a b

      6       payxxxx
  2 3 4 5 6 7 8 9 a b c d
  0 1 2 3 4 5 6 7 8 9 a b

    5       payxxxx
  3 4 5 6 7 8 9 a b c d e
  0 1 2 3 4 5 6 7 8 9 a b

  */
  size_t adjustment = 2 * alignment;
  size_t allocate = total + adjustment;
  assert (adjustment < allocate);
  assert (total < allocate);
  size_t *start = allocate_block (allocate);
  uintptr_t word = (uintptr_t) start;
  word &= ~((uintptr_t) alignment - 1);
  word += alignment;
  size_t *middle = (size_t *) word;
  assert (start <= middle);
  word += alignment;
  res = (size_t *) word;
  assert ((char *) res + total <= (char *) start + allocate);
  middle[0] = (char *) res - (char *) start;
#ifndef NDEBUG
  middle[1] = alignment;
#endif
#endif
  return res;
}

void deallocate_aligned (size_t alignment, void *ptr) {
#if defined(GIMSATUL_HAS_POSIX_MEMALIGN) || defined(GIMSATUL_HAVE_MEMALIGN)
  free (ptr);
#else
  assert (is_power_of_two (alignment));
  uintptr_t word = (uintptr_t) ptr;
  assert (!(word & (alignment - 1)));
  word -= alignment;
  size_t *middle = (size_t *) word;
  assert (middle[1] == alignment);
  char *start = (char *) ptr - middle[0];
  free (start);
#endif
}

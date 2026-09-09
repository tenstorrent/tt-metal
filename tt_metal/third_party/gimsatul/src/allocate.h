#ifndef _allocate_h_INCLUDED
#define _allocate_h_INCLUDED

#include <stdlib.h>

void *allocate_block (size_t bytes);
void *allocate_and_clear_block (size_t bytes);
void *allocate_array (size_t num, size_t bytes);
void *allocate_and_clear_array (size_t num, size_t bytes);
void *reallocate_block (void *ptr, size_t bytes);
void *allocate_array (size_t num, size_t bytes);

void *allocate_aligned_array (size_t alignment, size_t num, size_t bytes);

void deallocate_aligned (size_t alignment, void *ptr);

#define FREE(PTR) \
  do { \
    free (PTR); \
    (PTR) = 0; \
  } while (0)

#endif

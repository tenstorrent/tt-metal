#include "set.h"
#include "allocate.h"
#include "random.h"

#include <assert.h>
#include <stdbool.h>

#ifndef NDEBUG
#include "utilities.h"
#endif

#define DELETED ((void *) ~(size_t) 0)

static size_t hash_pointer (struct set *set, void *ptr) {
  size_t res;
  if (set->hash.function)
    res = set->hash.function (set->hash.state, ptr);
  else
    res = (size_t) ptr;
  return res;
}

static size_t hash_pointer_to_position (struct set *set, void *ptr) {
  size_t res = hash_pointer (set, ptr);
  res *= 1111111121u;
  return res;
}

static size_t hash_pointer_to_delta (struct set *set, void *ptr) {
  size_t res = hash_pointer (set, ptr);
  res *= 2222222243u;
  return res;
}

static size_t reduce_hash (size_t hash, size_t allocated) {
  assert (allocated);
  assert (is_power_of_two (allocated));
  size_t res = hash;
  if (allocated >= (size_t) 1 << 32)
    res ^= res >> 32;
  if (allocated >= (size_t) 1 << 16)
    res ^= res >> 16;
  if (allocated >= (size_t) 1 << 8)
    res ^= res >> 8;
  res &= allocated - 1;
  assert (res < allocated);
  return res;
}

static size_t reduce_delta (size_t hash, size_t allocated) {
  return reduce_hash (hash, allocated) | 1;
}

#ifndef NDEBUG

static bool set_contains (struct set *set, void *ptr) {
  assert (ptr);
  assert (ptr != DELETED);
  size_t size = set->size;
  if (!size)
    return false;
  size_t hash = hash_pointer_to_position (set, ptr);
  size_t allocated = set->allocated;
  size_t start = reduce_hash (hash, allocated);
  void **table = set->table;
  void *tmp = table[start];
  if (!tmp)
    return false;
  if (tmp == ptr)
    return true;
  hash = hash_pointer_to_delta (set, ptr);
  size_t delta = reduce_delta (hash, allocated);
  size_t pos = start;
  assert (allocated < 2 || (delta & 1));
  for (;;) {
    pos += delta;
    if (pos >= allocated)
      pos -= allocated;
    assert (pos < allocated);
    if (pos == start)
      return false;
    tmp = table[pos];
    if (!tmp)
      return false;
    if (tmp == ptr)
      return true;
  }
}

#endif

static void enlarge_set (struct set *set);
static void shrink_set (struct set *set);

void set_insert (struct set *set, void *ptr) {
  assert (ptr);
  assert (ptr != DELETED);
  if (set->size + set->deleted >= set->allocated / 2)
    enlarge_set (set);
  size_t hash = hash_pointer_to_position (set, ptr);
  size_t allocated = set->allocated;
  size_t start = reduce_hash (hash, allocated);
  void **table = set->table;
  size_t pos = start;
  void *tmp = table[pos];
  if (tmp && tmp != DELETED) {
    hash = hash_pointer_to_delta (set, ptr);
    size_t delta = reduce_delta (hash, allocated);
    assert (delta & 1);
    do {
      pos += delta;
      if (pos >= allocated)
        pos -= allocated;
      assert (pos < allocated);
      assert (pos != start);
      tmp = table[pos];
    } while (tmp && tmp != DELETED);
  }
  if (tmp == DELETED) {
    assert (set->deleted);
    set->deleted--;
  }
  set->size++;
  table[pos] = ptr;
  assert (set_contains (set, ptr));
}

void set_remove (struct set *set, void *ptr) {
  assert (ptr);
  assert (ptr != DELETED);
  assert (set_contains (set, ptr));
  assert (set->size);
  if (set->allocated > 16 && set->size <= set->allocated / 8)
    shrink_set (set);
  size_t hash = hash_pointer_to_position (set, ptr);
  size_t allocated = set->allocated;
  size_t start = reduce_hash (hash, allocated);
  void **table = set->table;
  size_t pos = start;
  void *tmp = table[pos];
  if (tmp != ptr) {
    assert (tmp);
    hash = hash_pointer_to_delta (set, ptr);
    size_t delta = reduce_delta (hash, allocated);
    assert (delta & 1);
    do {
      pos += delta;
      if (pos >= allocated)
        pos -= allocated;
      assert (pos < allocated);
      assert (pos != start);
      tmp = table[pos];
      assert (tmp);
    } while (tmp != ptr);
  }
  table[pos] = DELETED;
  set->deleted++;
  set->size--;
}

static void resize_set (struct set *set, size_t new_allocated) {
  assert (new_allocated != set->allocated);
  void **old_table = set->table;
  unsigned old_allocated = set->allocated;
  set->allocated = new_allocated;
#ifndef NDEBUG
  size_t old_size = set->size;
#endif
  assert (old_size < new_allocated);
  set->size = set->deleted = 0;
  set->table = allocate_and_clear_array (new_allocated, sizeof *set->table);
  void **end = old_table + old_allocated;
  for (void **p = old_table, *ptr; p != end; p++)
    if ((ptr = *p) && ptr != DELETED)
      set_insert (set, ptr);
  assert (set->size == old_size);
  assert (set->allocated == new_allocated);
  free (old_table);
}

static void enlarge_set (struct set *set) {
  size_t old_allocated = set->allocated;
  size_t new_allocated = old_allocated ? 2 * old_allocated : 2;
  resize_set (set, new_allocated);
}

static void shrink_set (struct set *set) {
  size_t old_allocated = set->allocated;
  size_t new_allocated = old_allocated / 2;
  resize_set (set, new_allocated);
}

void *random_set (uint64_t *random, struct set *set) {
  assert (set->size);
  size_t allocated = set->allocated;
  size_t pos = random_modulo (random, allocated);
  void **table = set->table;
  void *res = table[pos];
  while (!res || res == DELETED) {
    if (++pos == allocated)
      pos = 0;
    res = table[pos];
  }
  return res;
}

#ifndef _inlinereap_h_INCLUDED
#define _inlinereap_h_INCLUDED

#include "allocate.h"
#include "inline.h"
#include "logging.h"
#include "reap.h"
#include "utilities.h"

static inline void kissat_clear_reap(kissat *solver, reap *reap) {
  LOG("clearing radix heap buckets[%u..%u]",
        reap->min_bucket, reap->max_bucket);
  const unsigned max_bucket = reap->max_bucket;
  assert(max_bucket <= 32);
  for (unsigned i = reap->min_bucket; i <= max_bucket; i++) {
    CLEAR_STACK(reap->buckets[i]);
  }
  reap->num_elements = 0;
  reap->last_deleted = 0;
  reap->min_bucket = 32;
  reap->max_bucket = 0;
#ifndef LOGGING
  (void) solver;
#endif
}

static inline void kissat_push_reap(kissat *solver, reap *reap, unsigned e) {
  assert(reap->last_deleted <= e);
  const unsigned diff = e ^ reap->last_deleted;
  const unsigned bucket = 32 - kissat_leading_zeroes_of_unsigned(diff);
  LOG("push %u difference %u to radix heap bucket[%u]", e, diff, bucket);
  PUSH_STACK(reap->buckets[bucket], e);
  if (reap->min_bucket > bucket) {
    reap->min_bucket = bucket;
  }
  if (reap->max_bucket < bucket) {
    reap->max_bucket = bucket;
  }
  assert(reap->num_elements != UINT_MAX);
  reap->num_elements++;
}

static inline unsigned kissat_pop_reap(kissat *solver, reap *reap) {
  assert(reap->num_elements > 0);
  unsigned i = reap->min_bucket;
  for (;;) {
    assert(i < 33);
    assert(i <= reap->max_bucket);

    unsigneds *s = reap->buckets + i;

    if (EMPTY_STACK(*s)) {
      LOG("empty radix heap bucket[%u]", i);
      reap->min_bucket = ++i;
      continue;
    }

    LOG("first non-empty radix heap bucket[%u]", i);

    unsigned res;

    if (i) {
      res = UINT_MAX;
      unsigned *begin = BEGIN_STACK(*s);
      const unsigned *const end = END_STACK(*s);
      assert(begin < end);
      unsigned const *q = 0;

      for (const unsigned *p = begin; p != end; p++) {
        const unsigned tmp = *p;
        if (tmp >= res) {
          continue;
        }
        res = tmp;
        q = p;
      }
      assert(q);
      s->end = begin;

      for (const unsigned *p = begin; p != end; p++) {
        if (p == q) {
          continue;
        }
        const unsigned other = *p;
        const unsigned diff = other ^ res;
        assert(sizeof(unsigned) == 4);
        const unsigned j =
              32 - kissat_leading_zeroes_of_unsigned(diff);
        assert(j < i);
        unsigneds *t = reap->buckets + j;
        assert(t < s);
        PUSH_STACK(*t, other);
        LOG("moving %u difference %u to radix heap bucket[%u]",
              other, diff, j);
        if (reap->min_bucket > j) {
          reap->min_bucket = j;
        }
      }
      CLEAR_STACK(*s);

      if (i && reap->max_bucket == i) {
#ifndef NDEBUG
        for (unsigned j = i + 1; j < 33; j++) {
          assert(EMPTY_STACK(reap->buckets[j]));
        }
#endif
        assert(EMPTY_STACK(*s));
        reap->max_bucket = i - 1;
      }
    } else {
      res = reap->last_deleted;
      assert(PEEK_STACK(reap->buckets[0], 0) == res);
      assert(!EMPTY_STACK(reap->buckets[0]));
      reap->buckets[0].end--;
    }

    if (reap->min_bucket == i) {
#ifndef NDEBUG
      for (unsigned j = 0; j < i; j++) {
        assert(EMPTY_STACK(reap->buckets[j]));
      }
#endif
      if (EMPTY_STACK(*s)) {
        reap->min_bucket = MIN(i + 1, 32);
      }
    }

    reap->num_elements--;
    assert(reap->last_deleted <= res);
    reap->last_deleted = res;

    LOG("pop %u from radix heap bucket[%u]", res, i);

    return res;
  }
}

#endif

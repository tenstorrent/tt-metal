#ifndef HYPER_PROPAGATION

static inline void kissat_watch_large_delayed(kissat *solver,
      watches *all_watches, unsigneds *delayed) {
  assert(all_watches == solver->watches);
  assert(delayed == &solver->delayed);
  const unsigned *const end_delayed = END_STACK(*delayed);
  unsigned const *d = BEGIN_STACK(*delayed);
  while (d != end_delayed) {
    const unsigned lit = *d++;
    assert(d != end_delayed);
    const watch watch = {.raw = *d++ };
    assert(!watch.type.binary);
    assert(lit < LITS);
    watches *const lit_watches = all_watches + lit;
    assert(d != end_delayed);
    const reference ref = *d++;
    const unsigned blocking = watch.blocking.lit;
    LOGREF(ref, "watching %s blocking %s in", LOGLIT(lit),
          LOGLIT(blocking));
    kissat_push_blocking_watch(solver, lit_watches, blocking, ref);
  }
  CLEAR_STACK(*delayed);
}

#endif

#if defined(HYPER_PROPAGATION) || defined (PROBING_PROPAGATION)

static inline void kissat_update_probing_propagation_statistics(kissat *solver,
      unsigned propagated) {
  const uint64_t ticks = solver->ticks;
  LOG(PROPAGATION_TYPE " propagation took %u propagations", propagated);
  LOG(PROPAGATION_TYPE " propagation took %" PRIu64 " ticks", ticks);

  ADD(propagations, propagated);
  ADD(probing_propagations, propagated);

#if defined(METRICS)
  if (solver->backbone_computing) {
    ADD(backbone_propagations, propagated);
    ADD(backbone_ticks, ticks);
  }
  if (solver->failed_probing) {
    ADD(failed_propagations, propagated);
    ADD(failed_ticks, ticks);
  }
  if (solver->transitive_reducing) {
    ADD(transitive_propagations, propagated);
    ADD(transitive_ticks, ticks);
  }
  if (solver->vivifying) {
    ADD(vivify_propagations, propagated);
    ADD(vivify_ticks, ticks);
  }
#endif

  ADD(probing_ticks, ticks);
  ADD(ticks, ticks);
}

#endif

static inline void kissat_delay_watching_large(kissat *solver,
      unsigneds *const delayed,
      unsigned lit, unsigned other, reference ref) {
  const watch watch = kissat_blocking_watch(other);
  PUSH_STACK(*delayed, lit);
  PUSH_STACK(*delayed, watch.raw);
  PUSH_STACK(*delayed, ref);
}

static inline clause *
PROPAGATE_LITERAL(kissat *solver,
#if defined(HYPER_PROPAGATION) || defined(PROBING_PROPAGATION)
      const clause *const ignore,
#endif
      const unsigned lit) {
  assert(solver->watching);
  LOG(PROPAGATION_TYPE " propagating %s", LOGLIT(lit));
  assert(VALUE(lit) > 0);
  assert(EMPTY_STACK(solver->delayed));

  watches *const all_watches = solver->watches;
  ward *const arena = BEGIN_STACK(solver->arena);
  assigned *const assigned = solver->assigned;
  value *const values = solver->values;

  const unsigned not_lit = NOT(lit);
#ifdef HYPER_PROPAGATION
  const bool hyper = GET_OPTION(hyper);
#endif

  assert(not_lit < LITS);
  watches *watches = all_watches + not_lit;

  watch *const begin_watches = BEGIN_WATCHES(*watches);
  const watch *const end_watches = END_WATCHES(*watches);

  watch *q = begin_watches;
  const watch *p = q;

  unsigneds *const delayed = &solver->delayed;

  const size_t size_watches = SIZE_WATCHES(*watches);
  uint64_t ticks = 1 + kissat_cache_lines(size_watches, sizeof(watch));
#ifndef HYPER_PROPAGATION
  const unsigned idx = IDX(lit);
  struct assigned *const a = assigned + idx;
  const bool probing = solver->probing;
  const unsigned level = a->level;
#endif
  clause *res = 0;

  while (p != end_watches) {
    const watch head = *q++ = *p++;
    const unsigned blocking = head.blocking.lit;
    assert(VALID_INTERNAL_LITERAL(blocking));
    const value blocking_value = values[blocking];
    if (head.type.binary) {
#ifdef HYPER_PROPAGATION
      assert(blocking_value > 0);
#else
      if (blocking_value > 0) {
        continue;
      }
      const bool redundant = head.binary.redundant;
      if (blocking_value < 0) {
        res = kissat_binary_conflict(solver, redundant,
                    not_lit, blocking);
        break;
      } else {
        assert(!blocking_value);
        kissat_fast_binary_assign(solver, probing, level,
              values, assigned,
              redundant, blocking, not_lit);
        ticks++;
      }
#endif
    } else {
      const watch tail = *q++ = *p++;
      if (blocking_value > 0) {
        continue;
      }
      const reference ref = tail.raw;
      assert(ref < SIZE_STACK(solver->arena));
      clause *const c = (clause *)(arena + ref);
#if defined(HYPER_PROPAGATION) || defined(PROBING_PROPAGATION)
      if (c == ignore) {
        continue;
      }
#endif
      ticks++;
      if (c->garbage) {
        q -= 2;
        continue;
      }
      unsigned *const lits = BEGIN_LITS(c);
      const unsigned other = lits[0] ^ lits[1] ^ not_lit;
      assert(lits[0] != lits[1]);
      assert(VALID_INTERNAL_LITERAL(other));
      assert(not_lit != other);
      assert(lit != other);
      const value other_value = values[other];
      if (other_value > 0) {
        q[-2].blocking.lit = other;
      } else {
        const unsigned *const end_lits = lits + c->size;
        unsigned *const searched = lits + c->searched;
        assert(c->lits + 2 <= searched);
        assert(searched < end_lits);
        unsigned *r, replacement = INVALID_LIT;
        value replacement_value = -1;
        for (r = searched; r != end_lits; r++) {
          replacement = *r;
          assert(VALID_INTERNAL_LITERAL(replacement));
          replacement_value = values[replacement];
          if (replacement_value >= 0) {
            break;
          }
        }
        if (replacement_value < 0) {
          for (r = lits + 2; r != searched; r++) {
            replacement = *r;
            assert(VALID_INTERNAL_LITERAL(replacement));
            replacement_value = values[replacement];
            if (replacement_value >= 0) {
              break;
            }
          }
        }

        if (replacement_value >= 0) {
          c->searched = r - lits;
        }

        if (replacement_value > 0) {
          assert(replacement != INVALID_LIT);
          q[-2].blocking.lit = replacement;
        } else if (!replacement_value) {
          assert(replacement != INVALID_LIT);
          LOGREF(ref, "unwatching %s in", LOGLIT(not_lit));
          q -= 2;
          lits[0] = other;
          lits[1] = replacement;
          assert(lits[0] != lits[1]);
          *r = not_lit;
          kissat_delay_watching_large(solver, delayed,
                replacement, other, ref);
          ticks++;
        } else if (other_value) {
          assert(replacement_value < 0);
          assert(blocking_value < 0);
          assert(other_value < 0);
          LOGREF(ref, "conflicting");
          res = c;
          break;
        }
#ifdef HYPER_PROPAGATION
        else if (hyper) {
          assert(replacement_value < 0);
          unsigned dom = kissat_find_dominator(solver, other, c);
          if (dom != INVALID_LIT) {
            LOGBINARY(dom, other, "hyper binary resolvent");

            INC(hyper_binary_resolved);
            INC(clauses_added);

            INC(hyper_binaries);
            INC(clauses_redundant);

            CHECK_AND_ADD_BINARY(dom, other);
            ADD_BINARY_TO_PROOF(dom, other);

            kissat_assign_binary_at_level_one(solver,
                  values, assigned,
                  true, other, dom);

            delay_watching_hyper(solver, delayed, dom, other);
            delay_watching_hyper(solver, delayed, other, dom);

            kissat_delay_watching_large(solver, delayed,
                  not_lit, other, ref);

            LOGREF(ref, "unwatching %s in", LOGLIT(not_lit));
            q -= 2;
          } else
            kissat_fast_assign_reference(solver, values,
                  assigned, other, ref, c);
          ticks++;
        }
#endif
        else {
          assert(replacement_value < 0);
          kissat_fast_assign_reference(solver, values,
                assigned, other, ref, c);
          ticks++;
        }
      }
    }
  }
  solver->ticks += ticks;

  while (p != end_watches) {
    *q++ = *p++;
  }
  SET_END_OF_WATCHES(*watches, q);

#ifdef HYPER_PROPAGATION
  watch_hyper_delayed(solver, all_watches, delayed);
#else
  kissat_watch_large_delayed(solver, all_watches, delayed);
#endif

  return res;
}

#ifndef HYPER_PROPAGATION

static inline void kissat_update_conflicts_and_trail(kissat *solver,
      clause *conflict, bool flush) {
  if (conflict) {
    INC(conflicts);
    LOG(PROPAGATION_TYPE " propagation on root-level failed");
    if (!solver->level) {
      solver->inconsistent = true;
      CHECK_AND_ADD_EMPTY();
      ADD_EMPTY_TO_PROOF();
    }
  } else if (flush && !solver->level && solver->unflushed) {
    kissat_flush_trail(solver);
  }
}

#endif

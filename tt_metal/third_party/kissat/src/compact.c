#include "compact.h"
#include "inline.h"
#include "inlineheap.h"
#include "print.h"
#include "resize.h"

#include <string.h>

static void reimport_literal(kissat *solver, unsigned eidx, unsigned mlit) {
  import *import = &PEEK_STACK(solver->import, eidx);
  assert(import->imported);
  assert(!import->eliminated);
  LOG("reimporting external variable %u as internal literal %u (was %u)",
        eidx, mlit, import->lit);
  import->lit = mlit;
}

unsigned kissat_compact_literals(kissat *solver, unsigned *mfixed_ptr) {
  INC(compacted);
#if !defined(QUIET) || !defined(NDEBUG)
  const unsigned active = solver->active;
#ifndef QUIET
  const unsigned inactive = solver->vars - active;
  kissat_phase(solver, "compact", GET(compacted),
        "compacting garbage collection "
        "(%u inactive variables %.2f%%)",
        inactive, kissat_percent(inactive, solver->vars));
#endif
#endif
#ifdef LOGGING
  assert(!solver->compacting);
  solver->compacting = true;
#endif
  unsigned mfixed = INVALID_LIT;
  unsigned vars = 0;
  for (all_variables(iidx)) {
    const flags *const flags = FLAGS(iidx);
    if (flags->eliminated) {
      continue;
    }
    const unsigned ilit = LIT(iidx);
    unsigned mlit;
    if (flags->fixed) {
      const value value = kissat_fixed(solver, ilit);
      assert(value);
      if (mfixed == INVALID_LIT) {
        mlit = mfixed = LIT(vars);
        LOG2("first fixed %u mapped to %u assigned to %d",
              ilit, mfixed, value);
        if (value < 0) {
          mfixed = NOT(mfixed);
        }
        LOG2("all other fixed mapped to %u", mfixed);
        vars++;
      } else if (value < 0) {
        mlit = NOT(mfixed);
        LOG2("negatively fixed %u mapped to %u", ilit, mlit);
      } else {
        mlit = mfixed;
        LOG2("positively fixed %u mapped to %u", ilit, mlit);
      }
    } else if (flags->active) {
      assert(flags->active);
      mlit = LIT(vars);
      LOG2("remapping %u to %u", ilit, mlit);
      vars++;
    } else {
      const int elit = PEEK_STACK(solver->export, iidx);
      if (elit) {
        const unsigned eidx = ABS(elit);
        import *import = &PEEK_STACK(solver->import, eidx);
        assert(import->imported);
        assert(!import->eliminated);
        import->imported = false;
        LOG2("external variable %d not imported anymore", eidx);
        POKE_STACK(solver->export, iidx, 0);
      } else {
        LOG2("skipping inactive %u", ilit);
      }
      continue;
    }
    assert(mlit <= ilit);
    assert(mlit != NOT(ilit));
    if (mlit == ilit) {
      continue;
    }
    const int elit = PEEK_STACK(solver->export, iidx);
    const unsigned eidx = ABS(elit);
    if (elit < 0) {
      mlit = NOT(mlit);
    }
    reimport_literal(solver, eidx, mlit);
  }
  *mfixed_ptr = mfixed;
  LOG("compacting to %u variables %.2f%% from %u",
        vars, kissat_percent(vars, solver->vars), solver->vars);
  assert(vars == active || vars == active + 1);
  return vars;
}

static void compact_literal(kissat *solver, unsigned dst_lit,
      unsigned src_lit) {
  assert(dst_lit < src_lit);
  assert(dst_lit != NOT(src_lit));
  const unsigned dst_idx = IDX(dst_lit);
  const unsigned src_idx = IDX(src_lit);
  assert(dst_idx != src_idx);
  LOG("mapping old internal literal %u to %u", src_lit, dst_lit);
  solver->assigned[dst_idx] = solver->assigned[src_idx];
  solver->flags[dst_idx] = solver->flags[src_idx];
  solver->protect[dst_idx] = solver->protect[src_idx];

  solver->phases.best[dst_idx] = solver->phases.best[src_idx];
  solver->phases.saved[dst_idx] = solver->phases.saved[src_idx];
  solver->phases.target[dst_idx] = solver->phases.target[src_idx];

  const unsigned not_src_lit = NOT(src_lit);
  const unsigned not_dst_lit = NOT(dst_lit);
  solver->values[dst_lit] = solver->values[src_lit];
  solver->values[not_dst_lit] = solver->values[not_src_lit];

  cache *cache = &solver->cache;
  if (cache->valid) {
    const line *const end = END_STACK(cache->lines);
    line *begin = BEGIN_STACK(cache->lines);
    const unsigned old_vars = cache->vars;
    assert(old_vars == solver->vars);
    for (line *l = begin; l != end; l++) {
      assert(l->vars == old_vars);
      bits *bits = l->bits;
      const bool value = kissat_get_bit(bits, old_vars, src_idx);
      kissat_set_bit_explicitly(bits, old_vars, dst_idx, value);
    }
  }
}

static unsigned map_idx(kissat *solver, unsigned iidx) {
  COVER(ELIMINATED(iidx));
  int elit = PEEK_STACK(solver->export, iidx);
  if (!elit) {
    return INVALID_IDX;
  }
  assert(elit);
  const unsigned eidx = ABS(elit);
  assert(eidx);
  import *import = &PEEK_STACK(solver->import, eidx);
  assert(import->imported);
  if (import->eliminated) {
    return INVALID_IDX;
  }
  const unsigned mlit = import->lit;
  const unsigned midx = IDX(mlit);
  assert(midx <= iidx);
  return midx;
}

static void compact_queue(kissat *solver) {
  LOG("compacting queue");
  links *links = solver->links, *l;
  unsigned *p = &solver->queue.first, prev = DISCONNECT;
  solver->queue.stamp = 0;
  for (unsigned idx; !DISCONNECTED(idx = *p); p = &l->next) {
    const unsigned midx = map_idx(solver, idx);
    assert(midx != INVALID_IDX);
    l = links + idx;
    l->prev = prev;
    l->stamp = ++solver->queue.stamp;
    if (idx == solver->queue.search.idx) {
      solver->queue.search.idx = midx;
      solver->queue.search.stamp = l->stamp;
    }
    *p = prev = midx;
  }
  solver->queue.last = prev;
  *p = DISCONNECT;
  for (all_variables(idx)) {
    const unsigned midx = map_idx(solver, idx);
    if (midx == INVALID_IDX) {
      continue;
    }
    links[midx] = links[idx];
  }
}

static void compact_scores(kissat *solver, unsigned vars) {
  LOG("compacting scores");

  heap new_scores;
  memset(&new_scores, 0, sizeof new_scores);
  kissat_resize_heap(solver, &new_scores, vars);

  heap *old_scores = &solver->scores;

  if (old_scores->tainted) {
    LOG("copying scores of tainted old scores heap");
    for (all_variables(idx)) {
      const unsigned midx = map_idx(solver, idx);
      if (midx == INVALID_IDX) {
        continue;
      }
      const double score = kissat_get_heap_score(old_scores, idx);
      kissat_update_heap(solver, &new_scores, midx, score);
    }
  } else {
    LOG("no need to copy scores of old untainted scores heap");
  }

  LOG("now pushing mapped literals onto new heap");
  for (all_stack(unsigned, idx, old_scores->stack)) {
    const unsigned midx = map_idx(solver, idx);
    if (midx == INVALID_IDX) {
      continue;
    }
    kissat_push_heap(solver, &new_scores, midx);
  }

  kissat_release_heap(solver, old_scores);
  solver->scores = new_scores;
}

static void compact_trail(kissat *solver) {
  LOG("compacting trail");
  const size_t size = SIZE_ARRAY(solver->trail);
  for (size_t i = 0; i < size; i++) {
    const unsigned ilit = PEEK_ARRAY(solver->trail, i);
    const unsigned mlit = kissat_map_literal(solver, ilit, true);
    assert(mlit != INVALID_LIT);
    POKE_ARRAY(solver->trail, i, mlit);
    const unsigned idx = IDX(ilit);
    assigned *a = solver->assigned + idx;
    if (!a->binary) {
      continue;
    }
    const unsigned other = a->reason;
    assert(VALID_INTERNAL_LITERAL(other));
    const unsigned mother = kissat_map_literal(solver, other, true);
    assert(mother != INVALID_LIT);
    a->reason = mother;
  }
}

static void compact_frames(kissat *solver) {
  LOG("compacting frames");
  const size_t size = SIZE_STACK(solver->frames);
  for (size_t level = 1; level < size; level++) {
    frame *frame = &FRAME(level);
    const unsigned ilit = frame->decision;
    const unsigned mlit = kissat_map_literal(solver, ilit, true);
    assert(mlit != INVALID_LIT);
    frame->decision = mlit;
  }
}

static void compact_export(kissat *solver, unsigned vars) {
  LOG("compacting export");
  const size_t size = SIZE_STACK(solver->export);
  assert(size <= UINT_MAX);
  assert(size == solver->vars);
  for (unsigned iidx = 0; iidx < size; iidx++) {
    const unsigned elit = PEEK_STACK(solver->export, iidx);
    if (!elit) {
      continue;
    }
    const unsigned midx = map_idx(solver, iidx);
    if (midx == INVALID_IDX) {
      continue;
    }
    POKE_STACK(solver->export, midx, elit);
  }
  RESIZE_STACK(solver->export, vars);
  SHRINK_STACK(solver->export);
#ifndef NDEBUG
  assert(SIZE_STACK(solver->export) == vars);
  for (unsigned iidx = 0; iidx < vars; iidx++) {
    const int elit = PEEK_STACK(solver->export, iidx);
    assert(VALID_EXTERNAL_LITERAL(elit));
    const unsigned eidx = ABS(elit);
    const import *const import = &PEEK_STACK(solver->import, eidx);
    assert(import->imported);
    if (import->eliminated) {
      continue;
    }
    unsigned mlit = import->lit;
    if (elit < 0) {
      mlit = NOT(mlit);
    }
    const unsigned ilit = LIT(iidx);
    assert(mlit == ilit);
  }
#endif
}

static void compact_units(kissat *solver, unsigned mfixed) {
  LOG("compacting units (first fixed %u)", mfixed);
  assert(kissat_fixed(solver, mfixed) > 0);
  for (all_stack(int, elit, solver->units)) {
    const unsigned eidx = ABS(elit);
    const unsigned mlit = elit < 0 ? NOT(mfixed) : mfixed;
    const import *const import = &PEEK_STACK(solver->import, eidx);
    assert(import->imported);
    assert(!import->eliminated);
    const unsigned ilit = import->lit;
    if (mlit != ilit) {
      reimport_literal(solver, eidx, mlit);
    }
  }
}

static void compact_best_and_target_values(kissat *solver, unsigned vars) {
  const value *const best = solver->phases.best;
  const value *const target = solver->phases.target;
  const flags *const flags = solver->flags;

  unsigned best_assigned = 0;
  unsigned target_assigned = 0;

  for (unsigned idx = 0; idx < vars; idx++) {
    if (!flags[idx].active) {
      continue;
    }
    if (target[idx]) {
      target_assigned++;
    }
    if (best[idx]) {
      best_assigned++;
    }
  }

  if (solver->target_assigned != target_assigned) {
    LOG("compacting target assigned from %u to %u",
          solver->target_assigned, target_assigned);
    solver->target_assigned = target_assigned;
  }

  if (solver->best_assigned != best_assigned) {
    LOG("compacting best assigned from %u to %u",
          solver->best_assigned, best_assigned);
    solver->best_assigned = best_assigned;
  }
}

static bits *compact_bits(kissat *solver, bits *old_bits, unsigned new_vars) {
  bits *new_bits = kissat_new_bits(solver, new_vars);
  const unsigned old_vars = solver->vars;
  for (unsigned idx = 0; idx < old_vars; idx++) {
    const unsigned midx = map_idx(solver, idx);
    if (midx == INVALID_IDX) {
      continue;
    }
    const bool bit = kissat_get_bit(old_bits, old_vars, idx);
    kissat_set_bit_explicitly(new_bits, new_vars, midx, bit);
  }
  kissat_delete_bits(solver, old_bits, old_vars);
  return new_bits;
}

static void compact_cache(kissat *solver, unsigned new_vars) {
  cache *cache = &solver->cache;
  if (!cache->valid) {
    LOG("not compacting invalid cache");
    return;
  }
  LOG("compacting cache of size %zu over %u variables",
        SIZE_STACK(cache->lines), cache->vars);
  const line *const end = END_STACK(cache->lines);
  line *begin = BEGIN_STACK(cache->lines);
  for (line *l = begin; l != end; l++) {
    assert(l->vars == solver->vars);
    LOGLINE(l, "resizing to %u variables", new_vars);
    l->bits = compact_bits(solver, l->bits, new_vars);
    l->vars = new_vars;
  }
  cache->vars = new_vars;
}

void kissat_finalize_compacting(kissat *solver, unsigned vars,
      unsigned mfixed) {
  LOG("finalizing compacting");
  assert(vars <= solver->vars);
#ifdef LOGGING
  assert(solver->compacting);
#endif
  if (vars == solver->vars) {
#ifdef LOGGING
    solver->compacting = false;
    LOG("number of variables does not change");
#endif
    return;
  }

  unsigned reduced = solver->vars - vars;
  LOG("compacted number of variables from %u to %u", solver->vars, vars);

  bool first = true;
  for (all_variables(iidx)) {
    flags *flags = FLAGS(iidx);
    if (flags->fixed && first) {
      first = false;
    } else if (!flags->active) {
      POKE_STACK(solver->export, iidx, 0);
    }
  }

  compact_trail(solver);

  for (all_variables(iidx)) {
    const unsigned ilit = LIT(iidx);
    const unsigned mlit = kissat_map_literal(solver, ilit, true);
    if (mlit != INVALID_LIT && ilit != mlit) {
      compact_literal(solver, mlit, ilit);
    }
  }

  if (mfixed != INVALID_LIT) {
    compact_units(solver, mfixed);
  }

  memset(solver->assigned + vars, 0, reduced * sizeof(assigned));
  memset(solver->flags + vars, 0, reduced * sizeof(flags));
  memset(solver->protect + vars, 0, reduced * sizeof(unsigned));
  memset(solver->values + 2 * vars, 0, 2 * reduced * sizeof(value));
  memset(solver->watches + 2 * vars, 0, 2 * reduced * sizeof(watches));

  compact_queue(solver);
  compact_cache(solver, vars);
  compact_scores(solver, vars);
  compact_frames(solver);
  compact_export(solver, vars);
  compact_best_and_target_values(solver, vars);

  solver->vars = vars;
#ifdef LOGGING
  solver->compacting = false;
#endif
  kissat_decrease_size(solver);
}

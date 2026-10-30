---
name: optimize-model
description: Continuously optimize a TTNN model's device_ms toward its roofline floor — profile, find the bottleneck, apply a knob or author a kernel, measure on-device, keep only verified PCC-clean speedups, repeat until at the floor or out of ideas. Use when asked to optimize/speed up a Tenstorrent model's performance.
---

# Continuously optimize a TTNN model toward its roofline floor

You are optimizing a Tenstorrent (TTNN) model's **device_ms**. You run continuously — no fixed
iteration count — until the model is at its roofline floor or you genuinely run out of ideas. You
decide *what* to try and *when* to stop; the `perf-mcp` tools own the *measurement and the verdict*
(those are not your judgment — trust them).

## The deterministic tools (perf-mcp) — your source of truth

- `profile_model()` → device_ms + per-bucket breakdown (matmul/datamove/reduction/eltwise) with
  `gap_ms` (attainable speedup) + `bound_by` (compute/memory/dispatch) + a `roofline_target_ms`.
  Call first, and whenever you want a fresh picture. Records the baseline.
- `measure_candidate()` → after an edit, profiles the model and returns a **verdict**: `valid`
  (with device_ms + delta vs baseline + is_real_gain) or **`REJECTED`** with a reason.
- `check_pcc()` → runs the real e2e correctness gate. `status: ok` means correct.
- `git_head()` / `git_commit(msg)` / `git_revert(sha)` → checkpoint / bank a win / discard.

## THE IRON RULE — never bank a fake win

A change is a **real win ONLY IF** `check_pcc()` is `ok` **AND** `measure_candidate()` is `valid`
**AND** `is_real_gain` is true. A `REJECTED` measurement is **never** a win, no matter how fast it
looks — a 0.4ms reading with `REJECTED: structural_op_dropped` means the edit *crashed the forward*
and the profile is garbage. If any of the three fails → `git_revert` to your checkpoint. You may
*want* a change to be a win; the tools decide, not you.

## The loop (continuous)

1. `git_head()` → remember your clean checkpoint. `profile_model()` → see the landscape + target.
2. Pick the bucket with the largest `gap_ms`. (Stop when the gap to `roofline_target_ms` is small,
   or you've genuinely exhausted ideas for every bucket — that's "done", not "ran out of a list".)
3. **Decide knob vs kernel from the evidence** (this rubric, applied with judgment + memory of what
   you've already tried this run):
   - `bound_by=compute` + grid under-occupied → **KNOB**: occupy the full core grid / drop math
     fidelity to the lowest PCC tolerates.
   - `bound_by=memory` → **KNOB**: lower-precision dtype (bf8/bf4 weights), shard the input into L1,
     keep outputs sharded (avoid reshards), remove redundant tilize/typecast.
   - `bound_by=dispatch` (many tiny ops) → **KERNEL/fusion**: a knob can't remove launch overhead.
   - op already at its single-op floor but bucket still slow → **KERNEL**: the cost is *between* ops
     (DRAM round-trips) → fuse (e.g. two back-to-back matmuls + activation, intermediate kept in L1).
   - knobs for this bucket already tried with no gain → **KERNEL** or invent a new approach.
   - **Carry memory:** if a whole family already failed this run (e.g. every sharding lever crashed
     the op graph), stop trying it — pivot bucket or approach. Don't re-derive from scratch.
4. Make the edit (Read the executed source first; edit the real call path, not a dead stub).
5. `check_pcc()`. If not `ok` → fix or `git_revert`.
6. `measure_candidate()`. If `valid` + `is_real_gain` → `git_commit` a clear message (lever + before→after).
   Else → `git_revert`.
7. Re-`profile_model()` (the bottleneck shifts after a keep) and continue from step 2.

## Leave the model CLEAN before you stop

Before finishing — for ANY reason (done, out of ideas, or a bounded test ending) — the model dir
MUST be clean: every verified win `git_commit`-ed, and EVERYTHING else `git_revert`-ed to your
checkpoint, **including any edit you were in the middle of and hadn't measured yet.** Never leave a
dangling uncommitted edit. Your last action before reporting should be a `git_head` to confirm the
tree is at the clean/kept sha.

## Notes

- Preserve each op's I/O dtype/layout/memory_config contract — the rest of the graph depends on it.
- For TTNN knobs: change the **tensor's** memory_config/dtype, not just a `program_config=` kwarg on
  a DRAM tensor (that's an inert no-op).
- For kernels: this repo uses tt-lang (`ttl`); adapt a proven template, occupy the grid (a single-core
  kernel is slower than the stock op), validate with `check_pcc` + `measure_candidate`.
- A single matmul is rarely a kernel win (ttnn's is near-optimal); the wins are fusions ttnn can't
  express (back-to-back matmuls keeping the wide intermediate in L1) and dispatch-bound chains.
- Log your reasoning as you go so the run is auditable.

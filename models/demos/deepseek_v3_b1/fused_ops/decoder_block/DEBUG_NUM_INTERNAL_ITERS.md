# DeepSeek V3 B1 — `num_internal_iterations` MLA PCC shift (open)

## Symptom

Running the fused DeepSeek V3 B1 decoder block on Blackhole (4x2 galaxy) with
`num_internal_iterations >= 2` produces a deterministic, bit-repeatable shift
in MLA PCC relative to a `num_internal_iterations = 1` baseline.

Measured at `position_id = 511`:

| Configuration                                         | DecoderBlock MLA PCC |
|-------------------------------------------------------|----------------------|
| `num_internal_iterations = 1`                         | **0.9989370** (baseline, matches Pure MLA standalone) |
| `num_internal_iterations >= 2` (internal while loop)  | **0.9990956** (shifted) |
| Python-side `num_iters = 2`, `num_internal = 1` (external relaunch) | 0.9989370 (matches baseline) |

The shift is not a test failure — both PCC values are well inside the acceptance
band — but it is a real numerical divergence that only appears when the
decoder-block's inner-iteration loop runs more than once per kernel launch.

Reproduction command:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 \
pytest "models/demos/deepseek_v3_b1/tests/unit_tests/test_decoder_block.py::test_decoder\
[blackhole-True-validate_standalone_moe-validate_standalone_mla-full_routing-\
rigged_groups1-3-NOC_MODE.DM_DYNAMIC_NOC-device_params0-511-32768-1-4-2-1-False-1e-06-1-0]"
```

This branch sets `num_internal_iterations = [3]` in the test parametrize to
drive the loop three times per launch so that iter 0 / iter 1 / iter 2 values
can be compared via DPRINT.

## Evidence: period-2 cycling on edge X-position MLA cores

The `decoder_block_kernel.cpp` on this branch contains DPRINT probes at every
pipeline stage of the internal loop. With `num_internal = 3` and
`*pos_ptr += 1` disabled (so every iter sees identical `cur_pos`), the
observed pattern is:

**Identical on ALL cores across ALL iters (bit-for-bit):**

- `mla_q_in`     — Q from `create_q_heads`
- `mla_k_in`     — full 6144-word XOR of K
- `cur_pos`, `local_cur_pos`
- `PRE_mla_out_xor256` — `cb_out_final` sampled just before `FlashMLA` begins

**Period-2 cycling on edge cores (x=0 and x=3 of the 4-wide MLA x-grid) — middle cores (x=1, x=2) stable:**

- `interm_out` — raw SDPA output from `compute_sdpa_chunk`, pre-tree-reduction
- `interm_ms`  — max / log-sum-exp statistics from SDPA
- `mla_out`    — final FlashMLA output
- `sdpa_out`   — output of the SDPA reduce worker
- `out_in`, `out_ms` — NOC-received reduce outputs consumed downstream

Pattern on edge cores: `iter 0 == iter 2 ≠ iter 1`. Verified to be strict
period-2 by running with `num_internal_iterations = 4`. Middle cores match
bit-for-bit across all iterations.

Example (x=0, y=1, S1 row):

```
iter 0 interm_out: 400d3f95 4011bfbe bf8d3f6a 3faf3f83
iter 1 interm_out: 3ff23f12 3f91bf9c bf5d3f02 3f023fcb
iter 2 interm_out: 400d3f95 4011bfbe bf8d3f6a 3faf3f83
```

Example (x=1, y=1 — middle, stable):

```
iter 0 interm_out: c006c019 3ffa3e89 3edd4032 bf6cbfb4
iter 1 interm_out: c006c019 3ffa3e89 3edd4032 bf6cbfb4
iter 2 interm_out: c006c019 3ffa3e89 3edd4032 bf6cbfb4
```

The edge/middle distinction holds across every `y` position that hosts
an MLA core (y=1 for S1, y=3 for S2, y=7 for S3).

## Bug zone

The zone boundary is sharp:

- **Entering clean** (verified bit-identical across iterations on every core):
  Q (`mla_q_in`), K (`mla_k_in`), `cur_pos`, `local_cur_pos`,
  `PRE_mla_out_xor256` (CB state immediately before FlashMLA).
- **Exiting dirty on edge cores only**: `interm_out` and `interm_ms` — both
  produced inside a single call to `compute_sdpa_chunk`.

Because `compute_sdpa_chunk` receives verified-identical inputs yet produces
different outputs on x=0 and x=3, the non-determinism must be generated
inside that function. Middle cores run the same function with the same
inputs and produce bit-identical output across iterations, so whatever
state cycles has to be position-sensitive.

## Rejected hypotheses

### 1. `pos_ptr` increment — rejected

Disabling `*pos_ptr += 1` in the internal loop did not remove the shift.
Iter 2 still produces a different value than iter 1 even though it now
sees identical `cur_pos`, identical weights, identical KV cache state.

### 2. `DST_SYNC_HALF` ping-pong — rejected

Hypothesis: the dest section flip in `_llk_math_dest_section_done_<SyncHalf, ...>`
leaves the DST at a different half at the end of iter 0, so iter 1 runs on
the flipped half, and iter 2 flips back. Some SFPU / rounding path on edge
cores could be sensitive to the half index.

Test: set `dst_full_sync_en = True` in
`models/demos/deepseek_v3_b1/fused_ops/decoder_block/op.py`. Verified via
Python import that the flag propagates, and the path through
`tt_metal/jit_build/genfiles.cpp:387` emits `#define DST_SYNC_MODE DstSync::SyncFull`
when the flag is `True`. In SyncFull mode,
`_llk_math_dest_section_done_` skips `dest_section_flip()` entirely.

Result: **cycling values are bit-identical between SyncHalf and SyncFull**
(e.g., `x=0,y=1 iter=1 interm_out = 3ff23f12 3f91bf9c bf5d3f02 3f023fcb` in
both modes). DecoderBlock MLA PCC unchanged at 0.9990956. DST section flip
is not the cause.

### 3. Mid-loop user-level `deepseek_compute_kernel_init()` — rejected (as a fix)

Calling `deepseek_compute_kernel_init()` between internal iterations
actually made PCC **worse** (0.998091). User-level init does not reset the
relevant hardware state, and appears to perturb something productive in
the steady-state pipeline.

### 4. K mcast sender/receiver role — rejected

Within a single S-block row, K mcast sender and receivers are both observed
to cycle or both observed to be stable, depending only on edge/middle
x-position. For example, at y=3: (0,3) is the K mcast sender and cycles;
(3,3) is a K mcast receiver and also cycles; (1,3) and (2,3) are both K
mcast receivers and both stable.

## Remaining candidates (not yet tested)

Whatever cycles has to be:
1. Internal to `compute_sdpa_chunk`.
2. Period-2 across iterations (some flip-flop state that does not reset on
   internal-loop boundaries but does reset on full kernel launch).
3. Position-sensitive — present on x=0 and x=3, absent on x=1 and x=2.

Likely candidates worth investigation:

- **t6 semaphore accumulated count** — `FPU_SFPU` / `SFPU_FPU` semaphores
  used by `compute_sdpa_chunk`'s pipeline between FPU matmul and SFPU
  reduce. An off-by-one in per-iter counts could leave an odd/even
  semaphore state.
- **SFPU LREG persistence** — an SFPU lreg populated in iter 0 but only
  *conditionally* overwritten on edge cores could give period-2 behavior.
- **Shared replay buffer state** —
  `_init_sdpa_reduce_max_row_8x32_replay_buffers_()` is initialized once
  per kernel; if its internal pointer or bank-select advances on each
  compute call and the count is odd on edge cores but even on middle
  cores, period-2 divergence would result.
- **Edge-specific NoC / tensix config register** — anything that gets
  programmed differently based on whether a core is first or last in its
  row (e.g., for mcast endpoint logic) could accumulate state across
  iterations.

Next steps that would be informative:

1. DPRINT the SFPU LREG contents from the TRISC math kernel at the
   start / end of `compute_sdpa_chunk` across iterations on an edge core
   vs a middle core.
2. Force a DST reset or replay-buffer re-init between internal iterations
   without touching user-level init, and check whether cycling stops.
3. Bisect `compute_sdpa_chunk` internals — probe at each sub-step (mm1,
   softmax reduce, mm2) to find the first sub-step that differs on edge
   cores.

## Impact

This does **not** currently cause a test failure. Iter 1's PCC (0.999096)
is actually *higher* than baseline (0.998937). Because `num_internal = 3`
ends on iter 2 which matches iter 0, the reported DecoderBlock MLA PCC
still tracks the iter-0 value closely.

The concern is correctness reasoning: internal-loop iterations are **not**
bit-identical to external relaunches on edge cores. Any optimization that
relies on "run the loop N times internally is equivalent to relaunching
the kernel N times" is wrong on the order of ~6 ULPs in bf16 per element
on those two cores.

## Files on this branch

- `models/demos/deepseek_v3_b1/fused_ops/decoder_block/kernels/decoder_block_kernel.cpp`
  — DPRINT probes at every pipeline stage of the internal loop (Q, K, cur_pos,
  PRE_mla_out, mla_out, interm_out, interm_ms, out_in, out_ms).
- `models/demos/deepseek_v3_b1/fused_ops/attention_block/op.py`
  — exposes `mla_out_final_cb`, `mla_interm_out_cb`, `mla_interm_ms_cb`
  as compile-time args so the DPRINT probes can access them.
- `models/demos/deepseek_v3_b1/tests/unit_tests/test_decoder_block.py`
  — `num_internal_iterations = [3]` to drive three iterations per launch.

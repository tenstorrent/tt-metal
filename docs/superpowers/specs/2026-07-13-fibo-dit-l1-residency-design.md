# FIBO DIT Transformer — L1 activation residency (denoise Phase 2)

**Date:** 2026-07-13
**Branch:** `fibo-pipeline`
**Status:** design (auto-approved per bringup workflow; proceed to plan)
**Measured by:** `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py`

## Goal

Keep the FIBO DIT transformer's (`BriaFiboTransformer`) intermediate activation tensors
resident in **L1** instead of DRAM during the denoise forward, to cut the time spent by the
bandwidth-bound ops (elementwise, norm, TM) and to avoid DRAM write→read round-trips between
ops. Interleaved-L1 first (bit-exact, low-risk); L1-sharded is a deferred Phase 2.

This is Phase 2 of the FIBO denoise optimization (Phase 1 = matmul block-size tuning, done,
−2.2% forward, at its ceiling). It attacks a different lever: **where activations live**.

## Current state (verified 2026-07-13)

The entire DIT path is **DRAM-interleaved, TILE layout** — there is no L1 usage and no on-chip
sharding anywhere in `models/tt_dit`. ("Sharding" in this code = multi-device mesh fracture, not
L1 core sharding.) The chain stays DRAM because:

- Weights default to DRAM (`layers/module.py:326`).
- `ttnn.experimental.minimal_matmul` **inherits in0's memory config** when `memory_config` is
  omitted (`ttnn/cpp/.../minimal_matmul/device/minimal_matmul_device_operation.cpp:221`). Inputs
  originate in DRAM → all matmul outputs are DRAM. **The op DOES accept `memory_config=`; the
  tt_dit Linear helpers simply never pass it.**
- Elementwise (`BinaryNg`), unary, `LayerNorm`/`RMSNorm`, `concat`, `slice`, create/concat-heads
  all inherit their input's memory config → DRAM.
- CCL buffers are explicitly DRAM: all-gather ping-pong (`parallel/manager.py:181,317`),
  reduce-scatter output (`manager.py:511`), fused MM+RS (`layers/linear.py:434-435`).
- SDPA (`ring_joint_scaled_dot_product_attention` / `joint_...`) passes no output memory_config
  → DRAM.

### Denoise device-op profile (one forward, 172.8 ms total, `denoise_report/ops.csv`)

| Category | % device time | L1 sensitivity |
|---|---|---|
| Matmuls (all `MinimalMatmul` shapes) | ~45% | compute-bound (~15.7% DRAM roofline); L1 mainly removes the inter-op DRAM round-trip |
| SDPA (`RingJointSDPA`, 46 ops) | 13.9% | static CBs already ~1.57 MB, near the L1 ceiling — **risk** |
| Eltwise `BinaryNg` (981 ops, ~20 µs ea) | 11.6% | **bandwidth-bound — prime L1 target** |
| Unary + LayerNorm | 9.5% | **bandwidth-bound — prime L1 target** |
| TM (Concat / CreateHeads / ConcatHeads / Slice) | ~13% | **bandwidth-bound — good L1 target** |
| CCL (AllGather / ReduceScatter / Pre/Post) | ~10% | persistent buffers — separate handling |

The primary win is the ~30%+ in memory-bound eltwise/norm/TM ops that ride the residual stream:
once the stream is L1, they follow for free (inherit). Matmuls get a smaller, second-order win
(no DRAM round-trip between ops).

## Approach

A single opt-in **`activation_memory_config`** (a `ttnn.MemoryConfig | None`) threaded top-down
from `BriaFiboTransformer` through the shared blocks/helpers. Default `None` everywhere ⇒
**exactly today's DRAM behavior** ⇒ other DiT models (Flux, Wan, LTX) stay byte-identical. FIBO
sets it to `ttnn.L1_MEMORY_CONFIG` (interleaved) for Phase 1.

### Chain-breakers to address (the whole reason it isn't free)

1. **Matmuls** — thread `memory_config` into the `minimal_matmul` / `all_gather_minimal_matmul_async`
   / `minimal_matmul_split` / `minimal_matmul_strided_reduce_scatter_async` calls in
   `layers/linear.py`. Output produced directly in L1, **no extra copy op**.
2. **CCL persistent buffers** — opt-in L1 for all-gather ping-pong buffers + reduce-scatter output
   in `parallel/manager.py` (and the RS mem-config literals in `linear.py:434-435`).
3. **SDPA output** — pass L1 `memory_config` to the (ring/joint) SDPA in `blocks/attention.py`;
   **verify the SDPA static CBs still fit** (q_chunk=128/k_chunk=512 today; larger chunks already
   overflow per prior sweep — do not change chunk sizes here).
4. **DistributedLayerNorm** — output + internal stats all-gather → L1 in `layers/normalization.py`.

Everything else (add/mul/gate/residual, gelu/silu, concat, slice, create/concat-heads,
`split_..._and_split_heads` which already forwards `qkv.memory_config()`) **inherits for free**
once its inputs are L1.

### Where L1 residency begins

Produce the residual stream in L1 at the top of `BriaFiboTransformer.forward` (embedder outputs
→ L1, or a single `ttnn.to_memory_config(..., L1)` at entry). The one unavoidable DRAM→L1 copy is
amortized over 46 blocks.

## Files touched (all opt-in, `None` default)

- `models/tt_dit/layers/linear.py` — `memory_config` param on `Linear/ColParallelLinear/RowParallelLinear.forward` (+ `forward_fused_addcmul`), passed into the 4 minimal-matmul variants.
- `models/tt_dit/parallel/manager.py` — opt-in L1 for AG persistent buffers + RS output.
- `models/tt_dit/blocks/attention.py` — SDPA output + pre-out-proj all-gathers.
- `models/tt_dit/layers/normalization.py` — `DistributedLayerNorm` output + stats AG.
- `models/tt_dit/layers/feedforward.py` — thread through `ff1`/`ff2`.
- `models/tt_dit/blocks/transformer_block.py` — thread through dual block.
- `models/tt_dit/models/transformers/transformer_flux1.py` — thread through single block.
- `models/tt_dit/models/transformers/transformer_bria_fibo.py` — the switch (config flag) + set the residual stream L1 at entry.

`test_performance_bria_fibo.py` itself is unchanged in structure; it is the measurement harness.

## Correctness

L1-**interleaved** does not change the math (same compute, different buffer location) ⇒ output
must be **bit-exact** vs the DRAM baseline. Target PCC = baseline **99.53%** on
`test_fibo_pipeline_perf_breakdown` (and `test_fibo_transformer_mesh`, full 8+38 depth). Any PCC
drift is a bug (a real mis-wire), NOT a diffusion-trajectory shift — distinct from the reverted
SDPA-chunk change, which altered accumulation order.

## Measurement

- **Per-op iteration:** `test_fibo_denoise_device_profile` under Tracy (per the file's documented
  command + `denoise_report/` render) — shows which ops moved to L1 and their new device time.
- **End-to-end gate:** `test_fibo_pipeline_perf_breakdown` (untraced + traced) — denoise it/s and
  the saved-image PCC/validity. This is the keep/revert authority.
- Prior baselines (this branch): denoise ~2.34 it/s traced; whole pipeline ~13.84 s; denoise
  forward 172.8 ms (profile).

## Keep / revert rule (per site)

Enable L1 at a site only if **both** hold:

1. **Fits** — no L1 OOM at that site (this is the literal "if possible"). The single-block
   `concat` (~30 MB/dev) and `proj_mlp` (~24 MB/dev) already carry a `# OOM` note in DRAM; expect
   some sites to refuse. On OOM, that tensor stays DRAM.
2. **Helps (or is neutral)** — denoise it/s does not regress; PCC holds at 99.53%.

Sites are enabled incrementally so a regression/OOM is attributable to one change.

## Risks

- **L1 pressure / OOM** — many large activations (concat 30 MB, ff1 24 MB, qkv 18 MB per device)
  plus op circular buffers plus SDPA's ~1.57 MB static CBs must coexist. Interleaved spreads a
  30 MB tensor to ~250 KB/core across ~120 cores, which fits per-core, but *simultaneous* live
  tensors + CBs are the real limit. Mitigate by per-site enablement + the fit gate.
- **SDPA CB overflow** — do not touch chunk sizes; only the output buffer moves to L1, and only if
  it still fits.
- **Shared-code regressions** — mitigated by the `None` default (other models untouched) and a
  Flux/Wan spot-check if any shared default is at risk of changing.

## Phasing

- **Phase 1 (this spec):** interleaved-L1 residency across matmul outputs, CCL buffers, SDPA
  output, norm output, and the inherit bucket. Bit-exact. Measure and lock in the survivors.
- **Phase 2 (deferred):** escalate the hottest surviving tensors to L1 block/height-sharded where
  the profile justifies the per-op reshard cost. Introduces the first on-chip sharded pattern in
  `models/tt_dit`; larger surface, revisit after Phase 1 numbers land.

## Non-goals

- Weights stay in DRAM (46 blocks of weights will not fit in L1).
- Encoder (SmolLM3) and VAE decoder are out of scope here — transformer first, per the request.
  Extending "everything in L1" to those stages is a later sub-project.
- No matmul block-size re-tuning (Phase 1 is done); no SDPA chunk-size changes (reverted, dead end).

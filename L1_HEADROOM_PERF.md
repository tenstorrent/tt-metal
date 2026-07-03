# Spending In-Place-Halo L1 Headroom for Blackhole Conv Perf (single-source log)

Branch: `wransom/in_place_halo_redo` (continues the in-place halo work). Arch focus:
**Blackhole p100a** (company priority). Owner: wransom + Claude.

Goal: in-place halo freed L1 at large-feature-map downsampling conv/pool layers; **spend that
headroom on higher-L1, higher-perf conv config paths** where it delivers a measured BH win.
Per-trace methodology (user): (1) identify candidate, (2) confirm L1 headroom, (3) find a
higher-L1 perf setting, (4) confirm headroom suffices to activate it, (5) confirm measured
benefit. Log what worked AND what didn't.

## Key findings that shape the effort

1. **Pool is a dead end for perf on BH.** Curriculum T-5: small-window pool is
   unpack/tilize-bound (`TRISC0≈KERNEL`); reader-batch / CB-depth / math-fidelity / DST all
   move it ~0%. So spending L1 on pool won't buy perf. **Focus on conv** (also matches "conv
   is more expensive than pool").
2. **The primary enabler is likely DRAM-slicing avoidance, via the L1 *estimate*.** conv2d is
   `halo → matmul` (separate device ops). The conv2d op's L1 *estimate* drives whether it
   DRAM-slices (big perf hit) and which config it picks. In-place halo lowers the halo phase's
   L1; IF the conv L1 estimate credits that saving, convs that currently slice (or pick a
   lower-perf config) because the estimate over-counts could now run L1-full / higher-config.
   This is the deferred "DRAM-slicing L1 estimate" item from the in-place work.

   **LINCHPIN RESOLVED (conv2d L1-estimate → slicing/config map):**
   - **DRAM-slicing is AUTOMATIC + L1-fit-driven.** Path: input in L1 → L1 path; input in
     DRAM-interleaved → DRAM path, which auto-searches the smallest `num_slices` that fits free
     L1, and promotes a 1-slice fit to `L1_FULL` (whole conv in L1). (`op_slicing.cpp:156-269,
     283-324`.)
   - **The halo input+output coexistence is counted in exactly ONE place:**
     `conv2d.cpp:618` `Conv2dSliceAttr::get_L1_usage` = `max(halo_input_size + halo_output_size,
     total_size)`. That `halo_input_size` is precisely what in-place eliminates. Crediting it
     here (when `should_halo_be_in_place` is true for the slice) lowers the fit estimate → fewer
     slices / `L1_FULL` promotion → **avoids DRAM slicing = the big automatic win.**
     → **WORKSTREAM A** (code change). CORRECTNESS-CRITICAL: must credit *exactly* when in-place
     activates at runtime for that slice geometry, else it under-estimates → OOM/FATAL.
   - **The L1 path has NO L1-budget-gated config.** `act_block_h`, `enable_act/weights_double_buffer`,
     `full_inner_dim` are model-set pass-through `Conv2dConfig` flags — nothing auto-enables them
     from freed L1. So on the L1 path, spending headroom = **per-trace manual config relaxation**
     at in-place-freed layers, validated for no-OOM + measured perf.
     → **WORKSTREAM B** (per-trace, via tt-blackhole-perf-optimizer).
   - Note: conv2d already calls the gate with `allow_in_place=true` (`conv2d.cpp:288`), so
     in-place is live for conv2d today; only the DRAM-slice *estimator* is in-place-blind.

## Two workstreams
- **A. DRAM-slicing credit** (`conv2d.cpp:618`) — automatic, model-agnostic; helps every
  auto-slicing conv where in-place activates. Gate: beneficiaries must exist (a conv that
  auto-DRAM-slices, e.g. SDXL VAE high-res). Impl must match runtime in-place activation exactly.
- **B. L1-path per-trace config relaxation** — enable double-buffer / larger act_block at
  in-place-freed layers. First concrete candidate: **UFLD-v2** height-sharded convs (double-buffer
  is OFF there: `enable_act/weights_double_buffer = True only if block-sharded`).
3. **Headroom is targeted, not blanket.** In-place lowers *whole-op* peak L1 only where the
   halo input↔output coexistence was the binding peak = large-feature-map **downsampling
   (stride≥2)** conv layers (measured −29..−34% last session). Stride-1 is peak-neutral.

## L1-for-perf levers (BH conv), from tt-blackhole-perf-knowledge

| Lever | Phase | Gain (BH) | Notes |
|---|---|---|---|
| **DRAM-slicing avoidance** (run L1-full where it now fits) | whole conv | large (avoids chunked streaming) | primary; gated by the L1 estimate crediting in-place saving |
| **T-2 `full_inner_dim` + act/weights double-buffer** | matmul | **−37..−60%** | block-sharded conv, pipeline-serialized (few tiles/many K-blocks); L1-gated factory auto-declines if tight |
| **`packer_l1_acc`** | matmul | +3..7% kernel / −3..5% e2e | multi-K-block **bias** convs only (`enable_bias && in0_num_blocks_w>1`); grep stale `=False`; auto-gated no-op else |
| **`act_block_h_override` (larger)** + `enable_act_double_buffer` / `enable_weights_double_buffer` | matmul | fewer matmul_block calls / deserialize reader | watch the "ceiling" pitfall — an existing override may already cap `act_block_h_ntiles < per_core_M`; relaxing IS the unlock |
| weights in L1 vs DRAM-sharded | matmul | mcast faster than DRAM-sharded | switch off `...DRAMShardedProgramConfig` if weights now fit L1 |
| subblock volume (T-3) | matmul | **anti-fit on BH** (−1.6..−4.9% on compute-bound) | conv subblock is auto-derived; DST cap arch-identical (NOT doubled on BH); needs ≥4× vol; usually a trap |

## Candidate BH models (conv/pool-heavy, priority order)
ResNet50 (BH) · UFLD-v2 (BH) · VGG-UNet (BH) · SDXL VAE/UNet (BH) · functional_unet · ViT
(conv stem). Isolated per-conv harness: `tests/ttnn/perf_tests/operations/conv/test_conv2d_device_perf.py`.
Focus: early large-feature-map downsampling convs (where in-place freed binding-peak L1).

## Measurement rigor (non-negotiable; from blackhole-perf.md)
Clock is north star, gated by: (1) PCC on FRESH JIT cache, (2) hang-free/stable across
program-cache hits, (3) faster on **DEVICE KERNEL DURATION** with median+std-dev over adaptive
3→5→10 trials. Clear JIT+program cache each side after kernel edits. Re-baseline after any
`tt-smi -r`. Prefer same-binary getenv-toggle A/B. Confirm picks changed via Tracy ATTRIBUTES
(`act_block_h_ntiles` vs `per_core_M/N`) BEFORE a kernel-time A/B (30s grep skips a 10min run).
Final single-claim sign-off → tt-perf-validator.

## Per-trace results log (what was tried, what worked/didn't)

_(to be filled per trace: model, conv shape, baseline µs, lever tried, headroom confirmed?,
config change, new µs, verdict win/loss/noise, why)_

| trace | lever | headroom? | before | after | verdict | notes |
|---|---|---|---|---|---|---|
| UFLD-v2 HS convs (BH) | B: enable act/weights double-buffer | N/A | — | — | ⛔ BLOCKED | see CRITICAL bug below; also in-place DECLINES at all UFLD HS convs (no headroom) — activates only at 1 block-sharded conv |

## ⛔ CRITICAL: in-place halo full-model robustness bug (found 2026-07-03)
Running the **full UFLD-v2 model** on BH deterministically `TT_FATAL`s in the in-place-halo
containment backstop (`halo_device_operation.cpp:154`): the L1 allocator placed the output
shard `[1472704,1522176)` (49472 B) so it does NOT contain the freed input shard
`[1530880,1548288)` (17408 B) — the input overhangs by 26112 B. **Root cause:** the aliasing
assumes `create_output_tensors` dealloc-then-`create_device_tensor` lands the (larger) output
*over* the freed input (top-down reuse). In a **fragmented full-model free-list** the allocator
picks a different, larger free block → no overlap → the backstop fires (safe: caught, no
corruption, but crashes the model). Confirmed repro (with & without profiler; input is
allocated at input size 17408, output needs 49472 → different block).

**Why isolated tests missed it:** the pool/conv unit tests run as standalone ops on a CLEAN
free-list (input at top → dealloc+realloc reuses it). Full models fragment L1 → the assumption
breaks. This is exactly the "subtle, only-under-specific-conditions" hazard flagged at project
start — surfaced on BH in a real model.

**Consequences:** (1) in-place halo (auto-activates for conv2d/pool) can crash real BH models →
correctness fix required before the perf effort can run any model. (2) Real-model headroom is
much smaller than the isolated −29..−34%: in UFLD-v2 in-place ACTIVATES at only ONE conv (a
block-sharded layer) and DECLINES at every height-sharded target → workstream B's premise is
largely void there.

**Fix direction (recommended): opportunistic graceful fallback** — if the allocator places the
output over the input → in-place; else → normal halo (never crash). Must handle the
freed-input-lifetime safety carefully. Alternatives: over-allocate input at output size in the
caller (deterministic reuse, costs L1 during input life); or gate in-place to guaranteed-overlap
cases only.
| SDXL VAE (BH) | A: credit in-place in conv2d.cpp:618 → reduce DRAM slices / L1_FULL | TBD | — | — | 🔜 candidate | uses DRAM-interleaved conv tensors + huge spatial → likely auto-slices; needs device confirm + the code change |

### Workstream sequencing
- Device is single; run ONE perf task at a time (no concurrent perf; no rebuild while a perf agent measures).
- After UFLD optimizer: implement Workstream-A credit at `conv2d.cpp:618` (correctness-matched to runtime in-place activation), confirm SDXL VAE actually auto-slices, validate the credit reduces slices → perf, no OOM.
- Then continue Workstream B across the other candidates (ResNet50 act-block, VGG-UNet double-buffer/packer_l1_acc, functional_unet) via the optimizer.

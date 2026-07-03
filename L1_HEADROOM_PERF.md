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
   This is the deferred "DRAM-slicing L1 estimate" item from the in-place work — now the KEY
   lever. **Linchpin to verify: does `calculate_L1_usage_for_conv_op` / the DRAM-slice decision
   account for the halo intermediate's coexistence peak, and where does the in-place saving
   propagate?** (under investigation)
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
| _pending_ | | | | | | |

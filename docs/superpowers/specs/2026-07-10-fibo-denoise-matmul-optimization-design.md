# FIBO denoise-step matmul optimization — design

Date: 2026-07-10
Branch: `fibo-pipeline`
Status: approved (brainstorming) → implementation

## Goal

Raise the compute efficiency of the FIBO DiT **denoise** forward (`BriaFiboTransformer`) on
the 2×2 Blackhole mesh (sp=2, tp=2) at production sizes (spatial seq 4096, 1024²), measured by
the per-op device profiler, **without regressing correctness** (PCC ≥ 0.99).

- **Implementation:** `models/tt_dit/models/transformers/transformer_bria_fibo.py` (and the shared
  blocks it reuses: `blocks/transformer_block.py`, `models/transformers/transformer_flux1.py`,
  `blocks/attention.py`, `layers/linear.py`, `utils/matmul.py`).
- **Metric harness:** `test_fibo_denoise_device_profile` in
  `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py` → renders `denoise_report/`
  via `tt-perf-report` (run command in that file's per-component section).
- **Correctness gate:** `test_fibo_transformer_mesh` (PCC ≥ 0.99 at production 2×2, sp=2/tp=2),
  run after **each** change. (PCC-only per decision; no per-change end-to-end image.)

## Baseline (from `denoise_report/`, one cache-warm forward, ~235 ms device-time-sum)

| Op class | % of forward | Note |
|---|---|---|
| MinimalMatmul | 38.9% | weighted 37.7% FLOP util; **min 0.66% / max 64%** — huge spread |
| RingJointSDPA (attention) | 18.7% | |
| CCL (all-gather + reduce-scatter + pre/post) | ~19.6% | tp=2 gathers before every attn/FF |
| Elementwise (BinaryNg + Unary) | ~13.9% | modulation muls/adds, gating, gelu, silu |
| TM (concat / create-heads / slice) | ~7% | |

Whole-forward DRAM roofline: **15.4%** — every op reads inputs from DRAM (`in0:dram_interleaved`).

Matmul time bucketed by FLOP utilization:

| FLOP util | matmul time | share of MM |
|---|---|---|
| **< 15%** | **32.8 ms** | **35.8%** (252 ops) — small-M prompt (M=128) + modulation (M=32) + caption |
| 30–44% | 5.3 ms | 5.8% |
| 45–59% | 33.2 ms | 36.3% — big M=2048 spatial FF/QKV |
| 60–100% | 20.2 ms | 22.1% |

FIBO config: `inner_dim=3072` (24 heads × 128), 8 dual + 38 single blocks, in=48,
joint_attention_dim=4096, text_encoder_dim=2048, patch=1. Per-device (tp=2): 1536.

## Two facts that shaped the plan

1. **No FIBO shape is in any tuning table.** All FIBO matmuls fall through `get_matmul_config()`
   to the generic `(8,8,8)` fallback (it logs "No known best blocking…"). A device-profiler
   block-size sweeper (`utils/sweep_mm_block_sizes.py`) and a `register_matmul_configs()` hook
   already exist for adding a model's shapes.
2. **`ttnn.experimental.minimal_matmul` exposes only block sizes** — no `memory_config`, no
   variant choice, no L1 sharding; it is a fixed DRAM-interleaved kernel. Using the
   `memory_config` / width-sharded-L1 / matmul-variant levers requires moving hot matmuls onto
   standard `ttnn.matmul`/`ttnn.linear` with a 1D/2D program_config + sharded memory_config.

## Phase 1 — Tune `minimal_matmul` block sizes for FIBO's shapes (safe, additive)

No structural/shared-code change; same kernels, better blocking → PCC-neutral.

1. **Enumerate** FIBO's real per-device matmul shapes: temporarily instrument `get_matmul_config()`
   to dump every unique `(M, K, N, grid, path)` during one denoise forward; run one forward; collect.
   Expected ~15–20 shapes in families: big spatial M=2048 (qkv K3072×N4608, ff1/proj_mlp N6144,
   ff2 K6144, single proj_out K7680, to_out), prompt twins M=128, modulation M=32, caption
   M=128 K2048 N1536 ×46.
2. **Add a `bh_2x2` device config** to the sweeper (mesh (2,2), FABRIC_1D, Linear, num_links=1,
   sp_axis=0, tp_axis=1), add FIBO's shapes to its SHAPE table with correct `use_case`
   (plain / qkv / ff1_gelu / to_out / ff2), run the sweep.
3. **Register winners** via a FIBO-scoped `register_matmul_configs()` call at model-build time
   (additive, keyed by `(M,K,N)` — cannot affect other models).
4. **Gate + measure:** `test_fibo_transformer_mesh` (PCC ≥ 0.99), then re-run
   `test_fibo_denoise_device_profile`, diff the util histogram vs. baseline.

## Phase 2 — Restructure top remaining hotspots (targets chosen from Phase-1 re-profile)

Deliberately not fully specified now. Likely: move the big M=2048 spatial FF/QKV to `ttnn.matmul`
with a 2D program_config + **L1 width-sharded** in/out `memory_config`, and keep activations
resident across norm→matmul→gate to cut DRAM round-trips (the 15.4% roofline). Each change is
independently PCC-gated and re-profiled; stop when returns flatten.

## Risks / constraints

- Shared block code (`transformer_block.py`, `attention.py`, `linear.py`) is used by Flux/SD35 —
  Phase 1 stays additive (registration only). Phase 2 changes must be FIBO-gated or proven neutral
  for other models.
- The sweeper times a matmul in isolation; the winning blocking should transfer (same kernel/shape/grid).
- Small-M (M=128/32) util is limited by too little M-parallelism for 120 cores; block tuning helps
  modestly — the real lever (grid restriction / fusion) is Phase 2.

## Out of scope

Attention (SDPA) kernel tuning, CCL topology changes, encode/VAE-decode stages, quantization.

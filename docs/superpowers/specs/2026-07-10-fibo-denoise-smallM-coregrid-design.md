# FIBO denoise Phase 2 — small-M matmul core-grid fix — design

Date: 2026-07-10
Branch: `fibo-pipeline`
Status: approved (brainstorming) → validation gate

## Context

Phase 1 (matmul block-size tuning) landed: matmul −5.8%, whole denoise forward −2.2%,
weighted FLOP-util 37.7%→40.0%, PCC 99.53%. New baseline: `denoise_report_after/`
(matmul 86.15 ms, forward 229.9 ms, DRAM roofline 15.7%).

Phase 1 did NOT move the `<15%`-util bucket: **252 ops, ~33 ms, 38% of matmul time** — the
`M=128` prompt matmuls and `M=32` modulation matmuls. The forward is only 15.7% bandwidth-bound,
so the earlier "L1 intermediates" idea has bounded upside; the dominant small-M waste is a
**core-grid occupancy** problem, not bandwidth.

## Root cause

`get_matmul_config` always passes the full **12×10 (120-core)** grid. `minimal_matmul` distributes
M-tiles across `grid_x`, so `M=128` (4 M-tiles) leaves 8/12 core-columns idle and `M=32` (1 tile)
leaves 11/12 idle. The op becomes dispatch/sync-overhead bound → 2–9% FLOP util.

## Approach (3 steps, Step 1 is a stop/go gate)

**Step 1 — Validate (hard gate).** Sweep representative small-M FIBO shapes at several candidate
grids (12×10 baseline vs smaller rectangles, e.g. 8×10, 6×10, 4×10, 8×8, 4×8) via the existing
`sweep_mm_block_sizes.py` (add entries with alternate `cgx/cgy`). Representative shapes:
`128×3072×4608` (qkv prompt, x92/run), `32×3072×9216` (modulation, x32/run),
`128×2048×1536` (caption, x92/run), `128×7680×3072` (single proj_out prompt, x76/run).
If no smaller grid meaningfully beats 12×10, **STOP and report** — the cost is dispatch/CCL, not
occupancy (and L1 wouldn't help either); Phase 1 was then the real ceiling for this lever.

**Step 2 — Mechanism (additive, opt-in).** Extend `get_matmul_config` so a registered config may
carry an optional core-grid override; when present, build the `MinimalMatmulConfig` with that grid
instead of the caller's full grid. Existing 3-/4-tuples unchanged → Flux/SD35 unaffected. Only
FIBO's registered small-M shapes opt in.

**Step 3 — Register + gate + measure.** Register best `(grid, block)` per improved small-M shape
in `transformer_bria_fibo.py::_register_fibo_matmul_configs()`. PCC-gate with
`test_fibo_transformer_mesh`. Re-profile vs `denoise_report_after`; confirm the `<15%` bucket
shrinks and forward time drops.

## Correctness / risk

- Grid + block are performance-only → PCC-neutral. Gate: `test_fibo_transformer_mesh` ≥ 0.99.
- `get_matmul_config` change is additive/backward-compatible; only FIBO's registered shapes use the
  override. The sweeper already handles arbitrary grids (it builds `MinimalMatmulConfig` directly).
- Only FIBO's (M,K,N) are registered, so no other model's lookup changes.

## Out of scope

L1/`memory_config` residency (bounded upside at 15.7% roofline; deferred), attention/CCL tuning,
the big-M matmuls (already tuned in Phase 1).

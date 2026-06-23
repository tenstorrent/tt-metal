# `wan_fused_distributed_rmsnorm`: `per_head_norm=True` on `ring_size>1` — RESOLVED

**Op:** `ttnn.experimental.wan_fused_distributed_rmsnorm` (branch `cglagovich/fused_rms_norm`)
**Status:** **FIXED** (2026-06-23). `per_head_norm=True` now runs on TP-sharded ring
meshes (FLUX TP=4 + TP=8 RING), deterministic, PCC 99.81–100.00% vs fp32 torch.

## What was wrong (the real root cause)

The earlier diagnosis — "back-to-back per-head matmul `reduce<SUM,REDUCE_ROW>`
calls wedge the matmul→pack pipeline, needs an LLK fix" — was **incorrect**. The
actual bug was a **circular-buffer routing/sizing mismatch** in the program factory:

- The factory's top-level `is_tp_1 = (ring_size==1) || per_head_norm` was passed to
  the **writer**, but the **compute** kernel's `is_tp_1` compile-time arg was set to
  `(ring_size==1)` only — it did **not** include `per_head_norm`.
- So for `per_head_norm && ring_size>1`: the writer took the `is_tp_1` drain-only
  path (no fabric AG, no stats forwarding), while the compute took the `is_tp_1==0`
  path → `stats_dest_cb = stats_local_cb`.
- For `per_head_norm`, `stats_local_cb` is sized for **1 tile** and has **no
  consumer** (the writer doesn't forward it). The PRE phase produces
  `num_heads_per_device` stat tiles per row, so it blocked on `cb_reserve_back`
  at the **2nd head's** reduce — presenting as a "reduce wedge."

This is why every attempted LLK-level workaround failed: there was nothing wrong
with the LLK pipeline.

## The fix

One line in `wan_fused_distributed_rmsnorm_program_factory.cpp`: set the compute
kernel's `is_tp_1` arg to the factory-level `is_tp_1` (`ring_size==1 ||
per_head_norm`) instead of `ring_size==1` only. The compute then takes the
`is_tp_1` path for `per_head_norm`: PRE pushes the `num_heads` per-row stat tiles
straight into `stats_gathered_cb` (sized for `num_heads`), POST consumes them
locally per head — fully self-contained, no AG, matching the drain-only writer.

## Validation (WH 4×8 galaxy)

`WAN_FLUX_PHN=1` `test_corr_det[flux_tp4_ring]` + `[flux_tp8_ring]`:
all 8 `per_head_norm=True` configs **det=OK**, no hang:

| config | pcc(F:torch) |
|---|---|
| flux_tp4_N64 / N512 / N2048 / N8192   | 99.81 / 99.999 / 99.994 / 99.997% |
| flux_tp8_N128 / N1024 / N4096 / N16384 | 99.999 / 100.001 / 99.989 / 100.003% |

`flux_tp4_N64_phn1` sits at 99.81% (below the 99.9% flag threshold) — this matches
the **composite baseline's own** accuracy for that tiny shape (`pcc(base:torch)`
≈ 99.81%), i.e. a bf16 precision characteristic of N=64 per-head norm, not a fused-op
regression. `per_head_norm=False` remains fully validated and unchanged.

# `wan_fused_distributed_rmsnorm`: `per_head_norm=True` deadlocks on `ring_size>1`

**Op:** `ttnn.experimental.wan_fused_distributed_rmsnorm` (branch `cglagovich/fused_rms_norm`)
**Severity:** blocks FLUX.2 per-head QK-norm (`per_head_norm=True`) on TP-sharded galaxy meshes.
**Status:** root-caused; needs a compute-kernel/LLK fix. `per_head_norm=False` is unaffected and fully validated (FLUX TP=4 + TP=8 RING, PCC 99.985–100%, deterministic).

## Symptom
`per_head_norm=True` (per-head RMSNorm over `head_dim`, no all-gather → `is_tp_1`):
- **Works** on a 1×1 mesh (`ring_size=1`) — `test_wan_fused_distributed_rmsnorm_device_op.py::...tp1_per_head_norm`, up to 8 heads.
- **Hangs** on a TP-sharded submesh (`ring_size>1`, e.g. the FLUX TP=4 / TP=8 RING on the 4×8 galaxy), with **and** without RoPE. Without a watcher the core silently halts (looks like a multi-hour hang); with the watcher it surfaces as a tripped state.

Repro: `WAN_FLUX_PHN=1 CORR_ONLY="flux_tp4_N64_phn1" pytest models/tt_dit/tests/test_distributed_rmsnorm_fused.py::test_corr_det[wormhole_b0-flux_tp4_ring] -x -s`

## Root cause (DPRINT-localized)
The per-head PRE phase does `num_heads_per_device` back-to-back row-reductions (`reduce<SUM,REDUCE_ROW>`) **within a single chunk**. Whole-row norm only ever does **one** reduce per chunk, separated by the AG-wait/POST boundary, so it never hits this. After ~3 back-to-back reductions the compute math/pack pipeline wedges — the **same matmul-reduce → pack wedge already documented and fixed for the POST whole-row reduce** (replaced with an FPU eltwise-add, commit `11bc6a0e056`).

A two-pass DPRINT experiment was the clincher: all `num_heads` per-head **squares** complete on every device (`pre_pass1_done` ×32), then the per-head **reduces** wedge before producing any output (`agwait_done`=0).

`reduce_uses_matmul()` is hardcoded `true` for `SUM`/`AVG` + `REDUCE_ROW`, so the matmul reduction can't be disabled via the helper.

## Fixes ruled out (each verified on-device)
1. Deepen `pre_intermediate_cb` 1 → `num_heads_per_device` — no change.
2. Force `num_workers=1` — still hangs.
3. Split PRE into a squares-pass then a reduces-pass — moved the hang from squares to reduces.
4. `reduce_uninit()` drain between reduces — no change.
5. Raw classic `reduce_tile` (non-matmul) instead of `compute_kernel_lib::reduce` — still hangs.
6. (5) + explicit scaler `cb_wait_front` — still hangs.

## Likely fix direction
The POST eltwise-add fix does **not** transfer directly: POST sums `ring_size` *tiles* (`add_tiles`), but the per-head PRE needs a *column* sum within a tile. Candidates: transpose + non-matmul reduce, a fused "squared-row-reduce" LLK, or batching the per-head reductions so the matmul→pack transition count stays ≤ the wedge threshold. Needs compute-kernel/LLK expertise.

## Workaround (shipped)
`models/tt_dit/tests/test_distributed_rmsnorm_fused.py` emits only `per_head_norm=False` FLUX configs by default; `WAN_FLUX_PHN=1` re-enables the hanging `per_head_norm=True` configs for fix work. Full analysis + speedup tables in `models/tt_dit/tests/RMSNORM_FUSION_FINDINGS.md` ("FLUX shapes" + "Known issue").

# HunyuanImage-3.0 — handoff (Nadim → Teja + Saurabh)

Handoff package for the 1-week bh-galaxy optimization sprint.

## At a glance
- **Model:** `tencent/HunyuanImage-3.0` — ~80B-total / ~13B-active MoE **text→image**, 32 layers, 64 routed experts (top-8) + 1 shared, hidden 4096. Diffusion-in-transformer: FlowMatch (Euler), 50 steps, CFG=2.
- **Parallelism ((8,4) mesh):** attention/dense **TP=8**; MoE **EP=32** (64 experts 2/chip across all 32 + shard-shared) — the 4-axis carries expert parallelism, not classic DP.
- **Current:** **~84 s/image warm** @1024²/50-step (216.5 cold, from 351.4 hybrid).
- **Bottleneck (latest Tracy):** **CCL-bound** — ReduceScatter+AllGather ~30% of device-kernel time, ahead of matmul (~55% of wall-clock with cross-chip sync).
- **Target:** ~40-50 s near-term · ~15-20 s aggressive (dense-MoE @ ~40% MFU). Floor ~5-8 s needs sparse dispatch (DEAD on TT).
- **Branch:** https://github.com/tenstorrent/tt-metal/tree/nkira/hunyuanimage3-bringup_bh_glx  ·  working: `hunyuan-image3-unified` @ `~/nkira/tt-metal-xtts`

## Files in this folder
- **`OPTIMIZATIONS_BY_DATE.md`** — every lever landed, dated, with perf deltas + phase summaries.
- **`STATUS_AND_MODEL_DETAILS.md`** — full model / TP-EP-DP details, roofline, ranked near-term levers, and how-to-run.

The **performance sheet** is the conventional [`PERF.md`](../PERF.md) at the model root (config table, E2E performance, optimization ladder, targets, references). Deepest ladder detail lives in the branch's `UNIFIED_STACK.md`.

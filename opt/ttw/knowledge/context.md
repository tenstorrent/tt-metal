# LTX-2.3 distilled AV — 3s opt loop context (project knowledge pack)

**North star:** drive end-to-end LTX fast-mode video generation (1088×1920, 145
frames = 6.04s clip, + audio) from the current fast operating point toward **3.0s
total device time** on the Blackhole 4×8 galaxy, WITHOUT shipping a broken/
different generation. Deliver the fastest point that still passes the quality gate,
and separately report any faster-but-lower-quality point honestly.

## Honest reachability (read this first)
3s is below FastVideo's own B200 headline (4.55s) and their number is NOT
apples-to-apples (silicon FP4 + FA4 + 121 frames vs our 145). The documented
lever ceiling from the handoff:
- L1 bf8 quant (already on): −4.4% (CCL-bound, so quant math barely helps).
- L2 step-cut: 8-step (6+2) HOLDS quality → ~6.5s; **7-step (5+2 or 6+1) FAILS**
  (PCC 0.31, a *different scene*). The current 5.71s "fast" point uses 6+1=7
  steps and was **never quality-gated** — treat it as suspect.
- L4 sparse/windowed attention (NOT built): ceiling 1.4–2.0× on SDPA ≈ 0.7–1.2s
  E2E. **This is the only lever with the headroom to approach 3s.**
So the realistic overnight arc is: (1) lock a genuinely quality-gated fast
baseline, (2) land L4 sparse attention on the dense RingJointSDPA (29% of every
DiT block), (3) chip CCL/data-movement (tilize/typecast/AG-fused matmul = 31–35%),
(4) Tier-3 (VAE trace, vocoder+BWE, RoPE fold). If 3s is not reachable at gate,
land the best gated point and document the exact gap + next lever. Do NOT fake 3s
by disabling the gate or shrinking frames (frame count is a product decision).

## The metric (what ms_view sums)
`ms_view` = sum of `TTW_TIMING <stage>=<ms>` from the **gen#1 traced steady-state
replay** (gen#0 is trace capture — never the reported number). Stages come from
`pipeline.last_timings`: stage1_denoise, latent_upsample, stage2_denoise,
vae_decode, audio_decode. Goal: ms_view ≤ 3000. Report p50/min/max if cheap;
the canonical number is the gen#1 replay total.

## The quality gate (metric_name = qgate)
The e2e test's built-in gates were OFF for the 5.71s measurement. Neither VBench
(consistency/quality) nor CLIP (prompt-alignment) catches the "different but
coherent scene" failure mode of over-aggressive step cuts — only comparison to a
**fixed 11-step bf16 reference generation (seed 10)** does. Gate:
- Primary: decoded-frame **PCC vs the 11-step bf16 ref** via `~/compare_videos.py`,
  threshold **≥ 0.85** (8-step holds at 0.90; leave margin). This is `qgate`.
- Secondary sanity (run on keeps, cheaper cadence): CLIP prompt-alignment
  (`RUN_CLIP=1`) and VBench (`RUN_VBENCH=1`) must not regress below the test's
  built-in thresholds.
A perf iteration that improves ms_view but drops qgate < 0.85 is a REJECT, not a
keep — it changed the output.

## Bench set / representative timing
Single canonical input for now: `DEFAULT_LTX_PROMPT`, seed 10, 145f, 1088×1920,
`bh_4x8sp1tp0_ring`. (One input — the "hero" — because a full e2e gen is minutes;
the gate compares this generation to its own 11-step ref. If a lever's win looks
input-specific, add a second prompt/seed before banking.)

## Pipeline stages & where the time is (fast point, ~5.71s ungated)
stage1_denoise ~2.02s · latent_upsample ~0.21s · stage2_denoise ~2.03s ·
vae_decode ~0.95s · audio (voc+bwe) ~0.48s. DiT denoise = 71% of total.
Per-DiT-block (Tracy, 518ms): matmul/CCL 35%, **RingJointSDPA 29% (dense)**,
tilize/typecast/other 31%, RMSNorm 4%. → attention/CCL/data-movement bound, NOT
GEMM/FPU. bfp8 correct; bfp4 useless; sparse attention is the ceiling.

## Env / how to run (all device work via tt-device-mcp broker)
Runner: `tmp/run_ltx.sh` (sets the ltxperf650 env bundle + pinned-mem fix).
Broker: `tt_device_job_run_bg` owner `[claude]smarton`, workspace = this worktree,
`inherited_env.PYTHON_ENV_DIR=<worktree>/python_env`, `timeout_sec` ≥ 2400.
Key env: `LTX_FAST=1` (bundles `LTX_QUANT=all_bf8_lofi` + 6+1 sigmas + traced +
weight cache), `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` (or trace hangs on
4×8), `TT_DIT_CACHE_DIR=/home/smarton/.cache/tt-dit` (warm weights),
`LTX_CHECKPOINT`/`GEMMA_PATH` in kevinmi's HF cache. `LTX_TIME_STAGES=1`.
For a quality-gated run: `RUN_VBENCH=1 RUN_CLIP=1` and compare output mp4 to the
11-step ref with `~/compare_videos.py`.

## Iteration-time tooling (already applied)
`d90194e3de4` (kernel prewarm, HEAD `386ee5b0e2f`) moves JIT compile out of the
device-held window (default-on, `TT_METAL_KERNEL_PREWARM=0` disables). Cuts the
cold per-iteration device window. Also: `audio_only` fast path, single-DiT-block
profiling harness, and `WARM_REPS` shrink warmup for sub-full-mesh iteration.

## DO NOT re-attempt (proven dead — evidence in ~/LTX_PERF_HANDOFF.md)
1x4 audio submesh default-on (BH -6 abort, +189s cold); hide-audio via disjoint
submeshes or multi-CQ (architecturally impossible); bfp4_b; TeaCache/FBCache;
NVFP4/FA4 (silicon); exp_ring_sdpa (only fires on 4×32). 7-step subset of the
11-step schedule (quality cliff) — a schedule *distilled for 7 steps* would be a
model change, out of scope.

## Milestone plan
- **M0**: reproduce fast point on prewarm build + establish 11-step bf16 ref +
  measure qgate of the current fast point (is 5.71s even valid?).
- **M1**: lock the fastest *quality-gated* config (likely 8-step 6+2, ~6.5s) as
  the honest baseline.
- **M2**: L4 sparse/windowed SDPA — single-chip PCC probe → ring+joint integration
  → mask search. Biggest lever.
- **M3**: CCL/data-movement reduction (reshard/format/AG-link tuning).
- **M4**: Tier-3 (VAE decode trace/skip-tile, vocoder+BWE fusion, RoPE fold).

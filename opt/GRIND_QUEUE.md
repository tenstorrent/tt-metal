# GRIND QUEUE — concrete unmeasured experiments (the loop's never-empty work source)

**The loop NEVER concludes "exhausted."** Each lap: if no experiment is in flight, dispatch the next
`[ ]` item below (in-budget, completes, real data), mark it `[~]` in flight, and on completion record the
number here + PROGRESS LIVE LOG and tick it `[x]`. When ALL are `[x]`, generate the next batch by
re-profiling the current dominant op and enumerating its config space — do not stop.

All device runs via broker, owner `[claude]smarton`, `timeout_sec<=240`, in-budget (block/op harness, NOT
full pipeline — full pipeline cold-compiles >280s and times out even with prewarm; measure denoise deltas
per-block, VAE via prof_vae_ltx.py). Env: TT_METAL_HOME/PYTHONPATH=worktree, caches tt-*-ltxrt, PINNED=0.

## Batch A — SDPA chunk sweep (SDPA = 30% of block = 6.13ms; baseline chunk (192,512)=4.85ms; NEVER swept)
Measure RingJointSDPA FW (largest-FW op in the profile) per chunk. Test:
`test_ring_joint_attention.py::test_ring_joint_sdpa -k "ltx_s2 and bh_glx and 8rpx4up and no_trace_no_check and <qID> and <kID>"` under `tracy -p -r`.
**⚠️ HARNESS BUG — the REAL no-data cause (2026-07-11 21:56Z, from job 215000-35 FULL log): `tracy` DESTROYS a
multi-word `-k`. `tools/tracy/__main__.py:402` does `osCmd=" ".join(originalArgs)` then `:415` re-runs it under a 2nd
`shell=True` → the quotes around `-k 'ltx_s2 and bh_glx and …'` are lost → pytest sees `and` as a positional →
`ERROR: file or directory not found: and` → `collected 0 items`. This hits the driver's runs IDENTICALLY (also ~6.1s /
0 tests) ⇒ every prior Batch-A job (driver's + my 215000-35) collected NOTHING = ZERO data. My earlier "wh_glx wrong mesh"
theory was NOT the operative cause (both meshes fail at collection first). FIX = select by exact node-id BRACKET (no spaces,
survives tracy's join), e.g. `::test_ring_joint_sdpa[bh_glx-line-no_trace_no_check-k512-q128-8rpx4up-ltx_s2]` (exact id
being confirmed via collect-only job 215404-37). Secondary correctness (only matters once collection works): use `bh_glx`
=[(8,4),2] (num_links=2, has ltx_s2), NOT `wh_glx`=[(8,4),4] (num_links=4 HW-capped on BH, no ltx_s2). `SDPA_SWEEP_TAG` is a
no-op; run_test_ring_joint_sdpa prints no FW → extract SDPA device FW from newest `generated/profiler/reports/<ts>/
ops_perf_results_*.csv`. Do NOT `rm -rf .logs` (cold tracy recompile → 220s timeout, killed 213135-31). ALL prior Batch-A VOID.**
EXACT node-id format (from collect-only 215404-37, 384 ltx_s2 cells): `test_ring_joint_sdpa[blackhole-<mesh>-line-<check>-<kID>-<qID>-<parallelID>-ltx_s2]`.
Target cell dispatched: `...[blackhole-bh_glx-line-no_trace_no_check-k512-q128-8rpx4up-ltx_s2]`.
- [x] q128 k512  → **SDPA FW median 4.74ms / mean 4.91ms** (n=32, job 215546-39, JIT 173/175 warm, PASSED). vs 4.85ms baseline = within ±2.3% = **NO WIN** (>5% gate not met). Env-file dispatch: broker param is `env:` (not `env_file`), `opt/env_sdpa.yaml`.
- [x] q256 k512  → **SDPA FW median 5.335ms / mean 5.359ms** (n=32, tight 5.30–5.55, job 220638-41, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+10% SLOWER = NO WIN**. Trend now clear: q128 4.74 < baseline(q192) 4.85 < q256 5.34 ⇒ **smaller q is better; q256 direction is dead**.
- [x] q64  k512  → **SDPA FW median 7.866ms / mean 7.886ms** (n=32, tight 7.79–7.98, job 220953-43, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+62% SLOWER = DEAD**. Small-q direction REFUTED: q64 over-chunks (dispatch overhead dominates). U-shape confirmed → **q128 is the minimum** (7.87 q64 ≫ 4.74 q128 < 4.85 base < 5.34 q256). Small-q reorder abandoned.
- [~] q128 k256   (q=128 held at best, sweep k axis — job 221355-47, env=opt/env_sdpa.yaml, node-id `[blackhole-bh_glx-line-no_trace_no_check-k256-q128-8rpx4up-ltx_s2]`)
- [ ] q128 k128
- [ ] q256 k256   (deprioritized — q256 measured +10% worse at k512)
- [ ] q256 k128   (deprioritized — q256 direction dead)
→ record SDPA FW per config; fastest vs 4.85ms baseline is the win. If any config beats baseline by >5%, wire it into the LTX ring SDPA call (attention_ltx.py) + PCC-gate + block WARM_FWD_MS.

## Batch B — SDPA compute-config sweep (same harness, vary fidelity/accum)
- [ ] LTX_QUANT=all_bf8_lofi_sdpa_lofi_fp32acc on the one-block harness — SDPA LoFi + fp32 dest acc, PCC-gate.
- [ ] exp_ring_joint_sdpa variant (in-tree experimental kernel) — does it lower SDPA FW at ltx_s2 shape?

## Batch C — VAE decode sub-levers (prof_vae_ltx.py, NOT full pipeline; VAE stage = 1.0s)
- [ ] W-mask fold: drop the per-conv `ttnn.mul` width-mask before each W-sharded halo conv (~42 sites; kevinmi in-kernel mask). Measure decode wall delta + DECODE-TRACE-PCC.
- [ ] depth-to-space permute: profile the 4 upsample reshape→permute→reshape; is a fused kernel cheaper? (~84ms device bucket).

## Batch D — adaLN to_out fusion (attention_ltx.py; a2v/v2a/attn2 gated-residual into to_out epilogue)
- [ ] fold the 3 standalone `ttnn.addcmul` into the to_out matmul via the existing fused primitive attn1 uses; PCC-gate + block WARM_FWD_MS.

## Batch E — step-distill characterization (the 6s path; needs the PREWARM path to complete an E2E)
- [ ] Make a full-pipeline run actually COMPLETE: run the prewarm capture+compile, then submit the run TWICE (first warms the trace, second measures) OR raise the run-stage timeout via a 2-reservation split. Then measure 6+2 E2E speed + quality.
  NOTE: raw 4x8 full-pipeline = guaranteed timeout; this is why it's last + needs the multi-pass warm.

## DONE (measured, with the number)
- audio-trace: SHIPPED -0.3s. VAE-trace: 0.19ms DEAD. num_links=4: HW-capped. RMSNorm QK-merge: null (45.08 vs 44.03). tilize: cold artifact. all_bf8 weights: -0.04s null.

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
- [x] q128 k256  → **SDPA FW median 4.981ms / mean 5.158ms** (n=32, bimodal 16@~4.75 + 16@~5.1–6.0, job 221355-47, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+2.7% SLOWER = NO WIN**. k-axis now clear at q128: k512 4.74 < base 4.85 < k256 4.98 ⇒ **larger key-chunk better; smaller-k dead**. (q128,k512)=4.74 remains the global min (−2.3%, <5% gate).
- [x] q128 k128  → **SDPA FW median 5.884ms / mean 5.905ms** (n=32, tight 5.87–6.05, job 221731-49, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+21% SLOWER = DEAD**. k-axis at q128 fully closed: k128 5.88 > k256 4.98 > k512 4.74 ⇒ **larger key-chunk better; smaller-k dead** (mirrors q-axis U-shape). (q128,k512)=4.74 remains the global min (−2.3%, <5% gate).
- [x] q256 k256  → **SDPA FW median 5.882ms / mean 5.887ms** (n=32, tight 5.88–5.98, job 222219-51, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+21% SLOWER = DEAD**. Confirms the doubly-dead corner (q256 dir +10% × smaller-k dead). (q128,k512)=4.74 remains the global min (−2.3%, <5% gate).
- [x] q256 k128  → **SDPA FW median 7.368ms / mean 7.367ms** (n=32, very tight 7.36–7.37, job 222548-53, JIT 170/175 warm, PASSED). vs 4.85ms baseline = **+52% SLOWER = DEAD**. Closes the q256 row: k128 7.37 > k256 5.88 > k512 5.34 (same larger-k-better U-shape).
→ **BATCH A COMPLETE (full grid swept, all 7 cells).** Global min = **(q128,k512)=4.74ms, only −2.3% vs 4.85ms baseline (<5% act gate) ⇒ SDPA chunk offers NO shippable no-finetune win.** Both axes are U-shaped with baseline (q192,k512) near-optimal: q64 7.87 ≫ **q128 4.74** < q192 4.85 < q256 5.34 (q-axis @k512); k128 5.88 > k256 4.98 > **k512 4.74** (k-axis @q128). Nothing to wire into attention_ltx.py.

## Batch B — SDPA compute-config sweep (same harness, vary fidelity/accum)
- [x] SDPA LoFi + fp32-acc  → **DEAD (both counts)**. Job 223059-55, node-id `[blackhole-bh_glx-line-no_trace_check-k512-q128-8rpx4up-ltx_s2]`, env `LTX_SDPA_FIDELITY=LoFi LTX_SDPA_FP32ACC=1`. (1) **FAILS PCC gate** — `assert passing` (test:576, threshold 0.994) tripped (exact PCC at logger.debug, not captured). (2) **No speedup** — SDPA FW median **4.673ms** (n=32) vs HiFi2 q128k512 4.74ms = ~1.4%, within noise ⇒ **SDPA is NOT math-fidelity-bound** (HiFi2→LoFi barely moves FW). ⚠️ `LTX_QUANT` is a **no-op** in this test (never read); compute config hardcoded at test:384-389 → needed a temp env-gate source edit, **REVERTED** (no win + no PCC pass). fp32acc-alone = HiFi2 fidelity + extra acc cost ⇒ would PCC-pass but run *slower* ⇒ **whole fidelity axis closed, no further point worth a run.**
- [x] exp_ring_joint_sdpa variant (in-tree experimental kernel) — **DEAD: structurally incompatible with ltx_s2 at any performant chunk.** Probe **225026-62** (plain, non-tracy) crashed at **op validation** (SIGABRT, `1 failed in 42.61s`, NOT a hang/compile-timeout — post-job health-gate clean, no reset): `ExpRingJointSDPADeviceOperation::validate_on_program_cache_miss` → **"Q chunks per head (num_local=26+num_joint=0=26) exceeds SDPA grid columns (11) on device grid 12×10. Increase q_chunk_size or reduce sequence length."** The exp op maps one Q-chunk per grid column (≤11), so ltx_s2 (seq 38760 → 26 chunks at q≈192) can only run if q_chunk grows ≥~2.4× (into the ≥q454 regime). **Batch A already measured that direction monotonically SLOWER** (q256 +10%, q-axis U-shaped with q128=4.74 the min) ⇒ the exp kernel can ONLY run where it's already proven to lose. No win path; topology confound (Ring vs Linear) moot. Test-only scaffold commit fd2d0b2e1df kept (harmless, documents the attempt). **⇒ BATCH B CLOSED (both items DEAD): SDPA is neither chunk- nor fidelity- nor exp-kernel-improvable.**

## Batch C — VAE decode sub-levers (prof_vae_ltx.py, NOT full pipeline; VAE stage = 1.0s)
- [~] W-mask fold: drop the per-conv `ttnn.mul` width-mask before each W-sharded halo conv (~42 sites; kevinmi in-kernel mask). Measure decode wall delta + DECODE-TRACE-PCC. **Baseline in flight: job 225635-65** = plain (non-tracy) `prof_vae_ltx.py::test_prof_vae_ltx_trace` @ production 145f/1088×1920, env=opt/env_sdpa.yaml — establishes fresh-HEAD UNTRACED/TRACED_DECODE_WALL_MS (the A/B reference) before touching source. Plain not tracy: VAE-decode kernels are JIT-warm in ltxrt cache (185309-15 populated them); a tracy instrumented build would cold-compile (never built this grind → the exp-op timeout risk). NEXT: read 225635-65 → record baseline wall; then apply the ~42-site mask fold + re-run for the delta, PCC-gate vs with-mask output.
- [ ] depth-to-space permute: profile the 4 upsample reshape→permute→reshape; is a fused kernel cheaper? (~84ms device bucket).

## Batch D — adaLN to_out fusion (attention_ltx.py; a2v/v2a/attn2 gated-residual into to_out epilogue)
- [ ] fold the 3 standalone `ttnn.addcmul` into the to_out matmul via the existing fused primitive attn1 uses; PCC-gate + block WARM_FWD_MS.

## Batch E — step-distill characterization (the 6s path; needs the PREWARM path to complete an E2E)
- [ ] Make a full-pipeline run actually COMPLETE: run the prewarm capture+compile, then submit the run TWICE (first warms the trace, second measures) OR raise the run-stage timeout via a 2-reservation split. Then measure 6+2 E2E speed + quality.
  NOTE: raw 4x8 full-pipeline = guaranteed timeout; this is why it's last + needs the multi-pass warm.

## DONE (measured, with the number)
- audio-trace: SHIPPED -0.3s. VAE-trace: 0.19ms DEAD. num_links=4: HW-capped. RMSNorm QK-merge: null (45.08 vs 44.03). tilize: cold artifact. all_bf8 weights: -0.04s null.

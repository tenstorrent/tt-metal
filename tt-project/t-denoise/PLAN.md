# Denoise plan toward the 5.0 s stretch target (#181)

Date: 2026-10-07. CPU analysis only; no device jobs were run for this plan.

## Where the time goes today

Warm 4x8 1080p 6s on blx01, t48 `ttp/t48-ltx25-integrated` @ f6b806516cc: **5.671 s** mean of 5 seeds
(job 710: 5.719 / 5.666 / 5.669 / 5.619 / 5.683 s, tt-project/t171/NOTES.md).

| Stage | Time | Basis |
|---|---|---|
| S1 denoise, 8 steps | ~2.10 s (~260 ms/step, 48 blocks x ~5.2 ms + ~10 ms model ops) | t170 summary_configs.md, STEP_MS |
| S2 denoise, 3 steps | ~2.16 s (~715 ms/step, ~14.9 ms/block) | same |
| VAE decode | ~0.5 s | t171 stage table |
| Audio decode | ~0.38 s (overlap being checked in #179) | same |
| Encode / export / upsample | 0.2 / 0.15 / ~0.155 s | same |

To reach 5.0 s we need about **-0.67 s** e2e. Denoise is ~4.3 s of the 5.67 s.

Old per-block profile (t14, LTX-2.3 weights, before gate/adaln fusions; us per block, S1 / S2):
total 6006 / 16398; video self-attn 1602 / 6871 (ring SDPA 710 / 4286); FFN 979 / 2586;
V2A 834 / 2163; A2V 783 / 1841; text cross 754 / 1905; audio 1054 / 1031. At S2 by op type,
ring SDPA 5366, AGMM 2609, matmul 1759, all-gather 1321, MMRS 1302, strided AGMM 1060,
distributed RMSNorm 1001 (20 ops). The CSVs are gone; task 3 makes a fresh one.

## Key finding

The only levers that reach -0.67 s alone are **step cuts**, and the code already has them:
`LTX_S1_SIGMAS` / `LTX_S2_SIGMAS` (pipeline_ltx_distilled.py:37-63 on t48) and the `medium` tier
in `models/tt_dit/utils/ltx.py:126-166` (S1 8->6, S2 3->1, plus bf8). They were validated on
LTX-2.3 at 2x4 only. The comment there says the S1 head 1.0->0.975 is "redundant within the
structure basin" and one S2 step "holds composition". FastVideo runs the same LTX-2.3 distilled
checkpoint at 5+2 steps (LTX_PERF_PLAN.md, Glean). Nobody in this project has measured a step cut
on LTX 2.5. Each S2 step costs ~715 ms and each S1 step ~260 ms, so S2 3->2 alone gives ~-0.7 s.

Every step cut changes the output, so it is judged on 5-seed visuals and VBench, not PCC. The
kernel-level items (tasks 3-8) are smaller (each -0.05 to -0.35 s) but exact or nearly exact.

## Quality gate (all tasks)

1. PCC/PSNR against `ref_t48_f6b8` (DEFAULT prompt, seeds 0-4, tt-project/data/g15/ref_t48_f6b8).
   Bit-exact or PSNR >= 40 dB: accept on numbers.
2. Otherwise VBench (subject consistency, motion smoothness, aesthetic, imaging quality) on the 5
   seeds, against ref_t48_f6b8; within 0.01 of the reference on each.
3. Then a 5-seed side-by-side visual check (stills + videos shown to the user per the charter).
   Any visible degradation (detail, motion, artifacts, audio sync) rejects the change.

## Device jobs (all tasks)

- One config or one A/B pair per job, full 4x8 mesh, or create_submesh(2,4) of it for block tests.
  Never a bare (2,4) mesh. Through the box's broker; on blx03 only through the serial runner.
- Timeout = measured time +50%, at most 600 s. A 5-seed e2e config on a warm cache took 152 s
  (g15 job 406: 57 s warmup with LTX_WARMUP_T2V_ONLY=1 LTX_WARMUP_ENCODERS=0, then 5 gens), so use
  240 s. Block-trace tests are unmeasured on f6b8: 600 s on the first run, then resize.
- Box order: blx01 (READY, warm 4x8 cache, ref_t48_f6b8 was made there; skip configs that dropped
  on tray 3), then the blx03 runner, then an idle (2 h+) exabox dit node. g15blx02 only once its
  2026-10-06 23:00 pause is lifted.

## Ranked tasks

| # | Task | Expected saving | Exact? | Device |
|---|---|---|---|---|
| 1 | S2 step cut 3->2 | -0.70 s | no | 1 config job |
| 2 | S1 step cut 8->6 | -0.52 s | no | 1 config job (+1 combined) |
| 3 | Fresh per-op block profile at f6b8 | enabler; S1 gap ~-0.15 s | yes | 2 jobs |
| 4 | Audio branch without TP | -0.15 to -0.25 s | near | block A/B + e2e |
| 5 | Sliding-window S2 video self-attention | -0.2 to -0.35 s | no | block A/B + e2e |
| 6 | num_links 4 on the BH 4x8 ring | 0 to -0.3 s | yes | block A/B |
| 7 | bf8 weight-only DiT linears | 0 to -0.1 s | no | 1 config job |
| 8 | Exact host/step-boundary trims | -0.03 to -0.06 s | yes | 1 config job |

### 1. S2 step cut 3->2 (config only)

- **Saving:** -0.70 s. Basis: S2 step = 715 ms (t170 fast5 STEP_MS); the S2 pre-step work is unchanged.
- **Change:** none in code. `LTX_S2_SIGMAS=0.909375,0.421875,0.0` (a subset of the shipped nodes,
  so it stays on the distilled trajectory). Second arm if that fails visuals: `0.909375,0.725,0.0`.
- **Quality risk:** moderate. S2 refines detail at full res; dropping 0.725 may soften texture
  or hair. LTX-2.3 evidence says even one S2 step holds composition. Gate as above.
- **Device:** yes. One job: the t48 head with the sigma env, seeds 0-4, DEFAULT prompt, 240 s.
  Reference arm already exists (ref_t48_f6b8), so no baseline arm needed.
- **If accepted:** keep it opt-in (the env already exists). Making it the default changes the
  served High tier, so that goes to the user with the 5-seed videos.

### 2. S1 step cut 8->6 (config only)

- **Saving:** -0.52 s. Basis: S1 step = 260 ms x 2.
- **Change:** none. `LTX_S1_SIGMAS=1.0,0.9875,0.975,0.909375,0.725,0.421875,0.0` (the `medium`
  S1 schedule, bf16, no bf8). Second arm: `1.0,0.975,0.909375,0.725,0.421875,0.0` (8->5, -0.78 s).
- **Quality risk:** low to moderate. The four head nodes sit within 0.025 of sigma 1; LTX-2.3
  found them redundant. Watch interior-keyframe i2v (`_keyframe_s1_sigmas` reads the tier length)
  only if the served path uses it; the 6s t2v target does not.
- **Device:** yes. One job, seeds 0-4, 240 s. If task 1 was already accepted, run the combined
  S1+S2 config as a second job (expected ~4.45 s e2e).

### 3. Fresh per-op block profile at f6b8

- **Saving:** none directly; explains the S1 floor. The block test predicted 4.81 ms/block with
  gate+adaln (~241 ms/step) but e2e shows ~260 ms/step: ~20 ms/step, ~0.15 s over 8 steps, is
  unaccounted. It also re-ranks tasks 4-8 with current numbers.
- **Change:** none. `test_ltx_transformer_block_trace_perf` with the tracy device profiler, one
  stage per job (S1 shape N=9728, S2 shape N=38912), f6b8 defaults on. A full traced-pipeline
  tracy profile overflows its buffer (ltx-1080high-rt notes), so stay at block level.
- **Quality risk:** none.
- **Device:** yes, 2 jobs, full 4x8 (or create_submesh(2,4) only if 4x8 block test is unsupported),
  600 s first run. Output: per-op CSV + a table like t14's, committed to the task's NOTES.md.

### 4. Audio branch without TP

- **Saving:** -0.15 to -0.25 s. Basis: audio is ~1.0 ms/block at both stages (t14) for only 256
  tokens (32/device); most of it is TP collectives on tiny tensors. Replicated audio weights
  remove them. #22 plan estimate.
- **Change:** audio attention/FFN weights replicated across the TP axis (~+3.6 GB/device); drop the
  audio all-gather/reduce-scatter. Opt-in env first.
- **Quality risk:** low; same math, different reduction order. Expect PSNR >= 40 dB.
- **Device:** check DRAM headroom next to the VAE on CPU first (weights sizes from the checkpoint).
  Then one block A/B job (audio path on/off, 600 s first run), then one e2e config job (240 s).

### 5. Sliding-window S2 video self-attention

- **Saving:** -0.2 to -0.35 s. Basis: S2 ring self SDPA ~4.6 ms/op x 48 x 3 = 0.66 s (t32);
  a window that skips ~half the KV chunks cuts that by ~50%. Sliding-tile attention reports up to
  3.5x attention speedups at iso-quality on video DiTs (LTX_PERF_PLAN.md L4).
- **Change:** ring_joint_sdpa already has a sliding-window work plan
  (`sliding_window_work_plan.hpp`, used by GPT-OSS sliding prefill, PR 51438). Expose a temporal
  window (in latent frames) for S2 video self-attention only, opt-in env. Check upstream #57979
  (multicast multi-hop sliding halo) and #57190 (block-cyclic sliding ring SDPA) before writing
  new kernel code; cherry-pick if they fit.
- **Quality risk:** high; changes what the model attends to. Try S2 only (S1 sets composition),
  wide window first. Gate on VBench motion smoothness and visuals.
- **Device:** block A/B (window on/off, S2 shape) for speed, then one e2e config job per window.

### 6. num_links 4 on the BH 4x8 ring

- **Saving:** 0 to -0.3 s. Basis: collectives are ~6.3 ms of the 16.4 ms S2 block (t14:
  AGMM/sAGMM/MMRS/all-gather), partly link-bound at num_links=2. WH 4x8 uses 4.
- **Change:** `num_links` in the BH (4,8) device_config (pipeline_ltx.py ~595-640), opt-in env.
- **Quality risk:** none (exact). Hang risk: first confirm on CPU how many ethernet links each BH
  Galaxy neighbor pair has (cluster descriptor on the box, fabric `get_num_links`). If it is 2,
  close the task with no device job.
- **Device:** one block A/B job at S2 shape (num_links 2 vs 4), 600 s first run; e2e only if it wins.

### 7. bf8 weight-only DiT linears

- **Saving:** 0 to -0.1 s. Basis: 22B params / TP 4 = ~11 GB bf16 per device per step; at ~0.5 TB/s
  that is ~21 ms of a 260 ms S1 step. bf8 halves it: <= ~10 ms/step S1, less at S2.
- **Change:** none. `LTX_QUANT=all_bf8_lofi LTX_QUANT_ACTIVATIONS=0` (weights bf8, activations bf16).
  LoFi and bf8 SDPA inputs were null or harmful before (see rejected list), so this arm isolates weight bytes.
- **Quality risk:** moderate (~0.25 PCC drop for the full preset on 2.3). Gate as above.
- **Device:** one config job, 240 s. Needs a bf8 weight cache: check /home free space first
  (no new caches on g15blx02; on blx01 caches go under /var/tmp/fasth3).

### 8. Exact host/step-boundary trims

- **Saving:** -0.03 to -0.06 s. Basis: `LTX_LATENT_STATS` still defaults to 1
  (pipeline_ltx_distilled.py:1804), ~17 ms per gen (commit 925e11369b4); the 155 ms upsample and
  per-step host syncs between trace replays are unprofiled.
- **Change:** default `LTX_LATENT_STATS=0`; time the S1->S2 hand-off and upsample in a
  host-side log; remove per-step syncs that are not needed.
- **Quality risk:** none (exact; verify byte-identical against ref_t48_f6b8).
- **Device:** one config job, 240 s.

## Not planned, and why

- TeaCache/FBCache-style step or block caching: at 8+3 distilled steps the cache hit rate is too
  low to pay for its quality cost (LTX_PERF_PLAN.md); a straight step cut (tasks 1-2) gets the same
  saving with a cleaner trajectory.
- Already rejected: LTX_SDPA_EXP_APPROX, BH SFPU constant hoists, fused neighbor_pad+conv3d,
  LTX_FUSE_NORM_ADD, LTX_SDPA_MM_LOFI, agmm, hostcopy, V2A split-K (t32, 2x slower).
- exp_ring_sdpa: only fires on 4x32 (sp 32), not our 4x8.

## Sources

- tt-project/t171/NOTES.md, tt-project/data/g15/ref_t48_f6b8/run.log (job 710)
- blx01 /var/tmp/fasth3/t170/res/summary_configs.md (t170 eval pack, STEP_MS)
- tt-project/research/ltx_denoise_t14.md, denoise_plan.md, ltx_sdpa_pr57395_t31.md, upstream_scan.md
- origin/ttp/t32-ring-joint-sdpa-on-ltx-2-5-denoise-re-sw NOTES.md (chunk sizes, split-K)
- t48 models/tt_dit/utils/ltx.py:126-166 (fast/medium tiers), pipeline_ltx_distilled.py:37-63
- Glean: LTX_PERF_PLAN.md (Drive, 2026-06-15; FastVideo 5+2 on LTX-2.3, STA, caching skip)
- ~/.tt-buddy/notes/ltx-1080high-rt.md (LoFi null, SDPA-bound S2, tracy limits)

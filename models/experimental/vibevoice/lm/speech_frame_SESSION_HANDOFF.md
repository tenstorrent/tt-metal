# VibeVoice-1.5B speech-frame decode optimization — SESSION HANDOFF PROMPT

Paste this into a new session to continue. It is self-contained: current state, findings, the
acceptance methodology (with a correction), remaining levers, dead-ends, and concrete next steps.

---

## Objective
Optimize the **device performance** of the VibeVoice-1.5B full speech-frame decode on **Blackhole
P150**. A speech frame = neg-LM (28L Qwen2 decode) + diffusion (DPM CFG × 10 steps, B=2 head) + post
(acoustic decode conv → semantic encode conv → 2 connectors) + pos-LM (28L decode + constrained lm_head).

## ⚠️ OVERRIDING CONSTRAINT (do not forget)
The deploy target is the **90–100 min autoregressive render** (`4p_climate_100min`, ~42k frames, traced).
The non-negotiable acceptance criterion is that the **full-length audio stays clean and intelligible**.
The AR loop is **chaotically sensitive**: a change that is PCC-0.999x and clean on a short render can
shift the bf16 trajectory just enough that, compounded over ~40k steps, the render collapses (loud→clip,
or silent, or gibberish) — typically onset by min 8–55. **This session proved that live** (see Findings).
Short-render PCC/audio is NECESSARY BUT NOT SUFFICIENT. Byte-comparison to the clean baseline is the gold
standard. Treat every program-config / fidelity / fusion / precision change as **math-changing until
proven byte-identical (`maxabsdiff==0`)**.

## Current git state
- Branch **`ign/vibevoice1.5_fix`**. Work committed this session (on top of `f87bcd6bac2`):
  - `32d9014e1f3` — **revert prefill opts** (op-count `dbb709695b1` + HF-RoPE-fuse `d0d80b478bc` +
    lm_head-last-token `9653c9ffcf1`). These collectively collapsed the long-form render. Reverting
    restored the LM to the pre-prefill state (== the `0c21de7abeb` baseline LM).
  - `77feae8f536` — **opt A**: diffusion DPM-loop dedup (hoist step-invariant `cond_proj`+`cond_combined`
    out of the 10-step loop; compute `silu(c)` once/step shared across head+final layers). Tier-0
    byte-identical. **−643 µs / −58 ops per warm frame** (72,928 → 72,285 µs).
  - `81b16a47d70` — env-gated `VV_PROFILE_SPEECH_FRAME` op-level profiling signposts + restored
    `run_speech_frame_tracy.sh` (inert; eager-path only).
- Working tree clean except a pre-existing `lm/speech_frame_perf_prompt.md` edit (not ours; ignore).

## HARD-WON FINDINGS (read before doing anything)

1. **The current committed HEAD (`ign/vibevoice1.5_fix` tip, after the revert) reproduces the clean
   baseline BYTE-FOR-BYTE.** A full traced `4p_climate_100min` render of this state is
   `maxabsdiff=0` vs `~/vibe-voice/vv_lb_FULL/stream_loopbreak.f32` over all 42,170 frames (min-81
   residual peak 3.688 and every whisper transcript match). ⇒ **`stream_loopbreak.f32` is once again the
   valid byte-baseline for this branch.** Kept clean full render also at
   `~/vibe-voice/vv_cleanbase_optA_FULL/stream.f32` (42,268 frames, byte-identical over the common 42,170).

2. **The prefill opts caused a long-form regression (now reverted).** Before the revert, HEAD's traced
   100-min render collapsed to gibberish by ~min 55 (whisper degenerate; energy RMS drift 0.09→0.38,
   sustained peak-1.0 clipping from min 63) — vs baseline clean-to-min-77. The HF-RoPE fuse
   (`d0d80b478bc`) is only PCC 0.999999 (NOT bit-exact); the op-count (`dbb709695b1`) + lm_head-last-token
   (`9653c9ffcf1`) claim bit-exact. **Chaotic interaction confirmed:** reverting ONLY the RoPE-fuse (keeping
   the other two) made it WORSE (silent collapse by min 9), not better — so it is not a single culprit.
   Prefill is one-time (TTFT); reverting cost prefill ~52→82 ms but ZERO steady-decode-frame cost.

3. **DEFERRED (user's plan, do this later, NOT urgent):** re-add each prefill opt one-by-one to find
   which are long-form-safe (to recover the TTFT speedup). **Efficient method:** a prefill numerical
   change shows up at **frame 0** (prefill feeds the first diffusion condition), so a **cap-400
   byte-compare vs baseline** is a fast exact test — `maxabsdiff==0` ⇒ that opt is truly bit-exact ⇒
   long-form-safe by construction (no full render needed); only a non-bit-exact opt needs the full
   long-form render. Bisect by `git checkout <commit> -- tt/ttnn_vibevoice_lm.py` for each incremental
   state: `dbb709695b1` (op-count) → `d0d80b478bc` (+RoPE-fuse) → `9653c9ffcf1` (+lm_head-last-token =
   old HEAD, known-collapse). This is TTFT-only recovery — low priority vs the decode-frame levers below.

## Baseline op-level profile (warm EAGER speech frame, the optimization map)
`VV_PROFILE_SPEECH_FRAME=2 VV_PROFILE_SPEECH_FRAME_EXIT=1 VV_TRACE_SEGMENT=0`, demo `1p_CH2EN`.
One warm frame = **72,285 µs device / 4,963 ops** (post-opt-A). Distribution (pre-opt-A numbers,
proportions unchanged): Matmul 60.5% (44 ms, 722 ops, weight-DRAM-bound at M=1), rms_norm 7.9% (5.8 ms,
270 ops), Conv2d 7.3% (5.3 ms, 88 ops), **ArgMax 6.3% (4.6 ms) — EAGER-ONLY (deploy uses constrained
subset, ~µs)**, eltwise 4.6%. Deploy (traced) steady ≈ **9.7–12.5 tok/s** (~103 ms/frame wall; frame is
substantially HOST-bound — D2H syncs + loopbreaker FFT — so device wins reduce device time but wall gains
need host overlap too). ~7 ms of the eager frame (full-vocab argmax + full lm_head) is ABSENT from deploy.
Deploy-relevant device time is dominated by the **two 28-layer LM forwards + the 10-step B=2 diffusion**.
Report saved: `lm/speech_frame_baseline.txt`; full plan: `lm/speech_frame_optimization_plan.md`.

## Remaining optimization levers (ranked)

### 1. CFG batch-2 LM fusion — FLAGSHIP (~9 ms, ~20% of traced frame). DO THE PROBE FIRST.
Today the frame runs TWO separate B=1 28-layer forwards (neg-LM, pos-LM). Decode matmuls are
weight-DRAM-bound at M=1, so one B=2 forward reads each layer's weights ONCE for both rows. Requires
software-pipelining (batch pos-LM(k−1) with neg-LM(k), both feeding diffusion(k)), a batched KV cache
`[2,…]` with per-row positions, `sdpa_decode(cur_pos_tensor=[p_pos,p_neg])`, per-row `paged_update_cache`.
Mechanisms proven bit-exact in isolation (`tests/perf/cfg_batch2_probe.py`). **Byte-identity vs the
current B=1 path is UNPROVEN** (the width-sharded `_QO_DECODE_PROGCFG` is `per_core_M=1`/B=1-only and
overflows at B=2; a B=2 progcfg may change matmul K-reduction order). **First run the isolated probe
`tests/perf/cfg_batch2_byteident_probe.py`** (written this session, not yet run): it measures row-0
`maxabsdiff` of a B=2 matmul vs B=1 for every LM decode matmul (auto matmuls keep batch in dim 0 →
should be batch-independent/bit-identical; the width-sharded wq/wo with a `per_core_M=2` variant is the
question). If all `maxabsdiff==0` ⇒ Tier-0, integrate hard. If not ⇒ Tier-2 (full-render gate) with real
long-form-collapse risk — decide with the user.

### 2. FFN down-proj `32×8960×1536` is SLOW (39% DRAM vs 76% on gate/up) — candidate, prove byte-identity.
A better `MultiCast1D` progcfg keeping `in0_block_w` (⇒ same reduction order) MAY be byte-identical.
Adopt only if a cap-400 byte-compare is `maxabsdiff==0`; else it is math-changing → Tier-2.

### 3. More Tier-0 dedups in the frame graph. Audit for redundant elementwise / typecast / reshape in
the traced decode + post pipeline. Any that are byte-identical are free wins.

## Acceptance gate (tiered — byte-compare-to-baseline is the gold standard)
Baseline reference (now valid again): `~/vibe-voice/vv_lb_FULL/stream_loopbreak.f32` (42,170 frames).
The baseline has a KNOWN residual repeat-loop ~min 77–83 (loop-break only partially fixes it; recovers by
~min 85), so the criterion is "matches the baseline", NOT "passes an absolute oracle".
- **Tier 0 (minutes, EVERY change):**
  ```
  OUT=~/vibe-voice/vv_gate; rm -rf $OUT; mkdir -p $OUT
  VV_STREAM_AUDIO=$OUT/stream.f32 python models/experimental/vibevoice/demo_ttnn.py \
      --demo 4p_climate_100min --trace --max_new_tokens 400 --output_dir $OUT
  python models/experimental/vibevoice/tests/perf/stream_bytecompare.py $OUT/stream.f32
  ```
  `maxabsdiff==0` ⇒ math-preserving ⇒ SAFE, adopt. (cap-400 = ~1 segment; also run one cap-1600 to
  confirm ≥4 segment-boundary trace recaptures don't crash.) NOTE: the demo has a ~52 s TTFT (23k-token
  prefill) before decode starts.
- **Tier 1 (reject-only):** `longform_energy_oracle.py <stream> 15` after a ~cap-9000 render.
- **Tier 2 (REQUIRED for any math-changing opt AND the cumulative stack):** full render (no cap, natural
  EOS ~93 min, ~57 min wall) with `VV_STREAM_AUDIO`, then `longform_energy_oracle.py <stream> 100` +
  `longform_whisper.py <stream> 2 10 20 40 55 68 75 83 86 90 93` + `stream_bytecompare.py <stream>`.
  Clean zone min 0–~77 flat (RMS ~0.06–0.12, peak ≤~0.95); whisper coherent everywhere except the known
  min-78–83 residual, and it must RECOVER after (min 83/86/90 on-script). Divergence vs baseline may
  appear ONLY in the min-77+ residual band.
Baseline profile confirmed this session: coherent min 40 & 68 ("...nonlinear acceleration" / "sliver of
hope"), residual min 80 ("So" / "Is it too late to pay?" loop), recovers by min 83/86/90.

## Edit BOTH frame paths
Op-level profile = EAGER path (`VV_TRACE_SEGMENT=0`: `_neg_lm_step`/`_run_speech_diffusion`). Deploy =
fused split-capture trace (`VV_TRACE_SEGMENT=1`, set by `--trace`: `_run_segment_frame_traced` →
`_negtrace`/`_dptrace`/`_postrace`). A change in only one path is invisible to the other — apply to both.
The long-form gate always runs the trace path. (Opt A only touched `sample_speech_latents`, shared by both.)

## DEAD-ENDS to avoid (documented / proven)
- Diffusion-head program-configs, bf8_b weights, HiFi2/LoFi anywhere, post-FFN/lm_head decode progcfgs —
  math-changing, collapsed the render collectively (the whole reason `vv-122f-fast` exists).
- Prefill opts (op-count/RoPE-fuse/lm_head-last-token) — collapse long-form (this session). Left reverted.
- Fidelity walks — frame is DM/dispatch/host-bound, not math-bound; costs long-form margin.
- depthwise-conv shift-MAC — device-slower.

## Operational hazards (cost real time)
- **`df -h /` before any long render.** Root fills from `~/vibe-voice/vv_*` dirs (0.5–1 GB each) + a
  ~250 MB golden-wav COPY the demo makes per run under `$OUT/4p_climate_100min/` (delete it after each
  render). Keep >2 GB free. KEEP baseline refs: `vv_lb_FULL`, `vv_split_FULL`, `vv_122f_eager_90`,
  `vv_cleanbase_optA_FULL`. ENOSPC mid-render crashes with a MISLEADING deep traceback.
- The tracy op-level profile leaves ~4.5 GB of raw CSVs in `generated/profiler/.logs/`
  (`profile_log_device.csv`, `tracy_ops_times.csv`) — delete after extracting the report. Profiling the
  full 32-token run overflows the profiler DRAM buffer; the signpost+EXIT (frame 2) keeps it bounded.
- After killing a render, reset the device: `/home/ubuntu/.local/bin/tt-smi -r 0`.
- Traced decode is deterministic (seed 0 default via `--seed`); a single run is decisive.
- Renders are LONG: full 100-min ≈ 57 min wall; cap-400 ≈ 90 s (incl. 52 s TTFT); cap-1600 ≈ 4 min.

## Harnesses & key files
- `tests/perf/run_speech_frame_tracy.sh` — warm-frame op profile → `lm/speech_frame_expN.txt`.
- `tests/perf/stream_bytecompare.py <cand.f32> [baseline.f32]` — Tier-0/2c gate (baseline default
  `~/vibe-voice/vv_lb_FULL/stream_loopbreak.f32`).
- `tests/perf/longform_energy_oracle.py <stream> [max_min]` ; `tests/perf/longform_whisper.py <stream> <min...>`.
- `tests/perf/cfg_batch2_probe.py` (mechanisms) ; `tests/perf/cfg_batch2_byteident_probe.py` (NEW, run first).
- Model: `tt/ttnn_vibevoice_generator.py` (frame graph, both paths), `tt/ttnn_vibevoice_lm.py` (Qwen2 28L,
  decode-traced path = deploy), `tt/ttnn_diffusion_head.py`, `tt/ttnn_dpm_scheduler.py`,
  `tt/ttnn_acoustic_tokenizer.py`, `tt/ttnn_semantic_tokenizer.py`, `tt/ttnn_speech_connector.py`.
- HF reference (golden): `models/experimental/vibevoice/reference/modular/`. LM = stock
  transformers Qwen2 4.51.3 (hidden 1536, 28L, 12 heads, 2 KV, head_dim 128, intermediate 8960,
  rope_theta 1e6). Diffusion head: 4 layers, ffn 4608, latent 64, 10 inference steps (demo default).
  Component PCC tests: `tests/pcc/test_{lm_decode,decoder_layer,diffusion_head,acoustic_tokenizer,
  semantic_tokenizer,connector}_pcc.py`, `test_e2e_generate_pcc.py`.

## Environment
```
source python_env/bin/activate; export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
```

## CONCRETE NEXT STEPS
1. (Optional, low priority) Bisect the prefill opts (deferred) to recover TTFT — cap-400 byte-compare method above.
2. **Run `tests/perf/cfg_batch2_byteident_probe.py`** → decide CFG batch-2 is Tier-0 (integrate) or Tier-2 (risky, ask user).
3. Re-profile a warm frame (`run_speech_frame_tracy.sh`) on current HEAD to re-confirm the bucket ranking, then attack the biggest deploy-relevant bucket with a byte-identity-first mindset.
4. One commit per adopted opt (msg: device-µs before→after + safety tier). Re-run Tier-2 on the cumulative stack before final.

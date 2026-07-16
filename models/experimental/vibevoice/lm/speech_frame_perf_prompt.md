# VibeVoice full speech-frame decode — device-perf optimization (LONG-FORM-SAFE)

Load **SKILL.md** and follow its workflow (**profile → isolate → integrate → re-profile + PCC**).
Read the whole model and the LM in **both** TT and HF reference form first, then generate the perf
report with the harness below and produce an optimization plan.

Optimize the **device performance** of the VibeVoice-1.5B full speech-frame decode on **Blackhole
P150**. A speech frame = **neg LM** + **diffusion** (CFG × num_steps) + **post** (acoustic decode +
semantic encode + acoustic/semantic connectors) + **pos LM**.

## ⚠️ THE OVERRIDING CONSTRAINT — the full 100-min render must stay clean

The deploy target is a **90–100 minute** autoregressive render (`4p_climate_100min`, ~42k frames), and
the non-negotiable acceptance criterion is that **the full-length audio remains clean and
intelligible**. A speedup that corrupts the long render is a **regression, not a win** — reject it.

**Why short verification is insufficient (this is the whole point of this rewrite).** The AR decode is
a ~40k-step feedback loop (LM decode → diffusion CFG×steps → acoustic decode → semantic encode →
connectors → back to LM). It is **chaotically sensitive**: an optimization that is "PCC 0.999x and
audio-clean on a short 32-token / few-second render" can still shift the bf16 rounding trajectory just
enough that, compounded over 40k steps, the render goes **loud → clips → collapses to silence or
degenerates to gibberish** (typically onset by min 8–15, and/or a hard collapse by min 60–70). This
has already happened: the full perf stack (diffusion program-configs + bf8_b + tiling changes) — every
op individually PCC-green — **collectively collapsed** the 90-min render (clipping by min 12, collapse
by min 67). Short-render PCC is **necessary but NOT sufficient. Do not use it as the acceptance gate.**

**Explicitly override SKILL.md here:** SKILL.md says a matmul program-config change "changes tiling not
math, PCC vs auto ~1.0, the safety check is trivial." That is true for short renders and **FALSE for
this long-form loop** — changing matmul tiling changes the reduction/rounding order at the bit level,
which is exactly what destabilizes the loop. Treat every program-config / fidelity / fusion / precision
change as **math-changing until proven byte-identical.**

## Classify EVERY change before adopting it

1. **Math-PRESERVING (byte-identical) — SAFE by construction.** Bit-for-bit identical output. Examples:
   skipping a computed-but-discarded output (e.g. the negative-CFG lm_head → `need_logits=False`);
   constrained-decode (subset lm_head + argmax == full-vocab masked argmax); memory placement /
   L1-chaining / resharding; split-capture trace restructuring; affine-into-weight folds and
   native-padding rewrites that are *provably* bit-exact; dedup of redundant elementwise. Adopt after
   the **Tier-0** gate alone.
2. **Math-CHANGING — UNSAFE for long-form until proven at full length.** Anything that alters the
   numeric result even at ~1e-4: matmul program configs (tiling → reduction order), bf8_b / lower
   fidelity (HiFi2/LoFi), fusions that reorder reductions, non-bit-exact algebraic reformulations. Each
   must pass the **Tier-2** full-length gate — AND, because individually-green opts collapse
   *collectively*, the cumulative stack must pass Tier-2, not just each opt in isolation.

## Acceptance gate (tiered — fail fast, but the FINAL bar is the full render)

Clean reference = the committed baseline (branch `vv-122f-fast` @ `0c21de7abeb`, off
`vv-122f-fp32rope`: fp32-rope + split-capture + loop-break). Kept reference stream:
`~/vibe-voice/vv_lb_FULL/stream_loopbreak.f32` (clean loop-break render, 42,170 frames). If missing,
regenerate a full render from the committed `vv-122f-fast` branch.

**The baseline ITSELF has a known residual degeneration ~min 78–83** (a bf16-drift repeat-loop the
loop-break only partially fixes; it self-recovers by ~min 85) and therefore trips an absolute energy
oracle in that band. So the acceptance criterion is **"matches the clean baseline,"** NOT "passes an
absolute oracle." **Byte-comparison to the baseline is the gold standard.**

- **Tier 0 — fast byte-identity gate (minutes; run for EVERY change).**
  ```
  OUT=~/vibe-voice/vv_gate; rm -rf $OUT; mkdir -p $OUT
  VV_STREAM_AUDIO=$OUT/stream.f32 python models/experimental/vibevoice/demo_ttnn.py \
      --demo 4p_climate_100min --trace --max_new_tokens 400 --output_dir $OUT
  python models/experimental/vibevoice/tests/perf/stream_bytecompare.py $OUT/stream.f32
  ```
  `maxabsdiff == 0.0` ⇒ math-preserving ⇒ **SAFE, adopt.** (This is exactly how constrained-decode +
  neg-skip were proven bit-safe.) cap-400 covers ONE speech segment; run **one cap-1600 render** too to
  confirm ≥4 segment-boundary trace recaptures don't crash.
- **Tier 1 — early-instability reject (medium; only if Tier 0 is NOT byte-identical).** Render ~cap 9000
  (≈20 min) with `VV_STREAM_AUDIO`, then:
  ```
  python models/experimental/vibevoice/tests/perf/longform_energy_oracle.py $OUT/stream.f32 15
  ```
  FAIL if any voiced minute in min 0–15 has peak > 1.25 (clip) or RMS > 0.20 (loud) or RMS < 0.02
  (collapse). The perf stack tripped this at min 12 — a cheap kill. (A window that stops before a late
  collapse can FALSE-PASS — never conclude "clean" from a short window; Tier 1 only *rejects*, never
  *accepts*.)
- **Tier 2 — full long-form gate (REQUIRED to commit any math-changing opt, and for the cumulative
  stack).** Full render (`--demo 4p_climate_100min --trace`, no cap → natural EOS ~93 min, ~47 min
  wall) with `VV_STREAM_AUDIO`, then:
  ```
  python models/experimental/vibevoice/tests/perf/longform_energy_oracle.py $OUT/stream.f32 100
  python models/experimental/vibevoice/tests/perf/longform_whisper.py $OUT/stream.f32 \
      2 10 20 40 55 68 75 83 86 90 93
  python models/experimental/vibevoice/tests/perf/stream_bytecompare.py $OUT/stream.f32
  ```
  - (a) **Energy oracle over the FULL length** — the clean zone (min 0–~77) must stay flat (RMS
    ~0.06–0.12, peak ≤ ~0.95, no clipping), matching the baseline profile.
  - (b) **Windowed Whisper** — coherent, on-script climate dialogue everywhere EXCEPT the known
    min-78–83 residual, and it must **recover** after it (min 83/86/90/93 on-script).
  - (c) **`stream_bytecompare` vs the baseline** — any divergence must appear ONLY in the min-77+
    residual band and be no worse / no earlier than the baseline.
  Then **re-run Tier 2 on the full accumulated stack** before the final commit (collective effect).

Verification harness (committed, runnable out of the box; `VV_STREAM_AUDIO=<path.f32>` makes the demo
append each frame's fp32 samples so you can gate a still-growing or finished stream):
- `tests/perf/stream_bytecompare.py <cand.f32> [baseline.f32]` — maxabsdiff + first-diff frame vs the
  clean baseline (default `~/vibe-voice/vv_lb_FULL/stream_loopbreak.f32`). The Tier-0 / Tier-2c gate.
- `tests/perf/longform_energy_oracle.py <stream.f32> [max_min]` — per-minute peak/RMS → PASS/FAIL.
- `tests/perf/longform_whisper.py <stream.f32> <min...>` — 30 s whisper-medium window per minute.

## Frame graph (generator `_frame` / eager path)
```
cond_pos ← hidden
neg_hidden ← LM(neg_embed)                                   # full 28L + lm_head (logits DISCARDED)
latent   ← DPM/CFG(diffusion_head, cond_pos, cond_neg, noise) # × num_steps  (already a B=2 fwd)
fused, audio ← post(latent)                                  # acoustic decode → semantic encode → 2 connectors
logits, hidden ← LM(fused)                                   # full 28L + lm_head
```

## Model files
```
models/experimental/vibevoice/tt/ttnn_vibevoice_generator.py
models/experimental/vibevoice/tt/ttnn_vibevoice_lm.py
models/experimental/vibevoice/tt/ttnn_diffusion_head.py
models/experimental/vibevoice/tt/ttnn_dpm_scheduler.py
models/experimental/vibevoice/tt/ttnn_acoustic_tokenizer.py
models/experimental/vibevoice/tt/ttnn_semantic_tokenizer.py
models/experimental/vibevoice/tt/ttnn_speech_connector.py
models/experimental/vibevoice/tt/ttnn_vibevoice_model.py
```
## Component PCC tests (first cheap correctness screen — run before any long render)
```
tests/pcc/test_lm_decode_pcc.py   test_decoder_layer_pcc.py   test_diffusion_head_pcc.py
tests/pcc/test_acoustic_tokenizer_pcc.py   test_semantic_tokenizer_pcc.py   test_connector_pcc.py
tests/pcc/test_e2e_generate_pcc.py   # e2e frame-0 / forced-token gate when changing cross-module behavior
```

## Op-level perf harness (warm eager speech frame; Tracy start/stop around frame N)
```
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
export VV_TRACE_SEGMENT=0            # op-level report path (fused trace is for wall/e2e, NOT this op stream)
export VV_PROFILE_SPEECH_FRAME=2 VV_PROFILE_SPEECH_FRAME_EXIT=1
bash models/experimental/vibevoice/tests/perf/run_speech_frame_tracy.sh
# or manually:
python -m tracy -v -r -p --op-support-count 100000 \
  models/experimental/vibevoice/demo_ttnn.py --demo 1p_CH2EN --max_new_tokens 32
CSV=$(ls -td generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report "$CSV" --start-signpost start --end-signpost stop \
  > models/experimental/vibevoice/lm/speech_frame_expN.txt
```
Baseline: `models/experimental/vibevoice/lm/speech_frame_exp0.txt` (+ `_analysis.txt`) — **3123 device
ops, ~54.6 ms device, ~205 ms op-to-op gaps.** Segment the chronological op table into phases
(neg-LM / diffusion / post / pos-LM, split on the two lm_head matmuls + the conv block) and attack the
biggest **phase** — the top-line "62% Matmul" hides that **diffusion, not the LM, is the biggest phase.**

## Edit BOTH frame paths
The op-level profile runs the **eager** path (`VV_TRACE_SEGMENT=0`: `_neg_lm_step` /
`_run_speech_diffusion`); the **deployed** decode is the **fused split-capture trace**
(`VV_TRACE_SEGMENT=1`: `_run_segment_frame_traced` → `_negtrace` / `_dptrace` / `_postrace`). A change
in only the eager path is invisible to deploy; one in only the trace path is invisible to the profile.
Apply to both and confirm (op count / marker op appearing or vanishing). The long-form gate always runs
the **trace** path (`--trace`).

## Operational hazards (these cost real time — do not skip)
- **Check `df -h /` before any long render.** The root disk fills from accumulated `~/vibe-voice/vv_*`
  render dirs (250 MB–1.8 GB each) and a ~250 MB golden-wav COPY the demo makes at startup. A near-full
  disk crashes a render mid-way with a **misleading deep traceback that looks like a code bug but is
  ENOSPC.** Keep > ~2 GB free; delete old scratch renders but KEEP the baseline references
  (`vv_lb_FULL`, `vv_split_FULL`, `vv_122f_eager_90`).
- After killing a render, **reset the device**: `/home/ubuntu/.local/bin/tt-smi -r 0` (orphaned
  process locks → firmware-init crashes / host hangs).
- Traced decode is **deterministic** → a single run is decisive (no averaging).
- Fidelity/precision is **rarely the lever** for this DM-bound frame and it costs long-form margin —
  leave conv/matmul/SDPA fidelity high unless the profile is genuinely math-bound.

## Deliverables
- **One commit per adopted optimization**, message stating device-µs before→after AND the safety tier
  it passed (Tier-0 byte-identical, or Tier-2 full-render-clean with the oracle + Whisper evidence).
- A table: phase/op → device-time before→after → opt class (preserving/changing) → gate passed.
- For every rejected opt, a one-line reason (which tier it failed). A fast frame that corrupts the
  90-min render is not shippable — say so and move on.

# VibeVoice-1.5B speech-frame decode — baseline perf report + optimization plan

Blackhole P150. Branch `ign/vibevoice1.5_fix` @ `f87bcd6bac2` (= clean `vv-122f-fast` @ `0c21de7abeb`
+ prefill opts + emit-only energy limiter). Deploy target: `4p_climate_100min` (~42k-frame,
90–100 min autoregressive render). **Non-negotiable: the full-length render must stay clean.**

## Frame graph (deployed = split-capture trace, `VV_TRACE_SEGMENT=1`)
```
_negtrace  : neg-LM  = 28-layer Qwen2 decode on embed(speech_diffusion) @ neg_pos, kv_neg
             (need_logits=False → lm_head SKIPPED, bit-exact)            → neg_hidden
_dptrace   : diffusion = DPM loop, 10 steps × B=2 head fwd + CFG combine → latent
             post      = acoustic decode(conv) → semantic encode(conv) → 2 connectors → fused
_postrace  : pos-LM  = 28-layer Qwen2 decode on fused @ pos_pos, kv_pos
             + constrained lm_head (subset of 5 selectable tokens) + in-trace argmax → token
```
Already-committed SAFE opts in the baseline: neg-LM lm_head skip (bit-exact), constrained-decode
subset lm_head + in-trace argmax (bit-exact), fp32-rope host-write, split-capture (bit-exact vs eager).

## Baseline op-level profile — one warm EAGER speech frame
`VV_PROFILE_SPEECH_FRAME=2 VV_PROFILE_SPEECH_FRAME_EXIT=1 VV_TRACE_SEGMENT=0`, demo `1p_CH2EN`,
signpost-bounded frame 2. **Total: 72,928 µs device, 5,021 ops.**

| Op (grouped) | Device µs | % | Count | Bound / note |
|---|---|---|---|---|
| MatmulDeviceOperation | 44,106 | 60.5 | 722 | weight-DRAM-bound at M=1 (FLOPs ~10%, DRAM ~77%) |
| LayerNorm (rms_norm) | 5,782 | 7.9 | 270 | 2/layer × 28 × (neg+pos) + diffusion + conns; ~21 µs each (latency-bound) |
| Conv2d | 5,288 | 7.3 | 88 | acoustic decode + semantic encode |
| **ArgMax** | 4,625 | 6.3 | **1** | full-vocab 151936 — **EAGER-ONLY; deploy uses constrained subset (~µs)** |
| BinaryNg (eltwise) | 3,383 | 4.6 | 1,296 | adds/muls/rope |
| Unary (silu…) | 1,445 | 2.0 | 390 | |
| Tilize/Reshape/SDPA/Untilize/Slice/Concat/… | ~7,300 | ~10 | ~2,264 | glue |

Category stack: **Compute 82.3% (60.0 ms) · Other 9.4% (6.8 ms) · TM 6.8% (5.0 ms) · DM 1.5% (1.1 ms).**

Notable matmul instances: lm_head `32×1536×151936` = 1,203 µs (eager-only); FFN gate/up `32×1536×8960`
= 72 µs @ 76% DRAM (healthy); **FFN down `32×8960×1536` flagged SLOW @ 199 GB/s / 38.9% DRAM** (auto
config underperforming); qkv/o `32×1536×1536` = 14 µs.

### Eager-vs-deploy caveat (critical for reading the numbers)
The eager profile includes ops **absent from the deployed trace**: the full-vocab ArgMax (4.6 ms) and
the neg+pos full lm_head (~2.4 ms) — both replaced by the constrained-decode subset in deploy. So
**~7 ms of the 72.9 ms eager frame is not in deploy.** The eager profile is the accepted per-op device
map (both paths share the underlying ops), but absolute frame time comes from the traced steady-state
replay (measured separately). Deploy-relevant device time is dominated by the **two 28-layer LM
forwards + the 10-step B=2 diffusion**.

## Optimization plan (ranked; every change classified BEFORE adoption)

Gate policy (from the runbook): math-PRESERVING (byte-identical) ⇒ Tier-0 byte-compare only ⇒ SAFE.
math-CHANGING ⇒ Tier-2 full 100-min render (oracle + Whisper + bytecompare), individually AND on the
cumulative stack. Program-config / fidelity / fusion / precision = math-CHANGING until proven
byte-identical (`maxabsdiff==0`) — long-form loop is bit-chaotic.

### A. Diffusion loop redundant-compute dedup — **Tier-0 (byte-identical), adopt first**
In `sample_speech_latents` the CFG condition is **step-invariant** (only the noisy latent + timestep
change across the 10 steps), yet per step the head recomputes:
- `cond_combined = concat([neg,pos])` (10×/frame → 1×) and `cond_proj = linear(cond_combined, W)`
  (a `2×1536×1536` matmul, 10×/frame → 1×).
- `silu(c)` inside every HeadLayer + FinalLayer (5×/step on the *same* c → 1×/step).
Hoisting these is the *same op on the same inputs* ⇒ bit-identical. Est ~350 µs/frame (~0.5–0.8% of
traced). Must edit BOTH the eager head path and the traced `_dptrace` head path. Gate: Tier-0
byte-compare (cap-400 + one cap-1600).

### B. CFG batch-2 LM fusion — **flagship lever; byte-identity TBD → Tier-0 or Tier-2**
Today the frame runs **two separate B=1 28-layer forwards** (neg-LM, pos-LM). Decode matmuls are
weight-DRAM-bound at M=1, so a single **B=2** forward reads each layer's weights **once** for both
rows → saves ≈ one LM forward's weight-read (~9 ms; ~20% of the traced frame). Requires software
pipelining (batch pos-LM(k−1) with neg-LM(k), both feeding diffusion(k)), a batched KV cache `[2,…]`
with per-row positions, `sdpa_decode(cur_pos_tensor=[p_pos,p_neg])`, and per-row `paged_update_cache`.
- Mechanisms proven bit-exact in isolation (`tests/perf/cfg_batch2_probe.py`: per-batch KV write, per-
  batch SDPA position). Est LM win ~1.77×/−43% (memory).
- **Risk:** the production decode progcfg (`_QO_DECODE_PROGCFG`, `per_core_M=1`) is B=1-only and
  overflows at B=2; a B=2 progcfg may change matmul K-reduction order ⇒ NOT byte-identical ⇒ the loop
  can destabilize over 40k steps. **Verify byte-identity in isolation first** (B=2 fwd vs 2×B=1,
  `maxabsdiff` on the 28-layer output with matched `in0_block_w`). If `maxabsdiff==0` ⇒ Tier-0. Else ⇒
  Tier-2 full-render gate (and, given history that math-changing stacks collapsed the render, reject if
  it fails Tier-2).
- Effort: large; touches the trace path (merge `_negtrace`/`_postrace`), KV alloc, loop restructure,
  and free-running EOS-latency (+1 frame). Do AFTER A proves the gate pipeline.

### C. FFN down-proj `32×8960×1536` is SLOW (39% DRAM) — **candidate, needs byte-identity proof**
The auto config underperforms (199 vs ~390 GB/s on gate/up). A better `MultiCast1D` progcfg that keeps
`in0_block_w` (⇒ same reduction order) *may* be byte-identical. Only adopt if `maxabsdiff==0`; else it
is math-changing (this exact class — "post FFN down-proj progcfg" — was in the reverted 41.1 ms stack).

## Explicit DEAD-ENDS to avoid (prior campaign, whisper/oracle-verified)
- Diffusion-head program-configs (`per_core_M`/B tuning) — **collapsed the 90-min render** (clip min12,
  collapse min67). The `vv-122f-fast` baseline exists specifically to avoid these.
- `bf8_b` weights / HiFi2 / LoFi anywhere in the frame — math-changing, collapsed collectively.
- Post FFN down-proj progcfg + pos lm_head progcfg — math-changing (reverted 41.1 ms stack).
- Depthwise-conv shift-MAC — device-slower. Fidelity walks — frame is DM/dispatch-bound, not math-bound;
  costs long-form margin.

## Progress log (2026-07-20)

### Opt A — diffusion cond_proj/cond_combined/silu(c) hoist — ADOPTED (pending commit on fixed base)
- Impl: `ttnn_diffusion_head.py` (`project_condition`/`forward_pre_cond`, shared `silu(c)`), `ttnn_dpm_scheduler.py`
  (hoist `cond_combined`+`cond_proj` out of the DPM loop). One edit covers eager + traced (both use `sample_speech_latents`).
- Device: warm frame **72,928 → 72,285 µs (−643 µs), 5,021 → 4,963 ops (−58)** — exactly 9 `cond_proj` matmuls
  + 9 `cond_combined` concats + 40 redundant `silu(c)`.
- Gate: diffusion PCC PASS; **Tier-0 byte-identical to clean-HEAD** at cap-400 AND cap-1600 (maxabsdiff=0.0),
  cap-1600 completed with **4 segment-boundary recaptures, no crash**. Math-preserving → long-form-safe.

### ⚠ CRITICAL FINDING — HEAD long-form regression (pre-existing, NOT opt A)
Discovered during opt-A's full-length verification. The working branch **HEAD (`f87bcd6bac2`) traced
`4p_climate_100min` collapses to gibberish by ~min 55** — a regression vs the official baseline.
- Energy oracle (full length): clean-HEAD RMS drifts 0.09→0.38, sustained peak-1.0 clipping from min 63;
  official baseline stays flat RMS ~0.09 to min 77 (only the known min-78–83 residual spike, then recovers).
- Whisper: HEAD coherent at min 40, degenerate at min 55/68/85 ("you"/"We are."/"Okay."); official baseline
  coherent at min 40 AND 68, residual at min 80 ("So"), **recovers coherent by min 90**.
- The official byte-baseline `stream_loopbreak.f32` (@ `0c21de7abeb`) diverges from HEAD at frame 0 (maxabsdiff
  0.94) → HEAD's prefill drift shifted the whole trajectory. Prime suspect: **prefill RoPE fuse `d0d80b478bc`**
  (committed after the baseline, only PCC 0.999999 — not bit-exact). Content collapse ⇒ emit-only limiter can't fix.
- Action (user-approved): **reverted `d0d80b478bc`** (prefill RoPE back to manual fp32; prefill/TTFT-only, no
  steady-frame cost) and re-rendering full 100-min to confirm baseline-level cleanliness returns. Opt A re-applies
  on top (byte-identical). All subsequent Tier-2 gating uses the fresh HEAD(−fuse) full render as the re-baseline.

## Deliverable accounting
One commit per adopted opt (msg: device-µs before→after + safety tier). Rejections logged with the tier
they failed. Final: re-run Tier-2 on the cumulative stack.

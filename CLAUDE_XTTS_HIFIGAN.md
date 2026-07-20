# XTTS-v2 HiFi-GAN Vocoder — TTNN bringup (Block 4)

Parent: `CLAUDE_XTTS_TTNN.md` (read it first for shared decisions + integration contract).

## Status / Owner / Started
- Status: **DONE on TT** — full generator ported to TTNN; **waveform PCC 0.9983** vs the
  coqui golden (fp32 activations; standalone + pytest both pass the 0.99 gate).
- Owner: acicovic
- Started: 2026-07-20 · Ported: 2026-07-20

## TTNN result (2026-07-20)
- Files: `tt/ttnn_xtts_hifigan.py` (`TTNNHifiganGenerator` + `preprocess_hifigan_parameters`,
  reusing `load_hifigan_state`); `tests/test_hifigan_pcc.py` (pytest fixture + standalone
  `__main__` busy-retry; per-stage dbg oracles + final-waveform gate).
- **Final waveform PCC = 0.9983** (gate 0.99). Per-stage (fp32) all ≥ 0.9998:
  conv_pre 1.000, ups0 1.000, ups1 0.99999, ups2 0.99987, ups3 0.99996.
- **Layout:** the 1D signal `[1,C,L]` is carried as NHWC `[1,1,L,C]` end-to-end.
  `ttnn.conv1d` for conv_pre/conv_post/resblock convs (weight `[out,in,k]`, incl. dilation).
  **Upsamples = `ttnn.conv_transpose2d`** with a height-1 kernel `(1,k)`, stride `(1,s)`,
  pad `(0,p)` (no conv_transpose1d); weight reshaped to IOHW `[in,out,1,k]`; auto-prepares
  raw torch weights. Per-layer conditioning `conds[i](g)`/`cond_layer(g)` = a 1x1 conv on the
  length-1 d-vector, computed as a `ttnn.linear` -> `[1,1,1,C]` broadcast-add over time.
- **Long sequences:** activations reach `[1,1,101376,32]` (396×256). Convs with input
  length > 8192 use a DRAM width-slice (`Conv2dSliceConfig(Conv2dDRAMSliceWidth,
  num_slices=0)` auto; `dram_slice_config=` for conv_transpose2d, `slice_config=` for conv1d)
  — no OOM.
- **Precision (important):** bf16 activations cap the waveform at **~0.96** — per-stage
  transpose outputs stay ~0.999 in bf16, but the final 32→1 `conv_post` amplifies the bf16
  error accumulated across the 12 ResBlocks on the oscillatory output. **fp32 activations**
  clear the gate (0.9983). fp32's conv config/halo tensor needs a bigger L1_SMALL than bf16,
  so the device is opened with **`l1_small_size=65536`** (bf16's 32768 OOMs in fp32). See
  BUG-3 in `CLAUDE_XTTS_BUGS.md`.
- Compute config: HiFi4 + fp32_dest_acc (as Block 2). Weight-norm fold reused verbatim from
  `load_hifigan_state` (no re-derivation).

## Progress (2026-07-20)
- **Architecture mapped** (coqui `TTS.vocoder.models.hifigan_generator.HifiganGenerator`,
  wrapped by `TTS.tts.layers.xtts.hifigan_decoder.HifiDecoder`). Block boundary for TTNN =
  the generator: input `z [1,1024,L]` + d-vector `g [1,512,1]` → waveform `[1,1,L*256]`.
  (The two linear time-`interpolate`s in `HifiDecoder.forward` that build `z` from the GPT
  latents are cheap host resizes — kept on CPU, like the mel front-ends.)
- **Config** (from checkpoint): conv_pre `Conv1d(1024→512,k7,p3)`; `cond_layer Conv1d(512→512,k1)`;
  **4 upsamples** `ConvTranspose1d` (512→256→128→64→32) k/stride/pad
  `[(16,8,4),(16,8,4),(4,2,1),(4,2,1)]`; per-layer `conds[i] Conv1d(512→ch_i,k1)`; **MRF** =
  mean of 3 `ResBlock1` per upsample (kernels [3,7,11], dilations [1,3,5]); conv_post
  `Conv1d(32→1,k7,p3,no-bias)`; tanh. `cond_in_each_up_layer=True`.
- **Weight-norm:** ups + resblock convs are weight-norm parametrized in the checkpoint
  (`parametrizations.weight.original0/1` = g/v); fold with `torch._weight_norm(v, g, 0)`
  (dim 0). conv_pre/conv_post/cond_layer/conds are plain (weight-norm removed).
- **GOTCHA (cost me a per-stage debug):** the **final** `leaky_relu` before `conv_post` uses
  torch's **default slope 0.01**, NOT `LRELU_SLOPE=0.1` like every other leaky_relu. With
  0.1 the reference was 0.99985 (maxabs 0.03); with 0.01 it's PCC **1.0**.
- **CPU reference** `reference/xtts_hifigan_ref.py` — `HifiganReference`, matches coqui golden
  at **PCC 1.0** (and every stage conv_pre/ups0..3 at 1.0). Goldens in `golden/hifigan/`
  (`z, g, wav`, gitignored). Regen: `$CLAUDE_JOB_DIR/tmp/gen_hifigan_golden.py`.

## TTNN plan
- `ttnn.conv1d` exists (use for conv_pre/conv_post/cond_layer/conds/resblock convs, incl.
  dilation); `ttnn.conv_transpose2d` exists (no 1d) — use for the 4 upsamples with a
  height-1 kernel `(1,k)`. Reuse the Block-2 conv patterns (NHWC, `l1_small_size`, BN-free
  here). Per-layer conditioning = broadcast-add of `conds[i](g)` (a [1,ch,1] vector).
- **Watch memory / long sequences:** activations grow to `[1,32,101376]` at the end
  (396 frames × 256). May need DRAM-sliced conv configs (`Conv2dDRAMSliceWidth`) for the
  last stages.

## Role in pipeline
Final stage. Consumes GPT latents (Block 3) **directly** — **no mel intermediate** — and
the d-vector (Block 2), producing the 24 kHz waveform. The d-vector is injected via linear
projections at each upsampling layer (`g=speaker_embedding`).

## Interface contract (from master)
| Direction | Tensor | Shape | dtype |
|-----------|--------|-------|-------|
| in | `gpt_latents` | (1, T_code, 1024) | f32 (from Block 3) |
| in | `speaker_embedding` (d-vector) | (1, 512, 1) | f32 (from Block 2) |
| out | `waveform` | 24 kHz mono | f32 → output .wav |

## Foundation / template
`speecht5_tts` pattern, but note SpeechT5's HiFi-GAN takes a **mel**; XTTS's takes **GPT
latents** — different input contract. **Per master decision: run on CPU first** (as
`speecht5_tts` runs its vocoder on CPU), port to TTNN **last**.

## Reference source
- coqui `TTS/tts/models/xtts.py` → `self.hifigan_decoder(gpt_latents, g=speaker_embedding)`
  and the coqui HiFi-GAN decoder module (with per-upsample-layer speaker conditioning).

## Build steps
1. Mirror the vocoder in `reference/hifigan.py`; this doubles as the CPU runtime impl.
   PCC=1.0 vs coqui on golden `gpt_latents` + `speaker_embedding`.
2. Wire it CPU-side into `tt/ttnn_xtts_model.py` so the e2e pipeline produces audio early.
3. (Later) port convs / transpose-convs / group_norm to TTNN.

## PCC validation plan
Golden inputs = reference `gpt_latents` + `speaker_embedding`. Compare output waveform
(PCC ≈ 1.0 vs coqui). Test under `tests/`.

## Findings log (dated)
- 2026-07-20 — TTNN port complete, waveform PCC 0.9983. Debug notes worth keeping:
  - Both conv primitives validated in isolation at PCC ~0.9999 (conv1d, height-1
    conv_transpose2d). `ttnn.conv1d` returns the output length as a **scalar int** (not a
    list); `conv_transpose2d` returns `[out, [ho, wo]]`.
  - **Per-stage oracle gotcha:** a device intermediate stashed for later inspection reads as
    garbage after the forward (its DRAM buffer is reused). The *forward path is unaffected* —
    it just means the PCC harness must snapshot each stage to host (`ttnn.to_torch`) at
    capture time. Chasing this cost time (looked like the transpose was broken at 0.27 when it
    was actually 0.9999). The dbg_ups{i} oracles capture the **transpose output** (pre-cond,
    pre-MRF); dbg_conv_pre is conv_pre **only** (pre-cond).
  - bf16 vs fp32: see "Precision" above — bf16 ~0.96, fp32 0.9983.

## Open questions / TODO
- [x] conv1d / transpose-conv tile-alignment — done (NHWC height-1 idiom; DRAM width-slice
      for long sequences). No group_norm in this block.
- [x] Upsample factors/kernels + d-vector projection points — confirmed against checkpoint.
- [x] Output 24 kHz mono — `[1,1,L*256]`, matches coqui golden.
- [ ] Integration: wire `TTNNHifiganGenerator` into `tt/ttnn_xtts_model.py` (feed Block-3
      `gpt_latents` → z via the two host interpolates, Block-2 d-vector → g).

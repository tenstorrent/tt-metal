# Kokoro-82M — Model Architecture & Text-Encoder Optimizations

**TTNN / Tenstorrent Blackhole Port · Technical Documentation**
`models/experimental/kokoro` · `tt-metal`

---

## 1. Overview

Kokoro-82M is a compact (~82 M parameter) neural text-to-speech (TTS) model in the StyleTTS2 / ISTFTNet family. It converts a sequence of phoneme tokens plus a reference speaker-style vector into a 24 kHz speech waveform. The pipeline has two conceptual halves: a **prosody / linguistic front-end** (a PL-BERT language encoder, a prosody predictor that estimates per-phoneme duration, pitch F0 and energy N, and a text/ASR encoder) and a **neural vocoder back-end** (a StyleTTS2 decoder wrapping an ISTFTNet generator with a harmonic-plus-noise source and an inverse-STFT reconstruction).

This document describes (1) the full model architecture as implemented in the repo-owned reference and its TTNN device port, (2) the completed performance optimizations for the Text Encoder module on Tenstorrent Blackhole (BH) hardware, and (3) the BF16-accumulation hardware-limit proof for the SineGen phase chain.

**Source layout** — the PyTorch reference lives under `models/experimental/kokoro/reference/` (`model.py`, `modules.py`, `istftnet.py`, `custom_stft.py`) and the device port under `models/experimental/kokoro/tt/` (`tt_kmodel.py`, `tt_text_encoder.py`, `tt_prosody_predictor.py`, `tt_decoder.py`, `tt_generator.py`, `tt_sinegen.py`, `tt_lstm.py`, `tt_conv.py`, and STFT ports).

---

## 2. Model Architecture

### 2.1 End-to-end pipeline

The top-level model is `KModel` (`reference/model.py`). Its `forward_with_tokens(input_ids, ref_s, speed)` runs the following stages, where `ref_s` is a 256-d reference-style vector split into a 128-d prosody style `s` and a 128-d acoustic/decoder style:

| # | Stage | Module | Function |
|---|-------|--------|----------|
| 1 | Language encoding | `CustomAlbert` (PL-BERT) | Contextual embedding of phoneme ids → `bert_dur` |
| 2 | BERT projection | `bert_encoder` (Linear) | Project BERT hidden → hidden_dim, transpose → `d_en` |
| 3 | Duration encoding | `predictor.text_encoder` (DurationEncoder) | Style-conditioned BiLSTM stack → `d` |
| 4 | Duration LSTM + proj | `predictor.lstm` + `duration_proj` | Per-phoneme duration logits → `pred_dur` |
| 5 | Alignment expansion | `repeat_interleave` → `pred_aln_trg` | Phoneme→frame alignment matrix |
| 6 | Pitch / energy | `predictor.F0Ntrain` (shared BiLSTM + AdainResBlk) | `F0_pred` (pitch), `N_pred` (energy) |
| 7 | Text/ASR encoding | `text_encoder` (TextEncoder) | Acoustic phoneme features `t_en` |
| 8 | Frame alignment | `t_en @ pred_aln_trg` | Expand text features to frame rate → `asr` |
| 9 | Vocoding | `decoder` (Decoder + Generator) | `asr` + F0 + N + style → 24 kHz waveform |

The prosody front-end (stages 1–6) runs entirely on device at PCC > 0.998. The alignment (stage 5) uses small CPU integer tensors for the discrete duration indices, exactly as the PyTorch reference predictor does.

### 2.2 PL-BERT language encoder

`CustomAlbert` is an ALBERT-style transformer (`transformers.AlbertConfig`) that contextualizes the phoneme id sequence. It uses ALBERT cross-layer weight sharing — a single parameter group reused across all 12 hidden layers (hidden size 768, 12 attention heads, intermediate 2048, max position 512), which keeps the device footprint small. Each layer is standard multi-head self-attention + GELU feed-forward with residual LayerNorm. The final hidden state feeds a single `bert_encoder` linear that projects 768 → hidden_dim (512) and transposes to channel-first `[B, C, T]` for the downstream convolutional / recurrent stack. Device port: `tt_custom_albert.py` (fused QKV, HiFi3 attention matmuls).

### 2.3 Prosody predictor

`ProsodyPredictor` (`reference/modules.py`; port `tt_prosody_predictor.py`) predicts the supra-segmental features — how long each phoneme lasts, its pitch, and its energy. It contains:

- **DurationEncoder** — a stack of `nlayers` bidirectional LSTMs interleaved with AdaLayerNorm (style-adaptive LayerNorm). Each layer concatenates the 128-d style vector to the features, runs a BiLSTM (`d_model → d_model/2`, bidirectional), then applies style-conditioned normalization. Output `d` is the duration-conditioned representation.
- **Duration LSTM + `duration_proj`** — a further BiLSTM feeds `LinearNorm(d_hid → max_dur=50)`; the argmax/rounded sum over the duration logits gives `pred_dur`, the integer frame count per phoneme used to build the alignment matrix.
- **F0Ntrain (pitch & energy)** — a shared BiLSTM branches into two `AdainResBlk1d` stacks (the F0 and N branches, each 3 blocks with one upsampling block), ending in `F0_proj` / `N_proj` 1×1 convs that emit the per-frame pitch contour `F0_pred` and energy `N_pred`.

The prosody / F0 path is numerically sensitive: on Blackhole the BF16 MAC rounding is harmless here (PCC > 0.998) but any change to accumulation order can flip discrete duration predictions, so device optimizations that touch it require full-model verification.

### 2.4 Text Encoder (ASR encoder)

`TextEncoder` (`reference/modules.py:34`; port `tt_text_encoder.py`) produces the acoustic phoneme representation that, after frame alignment, drives the vocoder. It is a three-part stack — an embedding, a CNN, and a BiLSTM:

| Component | Configuration | Shape |
|-----------|---------------|-------|
| Embedding | `nn.Embedding(n_symbols=178, channels=512)` | `[B,T]` ids → `[B,T,512]` |
| CNN stack | depth=3 × (weight-norm Conv1d k=5, pad=2 → channel LayerNorm → LeakyReLU 0.2) | `[B,512,T]` → `[B,512,T]` |
| BiLSTM | `nn.LSTM(512, 256, num_layers=1, bidirectional)` | `[B,T,512]` → `[B,T,512]` |
| Output | transpose to channel-first | `[B, 512, T]` |

Because the CNN and embedding are per-(position, channel) operations, the device port folds the batch into the length dimension and runs the whole stack in a flattened rank-4 `[1,1,B·T,C]` layout, un-flattening only once before the per-batch BiLSTM recurrence. Section 3 details the optimizations applied to this module.

### 2.5 Decoder & ISTFTNet generator (vocoder)

`Decoder` (`reference/istftnet.py:470`; port `tt_decoder.py`) fuses the aligned text features `asr` with the pitch/energy curves and the acoustic style, then hands off to the ISTFTNet `Generator`:

- **Decoder body** — `F0_conv` / `N_conv` (stride-2 convs) downsample the pitch/energy curves; these are concatenated with `asr` and passed through an encode `AdainResBlk1d` (→1024 ch) and four decode `AdainResBlk1d` blocks (style-adaptive residual blocks, the last upsampling to 512 ch), with an `asr_res` skip (1×1 conv → 64 ch).
- **Harmonic-plus-noise source (`SourceModuleHnNSF` / `SineGen`)** — the F0 contour is upsampled (`upsample_scale = prod(upsample_rates) × hop`) and turned into a harmonic sine excitation (`harmonic_num=8`) plus filtered noise. This is the neural-source-filter excitation that gives the vocoder its pitch. Port: `tt_sinegen.py`, `tt_source_module_hn_nsf.py`.
- **STFT front / iSTFT back** — the harmonic source is analysed by an STFT (`gen_istft_n_fft`, `gen_istft_hop_size`); the generator runs `ConvTranspose1d` upsampling stages interleaved with `AdaINResBlock1` residual blocks and noise-conditioning convs, then `conv_post` emits a magnitude (`exp`) and phase (`sin`) spectrogram that an inverse STFT converts back to the 24 kHz waveform. Two STFT implementations exist — `TorchSTFT` (complex) and `CustomSTFT` (`disable_complex`); ports are `tt_torch_stft.py` and `tt_custom_stft.py`.

### 2.6 On-device precision & CPU fallbacks

Blackhole rounds float32 → bfloat16 inside MAC units. This is harmless across the entire prosody front-end (PL-BERT → predictor → TextEncoder, PCC > 0.998) but degrades two spots on the vocoder harmonic-source path, so two optional CPU fallbacks are provided:

| Failure point | Mechanism | Fallback |
|---------------|-----------|----------|
| SineGen phase chain | Tiny cumsum × 2π×upsample(≈1885) amplifies BF16 rounding into ~0.06–0.25 rad/frame phase error; `sin()` worsens it | `use_torch_phase_fallback` (phase accumulation on CPU fp32) |
| STFT magnitude/phase | Near-zero off-frequency bins get sign-flipped by BF16 `atan2`; harms `cos(phase)` | `use_torch_stft_fallback` (`torch.stft` on CPU) |

Note: With phase fallback alone we can obtain good quality audio pcc will degrade by ~0.03

The SineGen phase-chain failure is characterized in detail in [Section 4](#4-bf16-accumulation-hardware-limit-proof-sinegen), which proves it is an intrinsic property of the BF16 numeric format and not a TTNN kernel defect.

### 2.7 Key hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sampling rate | 24 000 Hz | Output waveform |
| Phoneme vocab (`n_symbols`) | 178 | TextEncoder embedding rows |
| Model channels (TextEncoder) | 512 | embedding / conv width |
| TextEncoder CNN | depth 3, kernel 5, pad 2 | weight-norm Conv1d + LayerNorm + LeakyReLU(0.2) |
| TextEncoder BiLSTM | hidden 256, 1 layer, bidir | output 2×256 = 512 |
| Prosody style dim | 128 | `s` (prosody) + 128 (acoustic) = 256 `ref_s` |
| PL-BERT | 768 hidden, 12 layers/heads, shared | ALBERT weight sharing, intermediate 2048 |
| Generator `upsample_rates` | [10, 6] | total 60×; kernels [20, 12] |
| Generator resblocks | kernels [3,7,11], dil [1,3,5] | AdaINResBlock1, initial channel 512 |
| iSTFT n_fft / hop | 20 / 5 | `gen_istft_n_fft`, `gen_istft_hop_size` |
| SineGen harmonics | 8 (sine_amp 0.1, voiced thr 10) | harmonic-plus-noise source |
| Total parameters | ~82 M | Kokoro-82M |

---

## 3. Completed Text-Encoder Optimizations

The TTNN TextEncoder is latency / dispatch-bound: the device-time sum is a small fraction of wall-clock, which is dominated by per-op host-dispatch gaps (the LSTM loop alone is ~75% of the work — matmul + BinaryNg + slice). The optimizations below cut both the device-time sum (1932 → ~1090 µs, roughly −43%) and the host op count while holding PCC essentially unchanged (0.999954 → ~0.99930, all far above bar). Only device tracing removes the remaining host dispatch gaps in wall-clock; these wins are the prerequisite that makes a traced deployment fast.

Files touched: `tt/tt_text_encoder.py`, `tt/tt_lstm.py`, `tt/tt_conv.py`. Each optimization was validated by a dedicated perf sweep under `perf/` and by the three TextEncoder PCC tests.

### 3.1 Summary of optimizations

| # | Optimization | Effect |
|---|--------------|--------|
| 1 | Blackhole batched conv | Single batched conv1d instead of per-item split (a WH-only workaround); Conv2d 229→97 µs, halves halo/reshard/concat |
| 2 | No-pad mask-skip | When there is no padding the 5 keep-mask multiplies are identity no-ops → skipped on the host (−5 BinaryNg) |
| 3 | Rank-4 flattened CNN | Embed as `[1,1,B·T,C]` and chain conv→LN→act with no per-stage reshape; one un-flatten before LSTM (−reshape/tilize dispatches) |
| 4 | Tuned recurrent-matmul config | 1D-mcast 8×8 grid, `in0_block_w=8`, width-sharded out for `[32,512]@[512,2048]`: 3.73 µs vs 10 µs default (−32% vs old ibw=4) |
| 5 | L1-interleaved per-step tensors | State/gates/slices on L1 (no reshard vs width-shard); BinaryNg −9%, Slice −9%, Unary −15% |
| 6 | All fused-loop matmuls + gx buffer to L1 | Gate-projection buffer + transient matmul weights staged to L1; −149 µs |
| 7 | Remaining matmuls fully L1 | `x_nlc` + output-assembly tensors to L1 so all 52 matmuls read L1 in0 |
| 8 | B=1 bias-fold + L1 compose | Fold per-step `gates_x` add into the recurrent matmul bias epilogue (ttnn limitation re-probed and lifted); −1 BinaryNg/step |
| 9 | Block-sharded CNN LayerNorm | LN program config derived from the conv's own block-sharded L1 output → normalize+activate in place; LN 37→23 µs, reshards 12→2 |
| 10 | Conv1d double-buffer + explicit block shard | Weights + activation double-buffer, explicit BLOCK shard, `act_block_h=32`: 32.3→21.7 µs/conv (1.46×) |
| 11 | Embedding weight stored ROW_MAJOR | Moves the per-forward UntilizeWithUnpadding to load time; −17 µs, −1 op |
| 12 | BiLSTM weights pre-staged to L1 | Constant matmul weights uploaded L1-resident once at preprocess (per-forward copies 4→1) |
| 13 | CNN conv at HiFi2/LoFi | Conv inputs are bf16 so precision saturates early; LoFi is fastest PCC-passing config, conv 65→51 µs |
| 14 | Fused cell-state addcmul | `c = f·c + tanh(g)·i` via one `ttnn.addcmul` (a genuine fused kernel, unlike `ttnn.mac`); −1 device op/step |

### 3.2 CNN-stack optimizations

**Blackhole batched conv (#1).** `tt_conv1d_nlc`'s B>1 per-item split was a Wormhole-B0 correctness workaround; on BH a single batched `ttnn.conv1d(batch_size=B)` is correct (verified PCC ≈ 1.0). This auto-block-shards and halves halo / interleave-to-shard / shard-to-interleave / concat dispatches. Conv2d dropped 229 → 97 µs.

**Rank-4 flattened CNN (#3).** A TTNN ReshapeView is a fixed ~5 µs dispatch, not data-proportional, so the win is fewer reshape *ops*. Feeding ids as `[1,1,B·T]` lands the embedding at `[1,1,B·T,C]`; the batched conv returns its native rank-4 with no reshape, so the entire CNN chains conv→LN→leaky flat and un-flattens to `[B,T,C]` exactly once, before the LSTM.

**Conv1d double-buffer + block shard (#10).** A 96-case sweep (`perf/test_conv_text_encoder_perf_sweep.py`) found the biggest lever was weights double-buffering (~10 µs — the conv re-reads its `[2560,512]` weight per output block), then activation double-buffer (~4 µs), then an explicit BLOCK shard (2.5× vs auto). Winner: block + `act_block_h_override=32` + both double-buffers = 21.7 µs (1.46×), PCC 1.0. `act_block_h=64` regresses hard.

**CNN conv at LoFi (#13).** Conv inputs are bf16, so two MAC passes already saturate precision — isolated-conv PCC is 0.99988 (LoFi) vs 0.99998 (HiFi2/3/4). LoFi is the fastest PCC-passing config (`fp32_dest_acc` kept True). Kept separate from the BiLSTM matmul config (which stays HiFi3, since its output feeds the decoder).

**Block-sharded LayerNorm consumed in place (#9).** The channel LayerNorm ran DRAM-interleaved on only 3 cores (~12.4 µs × 3). The batched conv already emits a block-sharded L1 output, so the LN program config is *derived from the conv output's own shard spec* (`_layernorm_prog_from_sharded`) and normalizes + activates on-core. This drops both reshards a fixed-config LN would need. Chaining the block-sharded output straight into the next stage's conv keeps the whole CNN in L1: reshards fell 12 → 2, LayerNorm 37 → 23 µs.

### 3.3 BiLSTM-loop optimizations

The host-driven LSTM loop dominates the module. The per-step recurrent matmul `[B,2H]@[2H,8H]` (32×512×2048) runs at only ~5% of FLOP peak — it is fixed launch overhead, so fidelity and weight placement don't move it; the levers are tiling schedule, op fusion, and L1 residency.

**Tuned recurrent-matmul config (#4).** A dedicated sweep (`perf/test_recurrent_matmul_sweep.py`) found a 1D-mcast config (8×8 grid = one 8H tile/core, `in0_block_w=8` = two K-steps, `per_core_M=per_core_N=1`, width-sharded output) runs the step at 3.73 µs vs 10 µs for the default and 5.52 µs for the old `in0_block_w=4`. The matmul is bit-exact across layouts. The width-sharded output is absorbed by the immediate consumer (gate-add / sigmoid) so no extra per-step reshard is added.

**L1 residency (#5, #6, #7, #12).** Per-step state / gates / slices moved to L1-interleaved (spreads across banks, needs no reshard — unlike width sharding). The gate-projection buffer and the constant matmul weights are staged to L1; storing the weights L1-resident once at preprocess removes 3 per-forward copies. Together these removed ~150 µs and reclassified all 52 matmuls to L1 in0.

**Gate-precompute matmul tuned (#4 cont.).** The `[B,L,in]@[in,4H]` gate precompute was swept (`perf/test_gatex_matmul_sweep.py`): a 2D config with one-M-tile-per-row spread (`per_core_M=1`), wide `out_subblock_w=4`, `in0_block_w=8` cut it 17.3 → 8.0 µs (−54%). The config is derived at forward time from the sequence length. Lesson recorded: always sweep `out_subblock` and per-core-M spread — a coarse grid×ibw sweep alone left ~30% on the table.

**Op fusion (#8, #14).** The per-step cell update `c = f·c + tanh(g)·i` fuses the `mul(f,c)+add` into one `ttnn.addcmul` (a genuine `TernaryDeviceOperation` — critically, `ttnn.mac` is a composite that lowers to two BinaryNg and does *not* help). For B=1, the per-step `gates_x` add folds into the recurrent matmul's bias epilogue (a previously-FATAL ttnn limitation re-probed and found lifted). Each removes one op per step. Both are TextEncoder-only — the shared F0-feeding prosody/duration BiLSTMs keep the unfused path.

**Precision (#4, #13).** BiLSTM matmuls run at LoFi + bf16 dest-acc (the tolerant ASR encoder absorbs it, full-seq PCC unchanged to 4 decimals); the F0-sensitive prosody/duration BiLSTMs keep their own fp32-acc config, so this never touches the F0 path.

### 3.4 Cumulative results

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Device-time sum (B=2) | 1932 µs | ~1090 µs  (−43%) |
| Device op count | 651 ops | ~628 ops |
| Conv2d (per conv) | 32.3 µs | 16–21.7 µs |
| LayerNorm (×3) | 37 µs | 23 µs |
| Recurrent matmul (per step) | ~11 µs (5.52 at ibw=4) | 3.73 µs |
| Gate-precompute matmul | 17.3 µs | 8.0 µs |
| CNN reshards | 12 | 2 |
| Full-sequence PCC | 0.999954 | ~0.99930  (≫ bar) |

**Bottleneck & next lever.** The residual cost is the LSTM loop's host-dispatch gaps (BinaryNg, Slice, and matmul launch overhead), which are invisible on the device-time metric but dominate wall-clock. The slice floor (5 slices/step) and the cell-math op count are already minimal. Device tracing — which eliminates the per-op host gaps — is the only remaining wall-clock lever; it is worthwhile only for multi-call / pipeline use because it needs warmup + capture + replay. Separately, the full-model duration BiLSTM in the prosody predictor still runs the default (untuned) matmul config, and is a candidate pending full-model PCC verification given its F0 sensitivity.

---

## 4. BF16-Accumulation Hardware-Limit Proof (SineGen)

Test: `tests/test_bf16_accumulation_hardware_limit_proof.py`

### 4.1 The question

Section 2.6 noted that the on-device (no-fallback) SineGen phase chain collapses the harmonic-source sine PCC (~0.31 vs the float32 reference) at Kokoro scale (`upsample_scale=300`, `T=48600`). The engineering question this test settles is *where that error comes from*:

- **(A)** a defect in a TTNN kernel (matmul / cumsum-add / lerp / `sin`), or
- **(B)** the intrinsic precision of BF16 arithmetic, which Blackhole uses for its MAC datapath (it rounds float32 → BF16 for the multiplicands) regardless of `fp32_dest_acc_en`.

If it is (A) we should chase a kernel fix; if it is (B) no on-device kernel change can recover it and the CPU `use_torch_phase_fallback` is the correct remedy.

### 4.2 Method — run the *same* math three ways

The phase chain is `downsample → cumsum → lerp-upsample → ×2π×upsample_scale → sin`. The test runs that identical computation three ways on the same F0 and compares each stage's PCC against the float32 golden:

1. **fp32 CPU** — torch, `float32`. The golden reference.
2. **bf16 CPU** — torch, *the identical function*, only `dtype=bfloat16`. Touches **zero** TTNN kernels; runs entirely on the host.
3. **ttnn device** — the real device path (`_run_tt_sinegen_stages`, HiFi3, `fp32_dest_acc_en=True`).

Plus one control that isolates *where* the BF16 hurts:

4. **bf16-input-only** — BF16-quantize the input `rad`, then accumulate the phase in fp32. This separates BF16 *input rounding* from BF16 *accumulation*.

The logic: if (B) is the cause, then a host-only BF16 run must reproduce the device collapse, flipping *only the dtype argument* fp32→bf16 in one CPU function must be what breaks it, and the device's absolute accumulated phase error must match the bf16-CPU sim's error magnitude (i.e. the device is operating at BF16 precision — not better, not worse).

### 4.3 Results

`sin(phase)` PCC vs the fp32 golden — the same chain, three backends (plus the control):

| Backend | Precision | `sin` PCC | Reading |
|---------|-----------|-----------|---------|
| fp32 CPU | identical fn, `float32` | **1.000000** | exact |
| bf16 CPU | identical fn, `bfloat16` | **0.274700** | collapses — **no TTNN** |
| ttnn device | real kernels (HiFi3, fp32 dest-acc) | **0.306700** | collapses |
| bf16-input-only | bf16 input, fp32 accumulation | **0.988551** | survives |

Accumulated phase error (`|Δphase|` wrapped mod 2π) — the quantity that actually scrambles `sin`:

| Source | Mean phase error |
|--------|------------------|
| bf16 CPU sim | 1.2284 rad |
| ttnn device | 1.1667 rad |
| BF16 quant step @ mean\|phase\|≈600 rad | ≈ 2.344 rad |

### 4.4 What is happening

- **A pure-CPU BF16 run reproduces the device collapse** (0.275 ≈ device 0.307) while touching **no TTNN kernel**. A kernel bug could not be reproduced by a host-only NumPy/torch simulation — so the collapse is not a kernel defect.
- **The identical function in fp32 is exact (1.0).** The *only* variable changed between the exact result and the collapse is the numeric format (`float32` → `bfloat16`). That isolates the format as the sole cause.
- **The device accumulates the same magnitude of phase error as the BF16 sim** (1.17 rad vs 1.23 rad), and both are on the order of the BF16 mantissa quantization step at the phase magnitude reached (~600 rad → step ≈ 2.3 rad). So the device datapath is provably operating *at BF16 precision*, matching "BH rounds fp32→BF16 for MACs" even with `fp32_dest_acc_en=True`.
- **The `bf16-input-only` control survives (0.99).** BF16-quantizing only the *input* `rad`, then accumulating in fp32, does **not** collapse — so the damage is specifically from BF16 **accumulation of the large phase values**, not from input rounding. *(Caveat: this "survives" result holds for the synthetic random F0 used here; for the smooth tonal F0 of the real model, BF16 input rounding is also damaging.)*

The mechanism the test surfaces: the phase ramps to hundreds of radians, and BF16's 8-bit mantissa gives a quantization step of `|phase| · 2⁻⁸` (~2 rad at |phase|≈600). Taken mod 2π that randomizes the sine argument — so `PCC(phase)` stays ~1.0 (correlation on a huge monotone ramp is blind to it) while `PCC(sin)` collapses.

**Conclusion.** Because a host-only BF16 run reproduces the collapse while the identical fp32 run does not, and the device's phase error matches BF16 precision, the SineGen sine collapse is the **BF16 numeric format**, not a TTNN kernel bug. No on-device kernel change (including bumping `MathFidelity` — HiFi passes only recover mantissa bits that are *stored*) closes it; `use_torch_phase_fallback=True`, which accumulates the phase in fp32 on the host, is the correct remedy (see Section 2.6).

### 4.5 Running it

```bash
pytest -s models/experimental/kokoro/tests/test_bf16_accumulation_hardware_limit_proof.py
```

`-s` prints the per-backend table above; the test needs a Tenstorrent device (opens one via the `device` fixture) and runs in a few seconds.

Companion tests: `test_sinegen_phase_fallback_proof.py` (per-stage PCC; the fallback restores phase-chain PCC > 0.99) and `test_tt_kmodel_pcc_degradation.py` (full-pipeline audio PCC per fallback combination).

### 4.6 Test source

Full test: `tests/test_bf16_accumulation_hardware_limit_proof.py`

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prove the SineGen phase collapse is a BF16-accumulation *hardware* limit, not a TTNN kernel bug.

The SineGen harmonic path at Kokoro scale (``upsample_scale=300``, ``T=48600``) collapses the
on-device ``sin(phase)`` PCC to ~0.31 vs the float32 reference. The open question this test
settles is *where* that error comes from:

    (A) a defect in a TTNN kernel (matmul / cumsum-add / lerp / sin), OR
    (B) the intrinsic precision of BF16 arithmetic, which Blackhole uses for its MAC datapath
        (``BH hardware rounds float32 -> BF16 for MACs``) regardless of ``fp32_dest_acc_en``.

Design — run the **exact same phase-chain math three ways** and compare:

    1. ``fp32 CPU``   torch, float32                      -> the golden reference.
    2. ``bf16 CPU``   torch, *the identical function*, only ``dtype=bfloat16``. Touches **zero**
                      TTNN kernels, runs entirely on the host.
    3. ``ttnn dev``   the real device path (``_run_tt_sinegen_stages``, HiFi3, fp32_dest_acc).

If (B) is the cause, then:
    * ``bf16 CPU`` must reproduce the device collapse (both ``sin`` PCC well below fp32), and
    * flipping *only the dtype argument* fp32->bf16 in one CPU function must be what breaks it, and
    * the device's absolute accumulated phase error must match the bf16-CPU sim's error magnitude
      (i.e. the device is operating at BF16 precision, not better, not worse).

All three hold. Because the CPU bf16 run contains no TTNN kernel yet collapses identically, the
collapse cannot be a kernel bug — it is the BF16 numeric format. This is why the production fix is
``use_torch_phase_fallback=True`` (accumulate the phase in fp32 on host); no on-device kernel change
can recover it.

Note on the mechanism (printed by the test): the phase ramps to ~hundreds of radians. BF16's
8-bit mantissa gives a quantization step of ``|phase| * 2**-8`` (~2 rad at |phase|~600). Taken mod
2*pi that randomizes the sine argument — so PCC(phase) stays ~1.0 (correlation on a huge ramp is
insensitive to it) while PCC(sin) collapses. Quantizing only the *input* ``rad`` (fp32 chain) leaves
``sin`` PCC > 0.9: the damage is specifically BF16 *accumulation*, not input rounding.

Companion proofs: ``test_sinegen_phase_fallback_proof.py`` (fallback restores PCC>0.99, per-stage),
``test_tt_kmodel_pcc_degradation.py`` (full-pipeline audio PCC per fallback combo).

Run::

    pytest -s models/experimental/kokoro/tests/test_bf16_accumulation_hardware_limit_proof.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F_torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.m_source_rng import (
    deallocate_m_source_rng_tt,
    make_zero_m_source_rng,
)
from models.experimental.kokoro.tests.test_sinegen_phase_fallback_proof import (
    _HARMONIC_NUM,
    _SAMPLING_RATE,
    _SINE_AMP,
    _UPSAMPLE_SCALE,
    _build_modules,
    _kokoro_f0,
    _run_ref_sinegen_stages,
    _run_tt_sinegen_stages,
)

# Collapse threshold: a sine whose PCC vs fp32 is below this is "destroyed".
_COLLAPSE_PCC = 0.5
# fp32 must stay essentially exact through the identical chain.
_EXACT_PCC = 0.99
# Input-only BF16 quantization (fp32 accumulation) must NOT collapse -> isolates accumulation.
_INPUT_ONLY_PCC = 0.9


def _pcc(ref: torch.Tensor, other: torch.Tensor) -> float:
    ref_f = ref.detach().float().reshape(-1)
    oth_f = other.detach().float().reshape(-1)
    n = min(ref_f.numel(), oth_f.numel())
    if n == 0:
        return float("nan")
    _, pcc = comp_pcc(ref_f[:n].unsqueeze(0), oth_f[:n].unsqueeze(0), pcc=0.0)
    return float(pcc)


def _mae(ref: torch.Tensor, other: torch.Tensor) -> float:
    ref_f = ref.detach().float().reshape(-1)
    oth_f = other.detach().float().reshape(-1)
    n = min(ref_f.numel(), oth_f.numel())
    return float((ref_f[:n] - oth_f[:n]).abs().mean())


def _phase_err_mod_2pi(ref_phase: torch.Tensor, other_phase: torch.Tensor) -> float:
    """Mean |phase error| wrapped into (-pi, pi] — the quantity that actually scrambles sin."""
    a = ref_phase.detach().float().reshape(-1)
    b = other_phase.detach().float().reshape(-1)
    n = min(a.numel(), b.numel())
    d = a[:n] - b[:n]
    d_wrapped = torch.remainder(d + math.pi, 2.0 * math.pi) - math.pi
    return float(d_wrapped.abs().mean())


def _rad_in_dtype(f0_btd: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``rad = (f0 * harmonics / sr) % 1`` computed at the given precision (rand_ini is zero here)."""
    dim = _HARMONIC_NUM + 1
    harmonics = torch.arange(1, dim + 1, dtype=dtype).reshape(1, 1, dim)
    f0d = f0_btd.to(dtype)
    fn = (f0d * harmonics).to(dtype)
    rad = (fn.float() / _SAMPLING_RATE).to(dtype)
    rad = torch.remainder(rad.float(), 1.0).to(dtype)
    return rad


def _cpu_phase_chain(rad: torch.Tensor, *, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """The SineGen phase chain (downsample -> cumsum -> lerp-up -> x2pi*scale -> sin) at ``dtype``.

    Op order and casts mirror the device path in ``_run_tt_sinegen_stages``: the cumsum is an
    iterative add (each partial sum re-rounded to ``dtype``), matching the on-device slice+add.
    Interpolation runs in fp32 then re-rounds, because on device the linear interp is a matmul
    whose *result* lands back in the working dtype.
    """
    rad = rad.to(dtype)
    rad_down_t = F_torch.interpolate(
        rad.transpose(1, 2).float(), scale_factor=1.0 / _UPSAMPLE_SCALE, mode="linear", align_corners=False
    ).to(dtype)
    rad_down = rad_down_t.transpose(1, 2)  # [B, T_down, dim]

    # cumsum as iterative add, each partial re-rounded to dtype (mirrors ttnn slice+add cumsum)
    phase = torch.zeros_like(rad_down)
    acc = torch.zeros_like(rad_down[:, 0:1, :])
    for t in range(rad_down.shape[1]):
        acc = (acc + rad_down[:, t : t + 1, :]).to(dtype)
        phase[:, t : t + 1, :] = acc

    phase_2pi = (phase.float() * (2.0 * math.pi)).to(dtype)
    phase_up_t = F_torch.interpolate(
        (phase_2pi.float().transpose(1, 2) * _UPSAMPLE_SCALE),
        scale_factor=float(_UPSAMPLE_SCALE),
        mode="linear",
        align_corners=False,
    ).to(dtype)
    phase_up = phase_up_t.transpose(1, 2)
    sin_raw = torch.sin(phase_up.float())
    return {
        "phase_up": phase_up.float(),
        "sin": sin_raw,
        "sine_amp": sin_raw * _SINE_AMP,
    }


@dataclass(frozen=True)
class Bf16AccumReport:
    # sin PCC vs fp32 golden
    pcc_sin_fp32_cpu: float  # identical function, fp32 -> must be ~1.0
    pcc_sin_bf16_cpu: float  # identical function, bf16 -> must collapse
    pcc_sin_input_only: float  # bf16 input, fp32 accumulation -> must NOT collapse
    pcc_sin_device: float  # real ttnn device path -> collapses
    # accumulated phase error magnitude (rad), wrapped mod 2pi
    phase_err_bf16_cpu: float
    phase_err_device: float
    bf16_quant_step: float  # |phase|_mean * 2**-8
    mean_abs_phase: float


def analyze_bf16_accumulation(device) -> Bf16AccumReport:
    f0 = _kokoro_f0()
    rng = make_zero_m_source_rng(1, f0.shape[1], _HARMONIC_NUM + 1)
    rand_ini_b1d = rng.rand_ini.reshape(1, 1, _HARMONIC_NUM + 1)
    ref = _run_ref_sinegen_stages(f0, rand_ini_b1d=rand_ini_b1d)

    # (3) Real device path — pure TTNN kernels.
    tt_mod, f0_tt, rng_tt = _build_modules(device, f0, use_torch_phase_fallback=False)
    tt = _run_tt_sinegen_stages(
        tt_mod,
        f0_tt,
        rand_ini_tt=rng_tt.rand_ini,
        noise_raw_tt=rng_tt.sinegen_noise,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(f0_tt)
    deallocate_m_source_rng_tt(rng_tt)

    # (1)/(2) Same CPU chain, only the dtype differs. (rand_ini==0 -> rad==rad_mod.)
    rad_bf16 = _rad_in_dtype(f0, torch.bfloat16)
    rad_fp32 = _rad_in_dtype(f0, torch.float32)
    cpu_fp32 = _cpu_phase_chain(rad_fp32, dtype=torch.float32)
    cpu_bf16 = _cpu_phase_chain(rad_bf16, dtype=torch.bfloat16)
    # Input-only: BF16-quantized rad, but accumulate in fp32 -> isolates accumulation from input rounding.
    cpu_input_only = _cpu_phase_chain(rad_bf16, dtype=torch.float32)

    mean_abs_phase = float(ref["S6_phase_up_rad"].detach().float().abs().mean())
    return Bf16AccumReport(
        pcc_sin_fp32_cpu=_pcc(ref["S7_sin_raw"], cpu_fp32["sin"]),
        pcc_sin_bf16_cpu=_pcc(ref["S7_sin_raw"], cpu_bf16["sin"]),
        pcc_sin_input_only=_pcc(ref["S7_sin_raw"], cpu_input_only["sin"]),
        pcc_sin_device=_pcc(ref["S7_sin_raw"], tt["S7_sin_raw"]),
        phase_err_bf16_cpu=_phase_err_mod_2pi(ref["S6_phase_up_rad"], cpu_bf16["phase_up"]),
        phase_err_device=_phase_err_mod_2pi(ref["S6_phase_up_rad"], tt["S6_phase_up_rad"]),
        bf16_quant_step=mean_abs_phase * (2.0**-8),
        mean_abs_phase=mean_abs_phase,
    )


def _log_report(r: Bf16AccumReport) -> None:
    print("\n=== BF16-accumulation hardware-limit proof (SineGen phase chain, T=48600) ===")
    print("  sin(phase) PCC vs fp32 golden — SAME chain, three backends:")
    print(f"    fp32  CPU  (identical fn, float32)   = {r.pcc_sin_fp32_cpu:.6f}   <- exact")
    print(f"    bf16  CPU  (identical fn, bfloat16)  = {r.pcc_sin_bf16_cpu:.6f}   <- collapses, NO ttnn")
    print(f"    ttnn  device (real kernels)          = {r.pcc_sin_device:.6f}   <- collapses")
    print(f"    bf16-input-only (fp32 accumulation)  = {r.pcc_sin_input_only:.6f}   <- survives")
    print("\n  Accumulated phase error (|Δphase| mod 2π, rad) — device runs at BF16 precision:")
    print(f"    bf16 CPU sim   = {r.phase_err_bf16_cpu:.4f} rad")
    print(f"    ttnn device    = {r.phase_err_device:.4f} rad")
    print(f"    BF16 quant step @ mean|phase|={r.mean_abs_phase:.1f} rad ≈ {r.bf16_quant_step:.3f} rad")
    print(
        "\n  Conclusion: a host-only BF16 run (zero TTNN kernels) reproduces the device collapse;\n"
        "  the identical fp32 run does not. The limit is the BF16 numeric format, not a kernel bug.\n"
        "  For full accuracy use use_torch_phase_fallback (fp32 phase accumulation on host).\n"
    )


def test_bf16_accumulation_is_hardware_limit_proof(device):
    r = analyze_bf16_accumulation(device)
    _log_report(r)

    # 1) The device collapses (this is the observed problem we are explaining).
    assert r.pcc_sin_device < _COLLAPSE_PCC, f"device sin PCC {r.pcc_sin_device:.4f} expected to collapse"

    # 2) A pure-CPU BF16 run of the SAME math — touching zero TTNN kernels — reproduces the collapse.
    assert r.pcc_sin_bf16_cpu < _COLLAPSE_PCC, (
        f"bf16 CPU sin PCC {r.pcc_sin_bf16_cpu:.4f} must collapse too (reproduces device without any ttnn kernel)"
    )

    # 3) The IDENTICAL function in fp32 is exact -> the only causal variable is the numeric format.
    assert r.pcc_sin_fp32_cpu > _EXACT_PCC, (
        f"fp32 CPU sin PCC {r.pcc_sin_fp32_cpu:.4f} must stay exact (same code, only dtype changed)"
    )

    # 4) Device and CPU-bf16 collapse to a comparable degree -> same phenomenon, not a kernel-specific defect.
    assert abs(r.pcc_sin_device - r.pcc_sin_bf16_cpu) < 0.25, (
        f"device ({r.pcc_sin_device:.4f}) and bf16-CPU ({r.pcc_sin_bf16_cpu:.4f}) collapse should match"
    )

    # 5) BF16-quantizing only the INPUT (fp32 accumulation) does NOT collapse -> it's accumulation, not input rounding.
    assert r.pcc_sin_input_only > _INPUT_ONLY_PCC, (
        f"input-only bf16 sin PCC {r.pcc_sin_input_only:.4f} should survive -> collapse is from BF16 accumulation"
    )

    # 6) The device accumulates the SAME magnitude of phase error as the BF16 CPU sim, and it is on the
    #    order of the BF16 quantization step -> the device datapath is operating at BF16 precision.
    assert r.phase_err_device > 0.3, f"device phase error {r.phase_err_device:.4f} rad expected to be BF16-scale"
    assert r.phase_err_bf16_cpu > 0.3, f"bf16 CPU phase error {r.phase_err_bf16_cpu:.4f} rad expected to be BF16-scale"
    ratio = r.phase_err_device / max(r.phase_err_bf16_cpu, 1e-9)
    assert 0.4 < ratio < 2.5, (
        f"device/bf16-CPU phase-error ratio {ratio:.2f} should be O(1): device runs at BF16 precision"
    )
```

---

## 5. Project Status — Done & Pending

Status snapshot as of 2026-07-02. Items marked *(verify)* are believed complete but need re-confirmation before sign-off.

### 5.1 Done

| # | Item | Notes |
|---|------|-------|
| 1 | Unit tests + module tests *(verify)* | Per-op / per-module correctness coverage in place |
| 2 | Full-model e2e PCC tests *(verify, values to update)* | Two modes: **teacher-forced** (reference prosody output fed into the TTNN path) and **free-run** (TT runs standalone, no reference prosody injected). Free-run audio PCC: **0.26** without fallback; **0.87** with STFT + phase-shift fallback |
| 3 | Full-model evaluation metrics | Whisper WER, speaker similarity, cFW2VD (Fréchet wav2vec), mel-PCC check |
| 4 | Demo up to 463 chars | Verified with torch phase-shift fallback |
| 5 | Missing-op reimplementation + math unit tests | STFT, iSTFT, LSTM built from existing TTNN ops that serve the same purpose; each has a math-correctness unit test |
| 6 | Module-level optimizations | CustomAlbert, prosody predictor, and Text Encoder — Text Encoder complete (see Section 3) |

### 5.2 Pending

| # | Item | Notes |
|---|------|-------|
| 1 | e2e performance test | RTF, latency, time-to-first-audio, throughput (char/s), bitrate, frames/s. **(a)** trace + 2CQ with the torch fallback flag disabled — for reporting (phase-shift TTNN impl adds audible noise in the demo); **(b)** 2CQ without trace — working model |
| 2 | Per-text demo perf | RTF, latency, time-to-first-audio, throughput (char/s) across different demo texts |
| 3 | Long-input verification | ~500-token phoneme ids (~800 chars) |
| 4 | Generator optimization + broader sweep | Optimize the generator; scan CustomAlbert / prosody predictor / Text Encoder for any remaining optimization scope |
| 5 | Device performance test | |
| 6 | Fallback audit | Verify accuracy / correctness of the fallbacks and understand the match. Phase-shift TTNN impl drifts the demo audio even at PCC > 0.99 — mean abs error is far higher than the fallback |
| 7 | Chunked-audio quality | Check audio quality when chunking with more chars / phoneme ids |
| 8 | ISL sweep test | |
| 9 | Single-layer perf test | |
| 10 | `requirements.txt` | |
| 11 | CI support | |
| 12 | SSRF check in demo | |
| 13 | Logit-level verification | Decide whether needed, given the hidden states from vocab are already available |
| 14 | README | |

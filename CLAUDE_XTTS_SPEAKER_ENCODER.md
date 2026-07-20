# XTTS-v2 ResNet Speaker Encoder (d-vector) — TTNN bringup (Block 2)

Parent: `CLAUDE_XTTS_TTNN.md` (read it first for shared decisions + integration contract).

## Status / Owner / Started
- Status: **DONE on TT** — full Block 2 (logmel → d-vector) at **PCC 0.99972** vs coqui.
  CPU reference PCC 1.0. All TTNN intermediate stages > 0.999.
- Owner: acicovic
- Started: 2026-07-20

## Role in pipeline
**Branch B** of conditioning. Consumes the reference clip (16000 Hz) and produces the
single **d-vector** (speaker embedding) injected into the HiFi-GAN vocoder (Block 4) as
`g=speaker_embedding`. Do NOT conflate with Branch A (Perceiver prefix, Block 1).

## Interface contract (from master)
| Direction | Tensor | Shape | dtype |
|-----------|--------|-------|-------|
| in | `reference_audio_16k` | waveform @ 16000 Hz | f32 |
| out | `speaker_embedding` (d-vector) | (1, 512, 1) | f32 → Block 4 (vocoder `g=`) |

## Files
- `reference/xtts_speaker_ref.py` — CPU torch mirror. `SpeakerReference` (full: audio→d-vector,
  lazy torchaudio front-end) + `ResNetSpeakerEncoder` (the ResNet **core**, logmel→emb, pure
  torch, no torchaudio). PCC **1.0** vs coqui golden (both logmel and final d-vector).
- `tt/ttnn_xtts_speaker.py` — `TTNNSpeakerEncoder` + `preprocess_speaker_parameters`. Runs the
  core (instancenorm → conv2d ResNet → ASP pooling → fc → L2) on device from the logmel.
- `tests/test_speaker_pcc.py` — PCC gate vs coqui golden (target 0.99). Prints per-stage PCC
  vs the CPU core for debugging. Standalone `__main__` opens its own device
  (`l1_small_size=32768`, needed by `ttnn.conv2d` halo) with a busy-device retry loop.
- Goldens in `golden/speaker/` (gitignored): `audio_16k [1,93680]`, `logmel [1,64,586]`,
  `spec`, `emb_prel2 [1,512]`, `emb_l2 [1,512]`, `speaker_embedding [1,512,1]`.

## Architecture (confirmed from coqui source, op-by-op)
Class: `TTS.encoder.models.resnet.ResNetSpeakerEncoder`, instantiated in
`TTS.tts.layers.xtts.hifigan_decoder.HifiDecoder` as:
`input_dim=64, proj_dim=512, layers=[3,4,6,3], num_filters=[32,64,128,256],
encoder_type="ASP", log_input=True, use_torch_spec=True`, audio_config
`{fft_size=512, win_length=400, hop_length=160, sample_rate=16000, preemphasis=0.97, num_mels=64}`.
Checkpoint prefix: `hifigan_decoder.speaker_encoder.*` (295 tensors).

Inference entry (`Xtts.get_speaker_embedding`): resample audio→16 kHz →
`speaker_encoder.forward(audio_16k, l2_norm=True).unsqueeze(-1)` → **[1, 512, 1]**.

`forward(x, l2_norm)`:
1. **Mel front-end** (`torch_spec` = PreEmphasis(0.97) + torchaudio.MelSpectrogram(64 mels, n_fft
   512, win 400, hop 160, hamming)) → `(spec + 1e-6).log()` = **logmel [1,64,T]** (T=586 for the
   5.86 s clip). *This is the TTNN block boundary — front-end stays on CPU (STFT is not a TTNN op;
   python_env has no torchaudio).*
2. `InstanceNorm1d(64, affine=False)` over time (biased var, eps 1e-5) → unsqueeze → [1,1,64,T].
3. `conv1` Conv2d(1→32,3x3,s1,p1) → **relu → bn1** (NB coqui's odd order: relu *before* bn).
4. `layer1..layer4`: SEBasicBlock ResNet, blocks [3,4,6,3], strides [1,2,2,2].
   - **SEBasicBlock**: conv1(3x3) → relu → bn1 → conv2(3x3) → bn2 → SE → (+ residual; downsample =
     Conv2d(1x1,stride)+BN on the first block of layers 2/3/4) → relu. Convs have **no bias**.
   - **SE**: global avg-pool over (H,W) → Linear(C→C/8) → relu → Linear(C/8→C) → sigmoid → scale.
   - Spatial: mel 64→32→16→8, time T→T/2→T/4→T/8. layer4 out [1,256,8,74].
5. `reshape` [1,256,8,74] → **[1, 2048, 74]** (feature index = channel*8 + mel; channel-major).
6. **Attentive Statistics Pooling (ASP)** `attention` = Conv1d(2048→128,k1) → relu →
   BatchNorm1d(128) → Conv1d(128→2048,k1) → Softmax over **time**. Then
   `mu = Σ_t x·w`, `sg = sqrt((Σ_t x²·w − mu²).clamp(min=1e-5))`, `x = cat([mu,sg]) → [1,4096]`.
7. `fc` Linear(4096→512) → `F.normalize(x, p=2, dim=1)` (L2) → unsqueeze(-1) → [1,512,1].

## TTNN implementation notes
- **Layout:** channels-last (NHWC) interleaved TILE throughout the ResNet — the natural
  `ttnn.conv2d` layout. After each conv: `to_memory_config(L1)` (sharded→interleaved) →
  `reshape (1,H,W,C)` → `to_layout(TILE)`. BN affine and SE scale are broadcast multiplies over
  the last (channel) dim, so they're trivial in NHWC.
- **BatchNorm (eval) folded to affine** on host: `scale = w/√(var+eps)`, `shift = b − mean·scale`
  (eps 1e-5). Applied as `x·scale + shift`. Avoids `ttnn.batch_norm`'s NCHW layout needs.
- **1×1 convs as linears:** the ASP `attention` Conv1d(k1) and SE FCs are plain `ttnn.linear`
  over the feature/channel dim.
- **ASP reshape:** `permute(layer4, (0,2,3,1)) → reshape (1, T, 2048)` reproduces coqui's
  channel-major (c*8+m) flatten. Softmax over time via `permute → softmax(dim=-1)`.
- **`ttnn.conv2d` needs `l1_small_size`** on the device (halo/sliding-window config tensor lives
  in L1_SMALL) — see BUG note; open device with `l1_small_size=32768`.

## PCC validation
- CPU reference vs coqui golden: **logmel PCC 1.0, d-vector PCC 1.0** (both the full audio→emb
  path in the coqui venv and the core(logmel)→emb path in python_env).
- TTNN vs coqui golden (target 0.99, bf16 convs): **PASSED, PCC 0.99972** on the [1,512,1]
  d-vector. Per-stage vs the CPU core: instancenorm 0.99998, conv1 0.99983, layer1 0.99953,
  layer2 0.99948, layer3 0.99950, layer4 0.99937, attn_w 0.99951, pool 0.99958, fc/emb 0.99972.
  (bf16 convs through 16 residual blocks; HiFi4+fp32-acc.) Error is dominated by the conv path
  and stays flat — the residual structure keeps it from accumulating.

## Findings log (dated)
- 2026-07-20: Confirmed encoder class is `ResNetSpeakerEncoder` (H/ASP), d-vector 512, layout
  (1,512,1) — resolves master open questions. Built + validated CPU reference at PCC 1.0.
  Wrote full TTNN core (conv2d ResNet + ASP). Hit L1_SMALL OOM in `ttnn.conv2d` when the device
  was opened without `l1_small_size`; fixed by `l1_small_size=32768`. Full on-device path passes
  at **PCC 0.99972** — the whole conv-heavy ResNet runs in TTNN (no CPU fallback needed beyond
  the mel/STFT front-end). `ttnn.conv2d` was NOT a blocker; auto-sharded `Conv2dConfig` with
  interleaved NHWC activations between ops worked for all 17 conv shapes (1×1 and 3×3, strides
  1 and 2). Wormhole warns HiFi4+fp32-acc can be less accurate than HiFi3 — did not matter here
  (0.99972 >> 0.99 target); switch to HiFi3 if a future clip is tighter.

## Open questions / TODO
- [x] Confirm encoder class / d-vector dim / layout — done (ResNetSpeakerEncoder, 512, (1,512,1)).
- [x] group_norm/conv tile-alignment risk when porting to TTNN — no issue (no group_norm here;
      conv2d handles non-tile-aligned spatial dims; ASP softmax over time T=74 correct).
- [ ] Integration: feed this d-vector as `g=` to Block 4 (HiFi-GAN) once that block exists.

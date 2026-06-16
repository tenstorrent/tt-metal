# Kokoro (experimental)

Repo-owned **Kokoro-82M** bring-up under `reference/` (HF weights) and **TTNN** ports under `tt/`.

## Full TTNN stack

- **PL-BERT + predictor** on device: `ttnn_kokoro_plbert`, `ttnn_kokoro_predictor`, `ttnn_kokoro_albert`, etc.
- **ISTFTNet vocoder** on device: `KokoroDecoderTt` / `KokoroIstftNetTt` (generator uses device `KokoroTtnnSineGen` by default).
- **End-to-end module**: `KokoroFullTtnn` in `tt/ttnn_kokoro_full_pipeline.py` composes the above with **no host torch vocoder**; discrete duration/alignment indices still use small CPU tensors (same as the PyTorch reference predictor).

---

## How to run the demo

From the **tt-metal** repo root:

```bash
export PYTHONPATH=$(pwd)
source python_env/bin/activate   # use the repo venv that matches your Metal build

# Full TTNN pipeline (default: on-device SineGen + on-device STFT)
python models/experimental/kokoro/demo/ttnn_kokoro_full_demo.py \
  --text "Hello from Tenstorrent." \
  --voice af_heart \
  --output out_ttnn.wav

# Recommended production parity (both CPU fallbacks — config E)
python models/experimental/kokoro/demo/ttnn_kokoro_full_demo.py \
  --text "Hello from Tenstorrent." \
  --voice af_heart \
  --torch-stft-fallback \
  --torch-phase-fallback \
  --output out_ttnn_config_e.wav
```

**Requirements:** `pip install "kokoro>=0.9.2" soundfile` and `espeak-ng` on PATH for G2P. Place `kokoro-v1_0.pth` locally or let `KModel` download from HuggingFace. Pass `--checkpoint /path/to/kokoro-v1_0.pth` to override auto-detection.

---

## How to run full-model tests

From the **tt-metal** repo root (requires a local Kokoro-82M checkpoint):

```bash
export PYTHONPATH=$(pwd)

# Full-pipeline PCC — recommended config E (STFT + phase fallbacks)
pytest -s models/experimental/kokoro/tests/test_tt_kmodel_pcc.py \
  -k test_tt_kmodel_stft_and_phase_fallback_pcc -v --timeout=600

# Full-pipeline PCC — no fallbacks (shows the BH-BF16 ceiling)
pytest -s models/experimental/kokoro/tests/test_tt_kmodel_pcc.py \
  -k test_tt_kmodel_generator_no_torch_fallback_pcc -v --timeout=600

# All kmodel PCC tests
pytest models/experimental/kokoro/tests/test_tt_kmodel_pcc.py -v --timeout=600

# Entire kokoro test suite
pytest models/experimental/kokoro/tests/ -v --timeout=600
```

**Fallback proof suite** — run with `-s` to print per-stage PCC tables:

```bash
pytest -s \
  models/experimental/kokoro/tests/test_sinegen_phase_fallback_proof.py \
  models/experimental/kokoro/tests/test_tt_kmodel_pcc_degradation.py \
  models/experimental/kokoro/tests/test_sinegen_voicing_input_not_op_proof.py \
  models/experimental/kokoro/tests/test_stft_atan2_sensitivity_proof.py
```

PL-BERT / predictor tests use the `mesh_device` fixture in `tests/conftest.py`. Vocoder PCC tests use the root `device` fixture.

---

## End-to-end quality metrics

Sample-wise waveform PCC is a poor free-run gate for a TTS vocoder (a tiny phase/source drift collapses it while the speech is perceptually identical), so the pipeline is scored with three perceptual / intelligibility metrics. All three generate audio on the Blackhole device and compare against the matched torch CPU-reference `KModel` for the same text/voice/seed:

| metric | what it measures | gate | test |
|--------|------------------|------|------|
| **ASR WER** | intelligibility — Whisper-small transcribes the TT audio; word error rate vs the prompt | < 30% | `test_tt_kmodel_asr_wer.py` |
| **mel PCC** | spectral parity — Pearson corr of the 80-band log-mel spectrogram vs the reference (phase-invariant) | > 0.95 | `test_tt_kmodel_mel_pcc.py` |
| **speaker cosine (SECS)** | speaker-identity parity — cosine of WavLM x-vector embeddings vs the reference (phase- & duration-invariant) | > 0.95 | `test_tt_kmodel_speaker_cosine.py` |

Measured on `af_heart`, text `"Hello world this is a speech synthesis test."`, seed 0, deterministic (2.95 s audio, 44 phonemes), across the full vocoder-fallback / STFT-formulation matrix:

| config | disable_complex | stft fb | phase fb | ASR WER | mel PCC | SECS |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| `phase_fallback` | False | off | on | 0.00% | 0.9929 | 0.9922 |
| `stft_and_phase_fallback` (config E) | False | on | on | 0.00% | 0.9928 | 0.9892 |
| `no_fallback` | False | off | off | 0.00% | 0.9722 | 0.9571 |
| `dc_phase_fallback` | True | off | on | 0.00% | 0.9932 | 0.9960 |
| `dc_no_fallback` | True | off | off | 0.00% | 0.9724 | 0.9686 |

All 15 runs PASS every gate. WER is 0.00% for every config (Whisper transcribes the audio verbatim) — speech stays fully intelligible even on the degraded `no_fallback` path, which is why ASR WER is a far more meaningful free-run gate than waveform PCC (≈0.28 there). mel PCC and SECS are tighter, frame-/timbre-level metrics: the CPU-fallback configs sit at ~0.993 / ~0.99, while the no-fallback configs drop to ~0.972 / ~0.957 — the residual BH-BF16 harmonic-source degradation surfaces here but stays above the 0.95 floor.

Run them with:

```bash
export PYTHONPATH=$(pwd)
# Full fallback / STFT-formulation matrix (all 5 configs each)
pytest -s models/experimental/kokoro/tests/test_tt_kmodel_asr_wer.py        --timeout=3600
pytest -s models/experimental/kokoro/tests/test_tt_kmodel_mel_pcc.py        --timeout=3600
pytest -s models/experimental/kokoro/tests/test_tt_kmodel_speaker_cosine.py --timeout=3600

# Single config (recommended config E only): add -k stft_and_phase_fallback
```

---

## Why CPU fallbacks are required (and why PCC degrades)

On Blackhole (BH), MAC ops round `float32 → bfloat16`. That is harmless across the prosody stack (PLBERT → predictor → TextEncoder, PCC > 0.998) but fatal in **two spots** on the vocoder harmonic-source path:

| failure point | mechanism | symptom without fallback |
|---------------|-----------|--------------------------|
| **SineGen phase chain** | tiny cumsum (~3×10⁻⁵ cycles) × `2π × upsample_scale` (≈ 1885) amplifies BF16 rounding into ~0.06–0.25 rad phase error per frame; `sin()` nonlinearly worsens it | `sine_wavs` PCC collapses to ~0.31; full audio PCC ≈ 0.28 |
| **STFT magnitude/phase** | near-zero off-frequency bins (~1e-5) get sign-flipped by BF16 `atan2` SFPU | `cos(phase)` PCC ≈ 0.6–0.8 on harmonic input |

Two CPU fallbacks recover these: `use_torch_phase_fallback` (SineGen phase accumulation on CPU float32) and `use_torch_stft_fallback` (STFT via `torch.stft` on CPU). Toggle them on `TTKModel`, `KokoroFullTtnn`, or the demo flags `--torch-phase-fallback` / `--torch-stft-fallback`.

**Recommended config E** (`stft + phase` fallbacks, reference built `disable_complex=False`):

| config | full-forward audio PCC | what it shows |
|--------|------------------------|---------------|
| `none` (all on-device) | ≈ 0.28 | BH-BF16 ceiling — badly degraded |
| `stft_only` | ≈ 0.29 | STFT fix alone is **not** enough; SineGen phase chaos still poisons audio |
| `phase_only` | ≈ 0.84 | phase fallback **alone** recovers the bulk — the dominant single lever |
| `stft + phase` (config E) | ≈ 0.88 (> 0.84 floor) | both fallbacks clear the production floor; STFT adds ~+0.04 on top of phase |

The degradation is **not fixable purely on-device** at Kokoro scale. The phase (SineGen) fallback is the one that matters most; STFT alone leaves the pipeline degraded, while phase alone recovers most of the deficit.

The residual ~0.10–0.15 gap to PCC 1.0 (config E ≈ 0.85–0.88) lives in the on-device harmonic source / F0 path at full sequence length, not in the prosody stack or the iSTFT reconstruction.

---

## Detailed proof tests

The four proof tests below isolate each failure mode. Run them together (see command above) and read the printed PCC tables.

### Proof 1 — SineGen phase fallback (`test_sinegen_phase_fallback_proof.py`)

**What it proves:** at Kokoro scale (`upsample_scale=300`, `T=48600`) the TTNN phase chain degrades on-device; `use_torch_phase_fallback=True` restores it.

**Method:** captures per-stage PCC/MAE in device execution order on the **same synthetic F0** against a float32 reference (measured PCC from the run above shown for the pure-TTNN path):

| step | stage | pure TTNN (BH BF16) | with `use_torch_phase_fallback` |
|------|-------|---------------------|----------------------------------|
| 00–07 | pre-phase + accumulation (`uv`, `fn`, `rad_frac`, `rad_rand_ini`, `rad_down`, `phase_cumsum`, `phase_up`) | **> 0.99** (phase_up PCC 0.99999) | > 0.99 |
| 08–10 | nonlinear sine (`sin`, `sine×amp`, `sine_wavs`) | **collapses to ~0.31** (sin PCC 0.307) | **> 0.99 (restored, sin PCC 0.999)** |

The collapse appears **only at `sin`**, not before it: `phase_up` keeps PCC ≈ 1.0 even though its MAE is already ~1.3 rad, because PCC is scale-invariant and the per-frame phase error is still a near-linear function of the reference at that point. The nonlinear `sin()` is what turns that error into decorrelation.

**Why it fails on-device:** the lerp upsample multiplies a tiny cumsum (~3×10⁻⁵ cycles) by `2π × upsample_scale ≈ 1885`. A BF16 rounding error of ~3×10⁻⁵ becomes ~0.06–0.25 rad of phase error per frame — comparable to `sine_amp = 0.1` — and `sin()` nonlinearly amplifies it further. Moving only the phase **accumulation** (`rad_down` → `cumsum` → lerp upsample → `× 2π`) to CPU float32 (while `sin`, `× amp`, and uv-mix stay on device) restores sine PCC from 0.307 to 0.999, a recovery of **+0.69 PCC** at full length.

**Key assertions:**
- pre-phase stages stay tight (PCC > 0.99) on pure TTNN
- `sin` and `sine×amp` PCC < 0.99 on pure TTNN at T=48600
- all phase-chain stages PCC > 0.99 with torch phase fallback
- `sin` PCC is worse than `phase_up` PCC (nonlinear amplification)

Note: this test uses synthetic F0 where `rad_frac` stays ≈ 1.0; modulo behaviour on real kmodel input is covered in Proof 3.

---

### Proof 2 — Full-pipeline degradation (`test_tt_kmodel_pcc_degradation.py`)

**What it proves:** the Kokoro vocoder PCC deficit cannot be closed on-device without the CPU fallbacks; the phase (SineGen) fallback is the dominant lever.

**Method:** runs the identical full `TTKModel` pipeline (text `"Hello from Tenstorrent."`, reference built `disable_complex=False` so it uses `TorchSTFT`, matching the TT `use_torch_stft_fallback` formulation) under four fallback configurations and compares full-forward audio PCC:

```
config          audio PCC    what it shows
─────────────────────────────────────────────────────────────────────────
none            0.275        BH-BF16 harmonic-source ceiling
stft_only       0.286        STFT fix alone does NOT recover — SineGen phase still broken
phase_only      0.842        phase fallback alone recovers the bulk
stft+phase      0.880        both fallbacks clear the production floor (> 0.84)

recovery (stft+phase − none)       = +0.604
phase-only vs stft-only            = +0.556   (phase is the dominant lever)
stft-on-top-of-phase increment     = +0.038   (STFT adds the final touch)
```

**Key assertions:**
1. no fallback: PCC < 0.6 (badly degraded)
2. STFT-only: PCC < 0.7 (phase chain still on-device poisons audio)
3. phase-only recovers > 0.2 PCC over no-fallback and beats stft-only
4. stft+phase: PCC > 0.84 and recovery > 0.3 over no-fallback
5. adding phase on top of STFT moves PCC by > 0.2 — proving phase fallback is necessary, not optional

**Interpretation:** the deficit is irreducible on-device. STFT fallback alone leaves the pipeline degraded because the SineGen phase chaos still corrupts the harmonic source. Phase fallback alone recovers ~0.55 PCC over stft-only; STFT on top of phase adds only ~+0.04.

---

### Proof 3 — A low `rad_frac` score is the F0 input, not any op (`test_sinegen_voicing_input_not_op_proof.py`)

**Vocabulary** (SineGen turns the pitch contour into a sine wave):

| name | meaning |
|------|---------|
| `f0` | per-sample pitch in Hz; `0` = unvoiced (silence) |
| `uv` | voiced/unvoiced mask, the boolean `f0 > 0` |
| `fn` | `f0 × harmonic numbers` |
| `rad_frac` | `(fn / sample_rate) % 1` — per-sample phase step, wrapped to `[0, 1)` |
| PCC | correlation of the on-device tensor vs the CPU-float32 reference; `1.0` = identical |

**What it proves:** `rad_frac` shows a low PCC (~0.54) end-to-end, which looks like a broken `% 1` (modulo) or `> 0` (threshold) op. Both ops are faithful — the drop is a sub-Hz disagreement in the **input** `f0_upsampled` that the discontinuous `> 0` turns into voicing-mask flips, which `rad_frac` then inherits.

**Method — three checks on the real kmodel `f0_upsampled`** (text `"Hello from Tenstorrent."`):

**A. Path-faithful** (ref stages on `f0u_ref`, TT stages on `f0u_tt`) — locates where the drop appears:

| step | PCC | what happens |
|------|-----|--------------|
| `f0_input` | 0.99995 | ref vs TT F0 correlate tightly (sub-Hz disagreement) |
| `fn_harmonics` | 0.99996 | smooth op — the F0 error is not amplified |
| `uv_mask` | **0.57** | `f0 > 0` flips near the voicing boundary (a discontinuous step) |
| `rad_frac` | **0.54** | inherits the `uv` drop — no new cliff at the modulo |

**B. Shared-input** (both paths fed `f0u_ref`) — isolates the modulo: `rad_frac` PCC = **1.0000** (MAE ~1e-5). On matched input `ttnn.remainder` is bit-faithful; at Kokoro scale `fn/sr ≈ 0.008` sits far from any wrap point, so `% 1` is a near-identity.

**C. Threshold torch vs device** — isolates `ttnn.gt`. Building `uv = f0 > 0` on the same `f0u_tt` both ways gives `PCC(uv_device, uv_torch) = 1.0`, and both score the **same 0.567506** against `uv_ref`:

| mask | built with | PCC vs `uv_ref` |
|------|------------|-----------------|
| `uv_device` | `ttnn.gt(f0u_tt, 0)` (on device) | **0.567506** |
| `uv_torch` | `f0u_tt > 0` (torch) | **0.567506** |

Moving `>` to torch changes the score by `+0.000000` — the op backend is irrelevant.

**Interpretation:** a low `rad_frac` PCC points upstream to F0 / voicing disagreements, not to `ttnn.remainder` or `ttnn.gt`. Both ops are faithful at Kokoro values, so a torch fallback for either recovers nothing — which is why the only fallbacks that move full-pipeline PCC are the SineGen **phase** and **STFT** ones (Proofs 1, 2, 4), not anything around `uv` / `rad_frac`.

**What it would take to recover this contribution:** the input divergence is seeded by the on-device prosody predictor — `f0_upsampled` is the upsampled predicted F0 (`_device_forward_prosody_stages`), computed in BF16, so it differs sub-Hz from the float32 reference. Since the SineGen ops downstream are exact, the *only* way to shrink this specific contribution is a **CPU fallback on the F0 prediction in the prosody predictor** (so `f0_upsampled` matches the reference), not any change around `uv` / `rad_frac`. This is a **secondary** lever, though: the prosody stages are individually high-PCC (> 0.998) and the dominant full-pipeline recovery comes from the phase and STFT fallbacks (Proof 2) — an F0 fallback would only tighten the residual harmonic-source gap, not move the headline number.

---

### Proof 4 — STFT atan2 sensitivity (`test_stft_atan2_sensitivity_proof.py`)

**What it proves:** STFT phase decorrelation is `atan2` sensitivity on near-zero-magnitude bins, not broken conv or a broken `ttnn.atan2` SFPU.

**Method:** on Kokoro harmonic input (`n_fft=20`, `hop=5`, `sine_amp=0.1`, `L=1500`) compares the on-device STFT conv path against `torch.stft`:

| metric | PCC | notes |
|--------|-----|-------|
| `X_real` (conv bins) | ≈ 1.0 | strided-conv STFT agrees with `torch.stft` |
| `X_imag` (conv bins) | ≈ 1.0 | same |
| `magnitude` | > 0.99 | `sqrt(x² + y²)` is faithful |
| `phase` (`atan2`) | ≈ 0.6–0.8 | decorrelates despite tight xy bins |
| `cos(phase)` | < 0.90 | phase error concentrated on near-zero bins |
| `atan2` on **identical** fp32 (x, y) inputs | > 0.99 | `ttnn.atan2` matches `torch.atan2` — not an SFPU bug |
| audio round-trip (transform → inverse) | > 0.999 | reconstruction is ~lossless despite low phase PCC |

Phase PCC by `|z|` region (measured, 3311 bins total):
- **high-|z| bins** (n=579): phase PCC ≈ 0.88 (faithful)
- **near-origin bins** (`|z| < 0.05`, n=2732): phase PCC ≈ 0.75 (ill-conditioned angle)

The near-origin bins are the **majority** (2732 of 3311), so the ~zero-energy ill-conditioned angles drag the overall phase PCC down to ≈ 0.79 even though every bin that carries real energy is faithful — which is exactly why the round-trip audio PCC stays at 0.999998.

**Key assertions:**
- `X_real` and `X_imag` PCC > 0.99 (conv is faithful)
- phase PCC well below xy PCC (degradation lives in `_magnitude_phase_from_xy`, not conv)
- `ttnn.atan2` on shared inputs PCC > 0.99 (sensitivity is geometric, not implementation)
- near-origin phase PCC worse than high-magnitude phase PCC
- audio round-trip PCC > 0.999 (bad phase on ~zero-energy bins never reaches the audio)

**Interpretation:** xy PCC ≈ 1.0 but phase PCC is low → the degradation is `atan2` near the origin, where the angle is ill-defined. Those bins carry negligible energy (`mag·e^{jφ} ≈ 0`), so the round-trip audio stays ~lossless. This is why `use_torch_stft_fallback` (CPU `torch.stft` with float64-precision atan2) adds ~+0.04 on top of the phase fallback in config E, and why raw phase PCC is a misleading proxy — reconstruction PCC is the metric that matters.

---

## Other tests

The `tests/` directory also holds per-module PCC tests (`test_tt_*_pcc.py`), additional injection proofs (`test_stft_refxy_injection_kmodel_proof.py`, `test_source_linear_tanh_fallback_proof.py`), and end-to-end ASR WER (`test_tt_kmodel_asr_wer.py`). Shared helpers live in `tests/kokoro_checkpoint.py`; config E kwargs are `STFT_PHASE_FALLBACK_KWARGS`.

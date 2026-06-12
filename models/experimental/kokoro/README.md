# Kokoro (experimental)

Repo-owned **Kokoro-82M** bring-up under `reference/` (HF weights) and **TTNN** ports under `tt/`.

## Full TTNN stack

- **PL-BERT + predictor** on device: `ttnn_kokoro_plbert`, `ttnn_kokoro_predictor`, `ttnn_kokoro_albert`, etc.
- **ISTFTNet vocoder** on device: `KokoroDecoderTt` / `KokoroIstftNetTt` (generator uses device `KokoroTtnnSineGen` by default).
- **End-to-end module**: `KokoroFullTtnn` in `tt/ttnn_kokoro_full_pipeline.py` composes the above with **no host torch vocoder**; discrete duration/alignment indices still use small CPU tensors (same as the PyTorch reference predictor).

## SineGen: device vs PyTorch (PCC / demos)

- Default: **`KokoroTtnnSineGen`** on device inside `SourceModuleHnNSF`.
- Toggle: `use_torch_sinegen=True` on `preprocess_kokoro_generator_parameters` / `preprocess_kokoro_decoder_tt_parameters`, or on **`KokoroFullTtnn`**, or **`--torch-sinegen`** on `demo/ttnn_kokoro_full_demo.py`. Harmonics then use the reference **PyTorch** `SineGen` on CPU (uploaded for the same TTNN `Linear` + `tanh` + STFT stack), which **raises waveform PCC** vs PyTorch compared to pure device SineGen. See **Why the CPU fallbacks are required** below and `tests/test_tt_source_module_hn_nsf_pcc.py` / `tests/test_tt_generator_pcc.py` for the per-module SineGen-mode PCC.

## Why the CPU fallbacks are required

The default Blackhole (BH) MAC pipeline rounds `float32 → bfloat16`, which is harmless across the
prosody stack but fatal in two spots on the vocoder's harmonic-source path. Two CPU fallbacks
recover them: `use_torch_phase_fallback` (SineGen phase accumulation) and `use_torch_stft_fallback`
(generator STFT). Both are off by default and toggled per the SineGen note above.

### Proof 1 — why the *phase* (SineGen) fallback is needed

`tests/test_sinegen_phase_fallback_proof.py` captures per-stage PCC of the SineGen phase chain at
Kokoro scale (`upsample_scale=300`, `T=48600`) against a float32 reference on the **same F0**:

| stage | pure TTNN (BH BF16) | with `use_torch_phase_fallback=True` |
|-------|---------------------|--------------------------------------|
| pre-phase (`uv`, `rad`, `rad_down`) | **> 0.99** (tight) | > 0.99 |
| phase chain (`phase_up`, `sin`, `sine×amp`) | **< 0.99 — collapses to ~0.21** | **> 0.99 (restored)** |

**Why:** the lerp upsample multiplies a tiny cumsum (~3×10⁻⁵ cycles) by `2π × upsample_scale ≈ 1885`.
A BF16 rounding error of ~3×10⁻⁵ is amplified into ~0.06–0.25 rad of phase error per frame —
comparable to `sine_amp = 0.1` — and `sin()` nonlinearly amplifies it further. Moving only the
phase **accumulation** to CPU float32 (sin / ×amp / uv-mix stay on device) restores sine PCC > 0.99,
a recovery of **> 0.5 PCC** at full length. This is the single largest fallback contribution.

### Proof 2 — where the residual ~0.10–0.15 PCC lives, and why it needs *another* fallback to close

`tests/test_tt_kmodel_pcc_degradation.py` runs the identical full pipeline (reference built
`disable_complex=False` so it uses `torch.stft`, matching the TT STFT fallback) under three configs:

| config | full-forward audio PCC | what it shows |
|--------|------------------------|---------------|
| `none` (all on-device) | ≈ 0.28 (**< 0.6**) | BH-BF16 harmonic-source ceiling — badly degraded |
| `stft_only` | ≈ 0.29 (**< 0.7**) | STFT fix alone is **not** enough; the SineGen phase chaos still poisons it |
| `phase_only` (STFT still on-device) | ≈ 0.84 | the phase fallback **alone** recovers the bulk — the dominant single lever |
| `stft + phase` (config E, recommended) | ≈ 0.88 (**> 0.84**) | both fallbacks clear the floor; STFT adds the final **+0.04** |

(measured on a 23-phoneme utterance; reference built `disable_complex=False`.) The asserted recovery
is **> 0.3** PCC end-to-end. Adding the *phase* fallback alone lifts PCC by **+0.55** over `stft_only`,
whereas adding the STFT fallback on top of phase contributes only **+0.04** — proving the deficit is not
fixable on-device and that the *phase* (SineGen) fallback, not STFT, is the one that matters.

### Proof 3 — the residual is NOT the merge `Linear`/`tanh`; a linear+tanh fallback does **not** help

A natural guess for the remaining ~0.10–0.15 gap (config E ≈ 0.85–0.88 → 1.0) is the `m_source`
harmonic **merge** (`sine_wavs → l_linear → tanh → sine_merge`), since
`tests/test_tt_source_module_hn_nsf_pcc.py` reports `linear_pre_tanh` ≈ 0.88. **That is wrong** —
and `tests/test_source_linear_tanh_fallback_proof.py` proves it by isolating the merge op-by-op
(trained weights, `dim=9`) under three configs:

| config | `sine_wavs` PCC | `linear_pre_tanh` PCC | |
|--------|-----------------|------------------------|---|
| phase **OFF**, on-device merge | 0.9446 | 0.8994 | merge output just **tracks** its degraded input |
| phase **ON**, on-device merge | 0.99999 | **0.99999** | given a tight `sine_wavs`, the BF16 merge is **faithful** |
| phase **ON**, CPU-fp32 merge | 0.99999 | 0.99999 | linear+tanh fallback ΔPCC = **+0.000003** (nothing) |

So the ≈ 0.88 `linear_pre_tanh` in the per-op diagnostic is **inherited** from a phase-degraded
`sine_wavs` (that diagnostic runs phase-fallback OFF), *not* produced by the merge op. Once the
**phase** fallback feeds it a good `sine_wavs`, the on-device BF16 `Linear` + `tanh` matches the
reference to PCC > 0.9999, and moving those two ops to CPU float32 changes PCC by ~1e-6.

**Conclusion / why more fallback doesn't close it:** adding `use_torch_linear_fallback` /
`use_torch_tanh_fallback` would **not** raise PCC — there is no BF16 loss left in the merge to
recover. The remaining ~0.10–0.15 lives upstream in the harmonic **source / STFT phase** at full
sequence length (the prosody stack is > 0.998 and the iSTFT is exact vs float64), and that is kept
on-device by design — so the test floors sit at the honest ~0.84–0.88 ceiling rather than being
engineered toward 0.99 with extra CPU fallbacks.

### Proof 4 — the residual enters at the STFT *inputs* (the harmonic source), not the STFT/atan2/iSTFT

The other suspect is the STFT phase (`phase = atan2(imag, real)`). Two injection experiments pin
the residual to the STFT **inputs** — i.e. the harmonic-source spectrum — and exonerate the entire
STFT→atan2→iSTFT chain downstream of those inputs.

**(a) End-to-end (`tests/test_stft_refxy_injection_kmodel_proof.py`).** Run the real config-E
pipeline; dump the *reference* `TorchSTFT.transform` `(X_real, X_imag)` (`KOKORO_DUMP_STFT_XY=1`),
then re-run the TT generator feeding those reference bins into the on-device `sqrt`/`atan2`
(`KOKORO_INJECT_REF_STFT_XY=1`, via `TTTorchSTFT._transform_inject_ref_xy`):

| config E (23 phonemes) | full-forward audio PCC |
|------------------------|------------------------|
| baseline (TT STFT of the **TT** harmonic source) | 0.8789 |
| inject **reference** atan2 inputs `(X_real, X_imag)` | **0.9962** |

So replacing just the STFT *inputs* with the reference recovers **+0.117** — essentially the whole
residual. Everything from the atan2 onward (atan2, the conv conditioning, `conv_post`, the iSTFT) is
faithful; the deficit is entirely in *producing* those inputs. Config E already runs the STFT on CPU
(`torch.stft`, atan2 identical to the reference), so the TT bins differ from the reference bins
**only** because the TT harmonic **source** (SineGen/m_source) differs — which is exactly where the
degradation proof localizes the deficit. (This is the same lever as injecting the reference
har_source directly → G7 audio ≈ 0.998.)

**(b) STFT round-trip on a *shared* input (`tests/test_stft_atan2_injection_proof.py`).** Confirms
the STFT op chain itself is faithful: on a shared harmonic signal the on-device transform→inverse is
~0.9999 even though the raw `phase` PCC is only ~0.79 (that bad phase sits on ~zero-energy bins where
`mag·e^{jφ} ≈ 0`, so it never reaches the audio; magnitude PCC ≈ 1.0). Note the *phase* metric itself
cannot be lifted to 0.99 by injecting reference inputs (only ~0.79→0.86) — it is the ill-defined
angle on near-zero bins — which is why phase PCC is a misleading proxy and the *reconstruction* is
the metric that matters.

Together: `atan2` is only an *amplifier* of an already-different harmonic source — the lever is the
on-device source, not the STFT/atan2.

## Demos

```bash
export PYTHONPATH=$(pwd)
# PyTorch reference (CPU/CUDA)
python models/experimental/kokoro/demo/reference_demo.py --text "Hello." --output out.wav
# Full TTNN (Tenstorrent device + `kokoro` for G2P/voice packs only)
python models/experimental/kokoro/demo/ttnn_kokoro_full_demo.py --text "Hello." --voice af_heart --output out_ttnn.wav
# Same, but PyTorch SineGen for harmonics (compare audio / PCC vs default)
python models/experimental/kokoro/demo/ttnn_kokoro_full_demo.py --torch-sinegen --output out_ttnn_torchsg.wav
```

Install: `pip install "kokoro>=0.9.2" soundfile` and `espeak-ng` on PATH where G2P is needed.

## Tests

From **tt-metal** root:

```bash
export PYTHONPATH=$(pwd)
pytest models/experimental/kokoro/tests/ -v --timeout=600
```

The `tests/` directory holds three kinds of tests (shared helpers live in `tests/kokoro_checkpoint.py`):

1. **Per-module PCC** — `test_tt_*_pcc.py`, each asserting a single TTNN module vs its reference
   (PLBERT/Albert, text/duration encoders, LSTM, prosody predictor, AdaIN/AdaLayerNorm/ResBlk,
   upsample, SineGen, source module, STFT (`custom`/`torch`), decoder, generator, full `kmodel`).
2. **Fallback proofs** — `test_sinegen_phase_fallback_proof.py` (Proof 1 above),
   `test_tt_kmodel_pcc_degradation.py` (Proof 2 above),
   `test_source_linear_tanh_fallback_proof.py` (Proof 3 above — the merge `Linear`/`tanh` is faithful
   in BF16 and a linear+tanh fallback does not raise PCC),
   `test_stft_refxy_injection_kmodel_proof.py` (Proof 4(a) — injecting the reference STFT atan2
   inputs into config E recovers PCC 0.879 → 0.996, localizing the residual to the harmonic source),
   `test_stft_atan2_injection_proof.py` (Proof 4(b) — STFT round-trip on a shared input is ~lossless;
   the STFT/atan2 chain itself is faithful), and
   `test_stft_atan2_correlation_proof.py` (shows the STFT `atan2` decorrelation is a near-zero-bin
   precision effect, not a `ttnn.atan2` bug).
3. **End-to-end** — `test_tt_kmodel_pcc.py` (full-pipeline PCC across fallback configs) and
   `test_tt_kmodel_asr_wer.py` (Whisper WER intelligibility gate).

PL-BERT / predictor PCC tests use the `mesh_device` fixture in `tests/conftest.py`. TTNN vocoder PCC tests use the root `device` fixture where applicable.

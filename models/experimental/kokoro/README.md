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
  models/experimental/kokoro/tests/test_sinegen_modulo_pcc_proof.py \
  models/experimental/kokoro/tests/test_stft_atan2_sensitivity_proof.py
```

PL-BERT / predictor tests use the `mesh_device` fixture in `tests/conftest.py`. Vocoder PCC tests use the root `device` fixture.

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

### Proof 3 — The `rad_frac` drop is inherited from voicing (`uv`), not the modulo (`test_sinegen_modulo_pcc_proof.py`)

**The question it settles:** in end-to-end diagnostics `rad_frac = (fn / sr) % 1` shows a low PCC (~0.54). Is `ttnn.remainder` the culprit — i.e. is the **modulo amplifying a small F0 error into a large one**? **No.** The modulo is faithful; the low PCC is *inherited* from an upstream **voicing-mask** disagreement and the modulo passes it through unchanged.

**Why one might suspect the modulo:** `% 1` *is* discontinuous at integer boundaries — `0.999` and `1.001` wrap to opposite ends, so a tiny input error near a wrap point could in principle explode. But at Kokoro scale `fn / sr ≈ 200 / 24000 ≈ 0.008` — nowhere near an integer boundary — so the modulo runs in its smooth, near-identity region and cannot amplify. The genuinely discontinuous op on this path is the boolean threshold `uv = f0 > 0`, not the modulo.

**Method — two experiments on the real kmodel `f0_upsampled`** (text `"Hello from Tenstorrent."`):

**A. Path-faithful** — ref stages on `f0u_ref`, TT stages on `f0u_tt` (each path's own F0):

| step | PCC | what happens |
|------|-----|--------------|
| `f0_input` | 0.99995 | ref vs TT upsampled F0 correlate tightly (sub-Hz absolute disagreement) |
| `fn_harmonics` | 0.99996 | `fn = f0 × harmonics` stays tight — the F0 error has **not** been amplified yet |
| `uv_mask` | **0.57** ← drop | `uv = f0 > 0` flips 0↔1 on frames near the voicing boundary (infinite-gain step) |
| `rad_frac` | **0.54** | **inherits** the `uv` drop — no *new* cliff appears at the modulo |

**B. Shared-input** — both paths fed the **same** `f0u_ref` (isolates the op itself):

| step | PCC | what happens |
|------|-----|--------------|
| `rad_frac` | **1.0000** | modulo is bit-faithful (MAE ~1e-5, pure BF16 `ttnn.remainder` noise) |

Every pre-phase op through `rad_rand_ini` also stays at PCC ≈ 1.0 in the shared-input run.

**Where the amplification really is:** the sub-Hz F0 disagreement (`f0_input` PCC 0.99995) is harmless through the *smooth* ops (`fn` stays 0.99996), but the **boolean `f0 > 0` near the voicing threshold flips the mask** — an effectively infinite-gain step on a binary signal — collapsing `uv_mask` to ~0.57. `rad_frac` is ~0 on unvoiced frames and small-positive on voiced ones, so it carries the same voiced/unvoiced structure and tracks `uv_mask` (within 0.05). The modulo passes that structure through; it adds no cliff of its own.

**Key assertions:**
- shared-input `rad_frac` PCC > 0.99, MAE < 1e-3 (modulo is faithful on matched input)
- path-faithful `fn` stays tight (> 0.99) but `uv_mask` PCC < 0.7 (the threshold is the amplifier)
- path-faithful `rad_frac` PCC tracks `uv_mask` within 0.05 — modulo does not add a separate cliff

**Interpretation:** when you see low `rad_frac` PCC in end-to-end diagnostics, look upstream at F0 / voicing (`uv`) disagreements, not at `ttnn.remainder`. The modulo is faithful at Kokoro operating values; the boolean voicing threshold is the op that turns a sub-Hz F0 error into a large PCC drop.

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

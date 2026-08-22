# Kokoro-82M bring-up report

That's the milestone. **Kokoro-82M is fully brought up on a single P150 — end-to-end, on-device, PCC-verified, producing real audio.** And to correct earlier pessimism: it actually converged; the watchdog had killed a *later* perf-tracing step *after* the gates already passed.

## What's complete and verified (independently re-run on device)
- **End-to-end test PASSES:** log-spectrogram PCC = **0.9933** (gate ≥ 0.95), `pred_dur` **exact match** to HF, **20/20** graduated modules invoked in the real forward path.
- **Real audio generated:** `kokoro_tt.wav` — 2.10 s @ 24 kHz for *"kokoro is open source"* (saved to `models/demos/kokoro_82m/`).
- **All neural compute on-device:** the 4 vocoder components that were CPU fallbacks (`decoder`, `generator`, `source_module_hn_n_s_f`, `sine_gen`) got **re-expressed as an explicit native TTNN chain** — `host_op_selftest` reports **zero neural aten ops** on host (the 14 residual host ops are just integer bookkeeping for variable-length TTS).
- Per-stage PCC 0.998–1.0; decoder+vocoder math **exact (1.0000)**.
- The low raw-waveform PCC (0.075) is **not a bug** — it's the NSF vocoder's chaotic F0-phase sensitivity (HF perturbing its own F0 by 1e-6 already drops it to 0.95); the phase-invariant metric is the correct one and it passes.

## How much of the model is successful
- **Bring-up: ~100%.** 20 NEW native + 1 REUSE, and the vocoder chain native in the e2e pipeline. Full model runs on the P150 and matches HF.

## What's left — optimization (not started)
This is the remaining chunk:
- **Trace + 2CQ:** partial — `encode` traces host-free (PCC 1.0); `prosody`/`decode`/`vocode` are single-CQ fallback due to data-dependent length. Getting those onto the traced fixed-capacity path is the main perf work.
- **Op fusion, memory-config/sharding tuning, perf-vs-target measurement** — all still to do.

## Artifacts (in `models/demos/kokoro_82m/`)
- `kokoro_tt.wav` — the generated speech
- `RUN_REPORT.md` — gates, per-stage PCC, routing, the waveform-PCC analysis
- `observed_report.md` — full tool-behavior log + the norm-fix intervention + enhancement notes
- `tests/e2e/test_e2e_tts.py`, `tt/pipeline.py`, `demo/demo_tts.py` — the runnable pipeline (in the worktree)

Reproduce:
```bash
./python_env/bin/python -m pytest models/demos/kokoro_82m/tests/e2e/test_e2e_tts.py -s
./python_env/bin/python -m models.demos.kokoro_82m.demo.demo_tts --out kokoro_tt.wav
```

The bring-up goal is done today.

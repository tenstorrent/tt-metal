Diagnostic scripts from the 2026-09-20 vocoder bring-up investigation, copied
here from `/tmp` scratch so they survive a session boundary. See
`models/demos/audio/cosyvoice2/BRINGUP_STATUS.md` for the full narrative,
findings, and what each of these was used to establish — this directory is
the supporting scripts, not documentation in its own right.

Not production code: these are one-off host-side repros/cross-checks (real
vs. our own reference/port), not covered by the PCC test suite, and not
imported by anything under `tt/`.

- `cv2_frontend.py` — real mel-spectrogram/xvec/speech-token frontend
  reimplementation (needed because installing `matcha-tts` broke the venv's
  torchaudio ABI earlier this bring-up).
- `stage1_eval_v3_savemel.py` — real zero-shot Stage 1 eval, saves the flow
  decoder's mel (`stage1_v3_mel.npy`, included) and our own vocoder's output
  on it.
- `real_cosyvoice_pkg/` — a minimal local shim of the real, unmodified,
  Apache-2.0 `cosyvoice` package (fetched from `FunAudioLLM/CosyVoice` on
  GitHub) with just enough of `cosyvoice.hifigan`/`transformer`/`utils` to
  import `HiFTGenerator`/`ConvRNNF0Predictor` standalone, without pip
  installing the full package (which risks the same torchaudio-ABI breakage
  `matcha-tts` caused).
- `real_vocoder_check.py` — the decisive test: feeds `stage1_v3_mel.npy` into
  the real official vocoder above with real `hift.pt` weights. Confirmed
  clean/crisp audio, proving the flow decoder's mel is fine and the bug is in
  our own vocoder port.
- `build_artifact_v3.py` — splices the vocoder isolation test's audio into
  the published artifact (https://claude.ai/artifact/9KQKjr5nHX9aR4Ez7A95t2).
- `attn_cross_check.py`, `ffn_cross_check.py`, `stage_bisect.py`,
  `op_by_op_bisect.py`, `euler_step_check.py`, `onnx_cross_check.py` — the
  (now largely superseded, likely-red-herring) flow-decoder bisection
  against `flow.decoder.estimator.fp32.onnx`. Kept because the bisection
  *technique* (substituting real `diffusers` classes, extracting ONNX
  intermediate tensors by mutating graph outputs, per-Euler-step divergence
  tracking) is reusable if the vocoder fix doesn't fully resolve the
  audio-quality complaint.

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

## The real root cause: `TtStft` silently corrupt at >= 65,536 samples (added later, 2026-09-20)

The vocoder-port bug that made the audio noisy/robotic/metallic was `TtStft`
(`tt/hifigan/stft.py`): its framing conv used a weight hoisted with
`ttnn.prepare_conv_weights` and returned garbage (PCC ~0.27, output ~30x too
small) for any input of 65,536 samples or more -- every real utterance longer
than ~2.7 s at 24 kHz. Fixed in commit `1cefdec6fc`; confirmed by ear. These
scripts are how it was found and pinned down. Outputs (wavs, dumps) go to
`$COSYVOICE2_DEBUG_OUT` (default `/tmp/cosyvoice2_debug`), never the repo. All
need the device and `PYTHONPATH` set so `import ttnn` and `models.` resolve.

- `f0_dtype_check.py` — F0 predictor device vs torch on the real 464-frame mel
  at bf16 and fp32, expressed in audibility terms (cents of pitch error,
  accumulated phase drift in cycles, voiced/unvoiced flips). Showed the earlier
  "F0 drift" numbers were already fp32 and that F0 error is small in cents.
- `f0_ablation_ab.py` — same mel, same noise draw, six runs: official class,
  torch reference, TT as shipped, TT decoder with torch F0 injected ("A"),
  torch decoder with TT F0 injected ("B"), torch with a different noise draw.
  Phase-insensitive log-mel distance per frame plus wavs to listen to. "A" not
  improving when F0 was corrected is what pointed at the decode path.
- `decode_stage_bisect.py` — taps every TT sub-module of `TtHiFTDecoder.decode`
  against a float64 torch reference (identical mel and excitation on both
  sides): CUM error (accumulated) vs LOCAL error (added by that stage), with
  gain so a level error is visible. Every module was locally accurate; the
  error entered through the `stft` tap.
- `stft_length_sweep.py` — `TtStft` vs `torch.stft` from 480 samples to 80 s.
  Pytest boundary cases: `tests/pcc/test_stft.py`.
- `repro_prepare_conv_weights_2p16.py` — standalone, pure-ttnn repro of the
  defect (no CosyVoice imports), suitable for an upstream report.
- `conv_config_matrix.py` — the 2x2 {prepared, raw weight} x {accurate, safe
  config} matrix on the six 128-ch k=11 convs at length 18560, each against
  float64 truth. Showed prepared == raw there and that the "safe" fallback is
  the *less* accurate config (2.7% vs 0.4%), which led to the resolver change
  in `conv.py`/`upsample.py`.

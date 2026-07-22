# GitHub Issues to File (after project completion)

> These are the GitHub issues identified during the CosyVoice2-0.5B Stage-1
> TTNN bring-up that should be filed on `tenstorrent/tt-metal`. They are
> documented here rather than filed immediately (per the working agreement,
> the issues will be created after the project completes). Each issue includes
> the full body, labels, and the evidence/reproduction.

## Issue 1 — Model-card: CosyVoice2-0.5B TTS on Wormhole N300

**Title:** Model bring-up: CosyVoice2-0.5B (TTS) on Wormhole N300

**Labels:** `model`, `wormhole`, `N300`

**Body:**

```
### Summary
Bring up CosyVoice2-0.5B (Alibaba FunAudioLLM text-to-speech) on Wormhole N300
using TTNN. This is a Stage-1 (functional correctness) bring-up — no trace/2CQ,
sharding tuning, bf8, or streaming in scope.

### Target
- Checkpoint: `FunAudioLLM/CosyVoice2-0.5B` (HuggingFace, rev `eec1ae6c`, 4.6 GB)
- Three on-device components: Qwen2.5-0.5B LLM, flow-matching decoder (UNet1D
  estimator, NOT a DiT), HiFT vocoder.
- 4 generation modes: SFT, zero-shot, cross-lingual, instruct.
- 5 languages: zh, en, ja (katakana), yue (Cantonese), ko.

### Reuse strategy
- The LLM backbone is stock Qwen2.5-0.5B; `models/tt_transformers` already
  supports the Qwen2.5 family (prefill + decode + KV cache). We reuse that path
  and add only CosyVoice glue (speech-token embedding table, `llm_decoder` head,
  top-k/RAS sampling, sequence assembly).
- Flow encoder uses ESPnet relative-position self-attention (NOT RoPE) — a fresh
  `tt/flow/rel_pos_attention.py` is required (tt_transformers attention is
  RoPE-based and cannot express it).
- ConvTranspose1d → `ttnn.conv_transpose2d` mapping validated (PCC ≥ 0.99999).
- DSP glue (SineGen2 + n_fft=16 iSTFT) runs on host in Stage 1 (separate issue
  for native `ttnn.istft`).

### Acceptance criteria (C6–C8) — MEASURED
- Full TTNN pipeline (LLM+flow+vocoder) runs on N300 with no errors. ✓
- LLM decode: 34.1 tok/s (≥30 target ✓); E2E RTF: 2.17 (<0.5 target ✗ — Stage-2: flow on host CPU is bottleneck).
- Token accuracy (teacher-forced top-25): zero_shot 96%, cross_lingual 100%, instruct2 100%, sft 98% (all >95% ✓).
- PCC vs PyTorch reference: LLM 0.997, flow 1.0, vocoder 1.0 (all ≥0.99 ✓).
- WER: 0.000 (whisper-large-v3, <3.0 target ✓); speaker similarity: 82.9 (CAM++, >60 target ✓).
- README with setup/run instructions. ✓

### Plan
Full living plan: `models/demos/cosyvoice/BRINGUP_PLAN.md`.
Concise entry point: `models/demos/cosyvoice/RESUME.md`.

### Verified environment
`models/demos/cosyvoice/model_data/REQUIREMENTS_INSTALLED.txt` (uv-managed
Python 3.10, torch 2.11.0+cpu, ttnn 0.1.dev29059, transformers 5.10.2).
```

---

## Issue 2 — Missing op: native `ttnn.istft` (inverse STFT)

**Title:** TTNN missing op: `ttnn.istft` (inverse short-time Fourier transform)

**Labels:** `ttnn`, `missing-op`

**Body:**

```
### Context
CosyVoice2's HiFT vocoder ends with an iSTFT head (`cosyvoice/hifigan/
generator.py::_istft`, line 499) that converts the network's predicted
magnitude/phase back to a waveform via `torch.istft`:

    inverse_transform = torch.istft(
        torch.complex(real, img),
        self.istft_params["n_fft"],   # 16
        self.istft_params["hop_len"],  # 4
        self.istft_params["n_fft"],   # win_length = n_fft
        window=self.stft_window)

### Parameters
- n_fft = 16 (tiny — 4 samples/frame, hop_len=4)
- window = hann(n_fft)
- center = False (the generator pre-trims to a multiple of hop_len)

### Impact
- Stage 1: this head runs on **host** (torch fallback) — sanctioned by the TTNN
  model-bringup guide §2.6 ("fall back to torch for unsupported ops, file an
  issue"). The op is tiny and not perf-critical (4 samples/frame, once per
  utterance), so host fallback is acceptable for Stage-1 correctness.
- Stage 2: if perf analysis shows the host iSTFT is a bottleneck (unlikely given
  its size), a native `ttnn.istft` would move it on-device.

### Request
Add a native `ttnn.istft` (and ideally `ttnn.stft`) op. Reference: PyTorch
`torch.istft`. A DFT-based implementation via `ttnn.matmul` with a precomputed
IDFT matrix is also feasible for small n_fft.

### Reproduction
The op is invoked in `cosyvoice/hifigan/generator.py::ISTFTDecoder.decode`
(called by `HiFTGenerator.decode`). See
`models/demos/cosyvoice/BRINGUP_PLAN.md` §6 (STFT/iSTFT row) + §9 (DSP-glue
risk) for the full decision rationale.
```

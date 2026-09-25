# Parakeet TDT 0.6B v3 — TTNN implementation for Tenstorrent Blackhole

SPDX-FileCopyrightText: © 2026 Abror Shopulatov
SPDX-License-Identifier: Apache-2.0

Production-style TTNN implementation of [nvidia/parakeet-tdt-0.6b-v3
](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): a 600M-parameter
conformer-based ASR model with TDT (Token-and-Duration Transducer) greedy
decoding, supporting 25 European languages. Ported from the pinned
checkpoint (revision `541d1f99…`) with all dimensions driven by config.

## Package layout

| Path | Contents |
|---|---|
| `tt/` | TTNN implementation: `ttnn_parakeet.py` (main backend), `precise_subsampling.py` |
| `demo/` | standalone CLI: 16 kHz wav/flac → transcript on TT device |
| `reference/` | PyTorch FP32 reference path (`torch_parakeet.py`) |
| `tests/` | portable tests (`test_parakeet.py`) + on-device checks (`check_*.py`) |
| `benchmarks/` | CPU FP32 baseline script, synchronized latency profiling |
| `docs/` | `PROFILING_PLAN.md` (measured evidence), `OPEN_ISSUES.md` (limitations) |

## Precision policy

`bf16` (default): bf16 encoder matmul weights with fp32 activations,
accumulation, residual stream, LayerNorm, softmax and subsampling stack.
The prediction network, encoder/decoder projectors and joint head keep
fp32 weights in every mode (TDT duration argmaxes can be near-ties).
Host RNE weight rounding to 9 mantissa bits before upload.

`fp32`: all weights fp32 (diagnostic mode).

`bfp8_b`: **unsupported by design** — the subsampling and TDT decoder
precision requirements are incompatible with block-float shared exponents
on this model. The factory fails clearly per the precision policy.

LayerNorm and softmax are composed from fp32 primitives rather than
using the fused kernels (the fused variants measured ~1.5e-3 NRMSE
vs exact on fp32 inputs on this runtime — see `tt/ttnn_parakeet.py`
docstring for the full derivation).

## Measured results

| Gate | Precision | Result | Details |
|---|---|---|---|
| smoke | bf16 | **PASS** | tokens exact 1/1, encoder NRMSE 0.018, p50 84 ms (RTF 26.6) |
| bringup | bf16 | **PASS** (8/8) | tokens exact 8/8, max NRMSE 0.033, median p50 **85 ms** |
| full | bf16 | 18/20 PASS | 2 failures: 1 NRMSE 0.060 (tokens correct), 1 token near-tie |

Matched TT-vs-CPU FP32 speedups (transcribe p50):
- short clip: **4.0×** (84 ms vs 238 ms)
- long clip (~16 s): **3.6×** (254 ms vs 910 ms)
- batch of 2: **3.9×** (108 ms vs 424 ms)
- RTF range: 26–64 (batch/long)

A100 FP32 context: median case p50 82 ms — TT bf16 bringup (85 ms) is
comparable. A100 BF16 itself fails 2/8 bringup cases on tone and
long-clip gates.

## Known limitations (measured, gates not widened)

1. Full-corpus: 2/20 clips fail — one encoder NRMSE 0.060 (libri-1993,
   longer speech; tokens still correct) and one greedy token near-tie
   (libri-2078; encoder NRMSE 0.008, well within gate).
2. BFP8_B unsupported by design (see above).
3. `ttnn.transformer.scaled_dot_product_attention` not used (its
   `attn_mask` input produces incorrect results on this runtime).
4. Requests serialized per backend; no streaming/chunked mode.

## Running

```bash
# Device inference
python demo/demo.py --checkpoint /weights --device-id 0 --precision bf16 clip.flac

# Portable tests (no device required)
pytest tests/test_parakeet.py -m "not device"

# On-device tests
pytest tests/test_parakeet.py
python tests/check_transcribe.py
```

Out of scope: >32 s clips, streaming/chunked inference, beam search,
fine-tuning, ASR-leaderboard claims.

## License

- Code: Apache-2.0 (SPDX headers on all source files)
- Model weights: [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Independent implementation; no NVIDIA code was copied.

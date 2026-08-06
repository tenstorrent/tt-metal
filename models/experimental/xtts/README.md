<!-- SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# XTTS-v2

## Platforms:
    Blackhole (p150)

## Introduction
[XTTS-v2](https://huggingface.co/coqui/XTTS-v2) is a zero-shot multilingual
text-to-speech model: given text and a few seconds of reference audio, it speaks
the text in that voice. It runs as three stages, all ported to TTNN here:

1. **Conditioning** — the reference waveform's 80-mel goes through a conditioning
   encoder to `cond_latents [1, 32, 1024]`, which becomes the GPT's audio prompt.
2. **GPT** — a 30-layer GPT-2-style decoder autoregressively emits discrete audio
   codes over a fixed KV cache, one code per step.
3. **HiFi-GAN** — a speaker encoder produces a global conditioning vector `g`, and
   the vocoder turns the GPT's latents plus `g` into a 24 kHz waveform.

Everything runs on device. The only host touchpoint is the BPE text tokenizer,
which is not a tensor op.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- The first run downloads ~1.9 GB of XTTS-v2 weights to the HF cache

```sh
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export ARCH_NAME=blackhole
```

## How to Run

Generate speech from text and a reference voice. Output is written to
`generated/xtts_demo/xtts_demo.wav`:

```sh
python models/experimental/xtts/demo/xtts_demo.py \
    --text "Hello from Tenstorrent." --ref-audio en_sample.wav
```

`--ref-audio` takes a local WAV path or an HF `coqui/XTTS-v2` sample name. The
demo is fully traced — three chained traces (setup, decode step, vocoder), so
every on-device op runs inside a trace — and logs latency and RTF (real-time
factor; RTF < 1 is faster than real-time).

Correctness tests — every ported module against its torch reference by PCC:

```sh
pytest models/experimental/xtts/tests/pcc
```

## Directory layout
| path | contents |
| --- | --- |
| `reference/` | pure-torch reference implementation; the PCC ground truth |
| `tt/` | the TTNN port (conditioning, mel frontend, GPT, speaker encoder, HiFi-GAN) |
| `tests/pcc/` | per-module PCC validation plus end-to-end and traced-pipeline tests |
| `demo/` | `xtts_demo.py` — text + reference audio to WAV |
| `eval/` | objective TTS metrics: CER (Whisper-large-v3), UTMOS, SECS (ECAPA2) |

## Details
- The GPT runs in bf16; its latents are cast to fp32 at the handoff to the fp32
  HiFi-GAN decoder.
- The traced decode loop replays a fixed number of steps and trims at the STOP
  token afterwards, because a captured trace cannot branch. `max_tokens` is
  therefore a cost, not a budget.
- Sampling (temperature 0.65 / top-k 50 / top-p 0.85) runs on device and is not
  bit-exact across runs, so takes differ from one another.

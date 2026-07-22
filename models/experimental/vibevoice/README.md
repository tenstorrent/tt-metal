# VibeVoice-1.5B (TT-Metal experimental)

Reference PyTorch setup for porting [VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) to TTNN. The backbone is **Qwen2.5-1.5B** (28 layers, hidden 1536, GQA); plan to reuse or wrap [`models/tt_transformers/`](../../tt_transformers/) for `language_model`.

Weights and demo assets are **not** vendored in this tree. On first run, demos and tests download:

- **Model weights:** [`microsoft/VibeVoice-1.5B`](https://huggingface.co/microsoft/VibeVoice-1.5B) into
  `models/experimental/vibevoice/weights/VibeVoice-1.5B` (requires `huggingface_hub`).
- **Demo text + voices:** [vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice/tree/main/demo)
  (`demo/text_examples` and `demo/voices`) into `models/experimental/vibevoice/resources/` via
  `common/resource_utils.py`.

Override the checkpoint location with:

```bash
export VIBEVOICE_MODEL_PATH=/path/to/VibeVoice-1.5B
```

## Layout

```
vibevoice/
├── README.md
├── common/
│   ├── config.py            # paths, HF repo id, transformers pin
│   ├── model_utils.py       # resolve path + auto-download weights
│   └── resource_utils.py    # download demo text/voices from upstream GitHub
├── reference/               # vendored 1.5B-only torch model (from VibeVoice repo)
│   ├── modular/             # config + modeling (imported as `modular.*`)
│   ├── processor/           # tokenizer/audio processor (`processor.*`)
│   └── schedule/            # DPM solver (`schedule.*`)
├── resources/               # auto-downloaded demo assets (gitignored content)
│   ├── voices/              # from github .../demo/voices
│   └── text/                # from github .../demo/text_examples
├── weights/                 # auto-downloaded HF checkpoint (gitignored content)
├── tests/
│   ├── conftest.py            # pytest: reference/ on PYTHONPATH + shared fixtures
│   └── pcc/
│   ├── pcc_helpers.py      # shared LM PCC helpers (HF ref, TT LM builder, PCC utils)
│   ├── test_decoder_layer_pcc.py  # Devstral-style layer-0 decode (no prefill)
│   ├── test_prefill.py   # full prefill chain vs HF reference
│   └── test_decode.py    # post-diffusion decode chain vs pinned ref conditions
└── tt/                      # TTNN layers (empty initially)
```

## Dependencies

Pin `transformers` — **4.57+** breaks `generate()` KV-cache behavior for this model.

```bash
pip install 'transformers==4.51.3' torch accelerate diffusers tqdm librosa scipy huggingface_hub
```

The processor also pulls **Qwen/Qwen2.5-1.5B** tokenizer assets from the Hugging Face cache (`QWEN_TOKENIZER` in `common/config.py`).

## Quick start (from tt-metal root)

```bash
export PYTHONPATH=$(pwd)

# PCC tests (auto-download weights; skipped if download fails)
pytest models/experimental/vibevoice/tests/pcc/ -v
```

## TTNN demo (on device)

`demo/demo.py` runs on-device TTNN inference (no HuggingFace reference model) and writes
`{output_dir}/{demo_id}/{demo_id}_tt.wav`. It is text-driven: `--text <path>` for a custom script,
or `--demo <id>` as a shortcut for `resources/text/<id>.txt`. Multi-speaker demos auto-enable
voice cloning from `resources/voices/`.

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
export ARCH_NAME=wormhole_b0 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml   # or blackhole

# Default demo (default script, eager — no trace)
python models/experimental/vibevoice/demo/demo.py

# Multi-speaker demo, cap the AR loop at 32 tokens, verbose stage/timing logs
python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_45min --max_new_tokens 32 --debug
```

### Run with trace

`--trace` ttnn-captures the whole steady-state speech-diffusion frame (neg-LM + diffusion +
post-diffusion + pos-LM) as one fully device-driven graph — the "llama shape": positions
self-advance on device, RoPE is gathered on device, and the pos hidden is loop-carried — and
replays it per frame. It gives **≈11–12 tok/s** steady-state decode vs ≈2.4 tok/s eager on the
45-min climate demo, and opens the device with a ~1.4 GB trace region + 2 command queues.

```bash
python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_45min --max_new_tokens 32 --trace
```

| Flag | Env var | Scope | Notes |
|------|---------|-------|-------|
| `--trace` | `VV_TRACE_SEGMENT=1` | whole segment, device-driven (llama shape) | fused frame replayed per frame; ~1.4 GB trace region + 2 CQs |

## Speaker similarity (SIM) test

`tests/pcc/test_e2e_sim.py` checks that on-device TTNN generation preserves the *cloned speaker's
identity*: it voice-clones a target speaker on TT, embeds the generated audio and the
reference/impostor voices with a speaker-verification (SV) model, and asserts the generated speech
is closer to the intended target than to any impostor (standard SIM-O verification), including a
4-speaker self-identification confusion matrix.

```bash
pytest models/experimental/vibevoice/tests/pcc/test_e2e_sim.py -v -s
```

**SV backend — why `microsoft/wavlm-base-plus-sv`, not the model from the paper.** The
[VibeVoice technical report](https://arxiv.org/abs/2508.19205) computes SIM with a **WavLM-large
fine-tuned** SV model — the UniSpeech `wavlm_large_finetune.pth` (WavLM-large backbone + an
ECAPA-TDNN x-vector head). We deliberately do **not** ship that model. Its code and checkpoint
([microsoft/UniSpeech](https://github.com/microsoft/UniSpeech), which in turn borrows from the
unlicensed [lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN)) are licensed
**CC BY-SA 3.0**, whose *ShareAlike* clause requires derivative works to carry the same license —
incompatible with this repo's **Apache-2.0**. Instead the test uses
**`microsoft/wavlm-base-plus-sv`**, a WavLM x-vector head that ships with 🤗 transformers
(Apache-2.0), needs no extra dependency or separately-licensed checkpoint, and keeps the whole path
license-clean.

Trade-off: base_plus produces a *compressed* cosine scale (different speakers still score ~0.5-0.7,
vs the fine-tuned model's ~0.9 same-speaker / ~0 impostor separation). So the test asserts a
**relative** target-vs-impostor margin — the SV model must still rank the correct speaker first, by
a margin — which is robust to the compressed scale, rather than the paper's absolute same-speaker
threshold.

## Language model / chain PCC tests

The LM prefill and decode paths are validated as part of the **full prefill / decode chain** PCC
tests (vs a bf16 HuggingFace Qwen2 reference), plus a standalone decoder-layer regression. Shared
helpers live in `tests/pcc/pcc_helpers.py`; fixtures (`vv_config`, `lm_state`) are in
`tests/conftest.py`.

- **Decoder layer (regression):** `test_decoder_layer_pcc.py::test_decoder_layer_decode_pcc` —
  Devstral-style layer-0 decode; random hidden states `[1, 1, H]`, empty KV cache, positions 0–9,
  no prefill. Isolates decode SDPA at low cache depth (min PCC ~0.99997).
- **Full prefill chain:** `test_prefill.py::test_full_prefill_chain_pcc` — the integrated
  prefill path (acoustic tokenizer → connector → scatter into embeddings → LM prefill →
  `last_hidden_state`) plus per-layer KV cache, vs the bf16 HF Qwen2 reference; synthetic-input ISL
  sweep 2k … 64k, gated at `PCC >= 0.99`.
- **Full decode chain:** `test_decode.py::test_decode_ref_cond_frame_pcc` — the
  post-diffusion decode chain against pinned reference diffusion conditions.

```bash
# Decoder-layer regression (fast)
pytest models/experimental/vibevoice/tests/pcc/test_decoder_layer_pcc.py -v -s

# Full prefill / decode chain
pytest models/experimental/vibevoice/tests/pcc/test_prefill.py \
       models/experimental/vibevoice/tests/pcc/test_decode.py -v -s
```

Individual component PCC tests (acoustic/semantic tokenizers, connector, diffusion head, DPM
scheduler, LM head) live alongside these in `tests/pcc/`.

## Porting notes

| Submodule | Reference | TT target |
|-----------|-----------|-----------|
| Language model | Qwen2 in `modeling_vibevoice` | `tt_transformers` Qwen2.5-1.5B |
| Acoustic / semantic tokenizers | `modular_vibevoice_tokenizer.py` | `tt/` (later) |
| Diffusion head | `modular_vibevoice_diffusion_head.py` | `tt/` (later) |
| Pipeline | `modeling_vibevoice_inference.py` | `tt/` generate loop |

Closest template: [`models/experimental/speecht5_tts/`](../speecht5_tts/) (`reference/` = PyTorch gold, `tt/` = TTNN, `tests/pcc/` = PCC).

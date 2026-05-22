# VibeVoice-1.5B (TT-Metal experimental)

Reference PyTorch setup for porting [VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) to TTNN. The backbone is **Qwen2.5-1.5B** (28 layers, hidden 1536, GQA); plan to reuse or wrap [`models/tt_transformers/`](../../tt_transformers/) for `language_model`.

Weights are **not** vendored in this tree. Point to a local checkout of the HF snapshot:

```bash
export VIBEVOICE_MODEL_PATH=/path/to/VibeVoice-1.5B
```

Default (if unset): `/home/iguser/devstral2/VibeVoice/VibeVoice-1.5B`

## Layout

```
vibevoice/
├── README.md
├── conftest.py              # pytest: reference/ on PYTHONPATH
├── common/config.py         # MODEL_PATH, voices, transformers pin
├── reference/
│   ├── vibevoice/           # vendored 1.5B-only Python (from VibeVoice repo)
│   ├── model_print.py
│   └── run_inference.py
├── resources/
│   ├── voices/              # demo voice presets (not weights)
│   └── text/                # short scripts for smoke / PCC
├── tests/pcc/
└── tt/                      # TTNN layers (empty initially)
```

## Dependencies

Pin `transformers` — **4.57+** breaks `generate()` KV-cache behavior for this model.

```bash
pip install 'transformers==4.51.3' torch accelerate diffusers tqdm librosa scipy
export VIBEVOICE_MODEL_PATH=/path/to/VibeVoice-1.5B
```

The processor also pulls **Qwen/Qwen2.5-1.5B** tokenizer assets from the Hugging Face cache (`QWEN_TOKENIZER` in `common/config.py`).

## Quick start (from tt-metal root)

```bash
export PYTHONPATH=$(pwd)
export VIBEVOICE_MODEL_PATH=/path/to/VibeVoice-1.5B

# Print architecture
python models/experimental/vibevoice/reference/model_print.py

# End-to-end reference TTS (CPU default; use --device cuda if available)
python models/experimental/vibevoice/reference/run_inference.py \
  --output_dir /tmp/vibevoice_out

# PCC reference tests (skip if weights missing)
pytest models/experimental/vibevoice/tests/pcc/ -v
```

## Porting notes

| Submodule | Reference | TT target |
|-----------|-----------|-----------|
| Language model | Qwen2 in `modeling_vibevoice` | `tt_transformers` Qwen2.5-1.5B |
| Acoustic / semantic tokenizers | `modular_vibevoice_tokenizer.py` | `tt/` (later) |
| Diffusion head | `modular_vibevoice_diffusion_head.py` | `tt/` (later) |
| Pipeline | `modeling_vibevoice_inference.py` | `tt/` generate loop |

Closest template: [`models/experimental/speecht5_tts/`](../speecht5_tts/) (`reference/` = PyTorch gold, `tt/` = TTNN, `tests/pcc/` = PCC).

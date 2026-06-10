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
├── conftest.py              # pytest: reference/ on PYTHONPATH
├── common/
│   ├── config.py            # paths, HF repo id, transformers pin
│   ├── model_utils.py       # resolve path + auto-download weights
│   └── resource_utils.py    # download demo text/voices from upstream GitHub
├── reference/
│   ├── vibevoice/           # vendored 1.5B-only Python (from VibeVoice repo)
│   ├── model_print.py
│   └── run_inference.py
├── resources/               # auto-downloaded demo assets (gitignored content)
│   ├── voices/              # from github .../demo/voices
│   └── text/                # from github .../demo/text_examples
├── weights/                 # auto-downloaded HF checkpoint (gitignored content)
├── tests/pcc/
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

# Print architecture (downloads weights on first run)
python models/experimental/vibevoice/reference/model_print.py

# End-to-end reference TTS (CPU default; use --device cuda if available)
python models/experimental/vibevoice/reference/run_inference.py \
  --output_dir /tmp/vibevoice_out

# PCC tests (auto-download weights; skipped if download fails)
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

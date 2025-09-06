## Falcon models on Tenstorrent (N150/N300/T3K/TG)

This document summarizes the changes and usage to run the following Hugging Face models on Tenstorrent hardware using the TT Transformers demo:

- tiiuae/Falcon-H1-0.5B-Instruct
- tiiuae/Falcon3-1B-Instruct
- tiiuae/Falcon-H1-7B-Instruct

### What changed in this repo

- Model config updates (`models/tt_transformers/tt/model_config.py`):
  - Enable `trust_remote_code` automatically for `falcon-h1` and `falcon3` HF repos.
  - Add tokenizer fallback mappings for `Falcon-H1-0.5B`, `Falcon3-1B`, `Falcon-H1-7B`.
  - Provide conservative per-device prefill chunk sizes for Falcon variants.
- HF weight loading (`models/tt_transformers/tt/load_checkpoints.py`):
  - Normalize Falcon-style paths from `transformer.*` to `model.*` and unify attention names.
  - Split fused attention `self_attention.query_key_value.{weight|bias}` into `q_proj/k_proj/v_proj`.
  - Map `self_attn.dense -> self_attn.o_proj`, and `mlp.dense_4h_to_h -> mlp.down_proj`.
  - Split Falcon MLP `mlp.dense_h_to_4h.{weight|bias}` into `mlp.gate_proj` and `mlp.up_proj`.
- Decode/output robustness:
  - In `tt/model.py`, restrict argmax to the first `vocab_size` columns before sampling.
  - In demo, clamp printed token IDs to `[0, vocab_size-1]` to avoid tokenizer overflow during logs.
- Norm key resolution
  - Dynamically resolve `ffn_norm`/`final_layernorm` aliases in `tt/decoder.py` and `tt/model.py`.
- Demo support (`models/tt_transformers/demo/simple_text_demo.py`):
  - Add `Falcon-H1-0.5B`, `Falcon3-1B`, and `Falcon-H1-7B` to `supported_models` list.

### How to run

Set environment (example N300):

```bash
source python_env/bin/activate
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export MESH_DEVICE=N300
```

Run batch-1 (latency) and batch-32 (throughput) demos:

```bash
# Falcon-H1-0.5B
export HF_MODEL=tiiuae/Falcon-H1-0.5B-Instruct
pytest -q models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
pytest -q models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-32"

# Falcon3-1B
export HF_MODEL=tiiuae/Falcon3-1B-Instruct
pytest -q models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
pytest -q models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-32"

# Falcon-H1-7B
export HF_MODEL=tiiuae/Falcon-H1-7B-Instruct
# If your local pytest config/plugins interfere, try a minimal runner:
pytest -q models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
pytest -q models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-32"
```

Numbers captured from demo logs on N300; tokens/s/user are greedy decode. Throughput is per-batch total.

| Model | Batch | 1st token (ms) | tok/s/user (avg) | Throughput (avg) |
|---|---:|---:|---:|---:|
| Falcon-H1-0.5B | 1 | ~10.7 | ~93.4 | ~93.4 |
| Falcon-H1-0.5B | 32 | ~10.9 | ~91.0 | ~2911 |
| Falcon3-1B | 1 | ~12.2 | ~83.5 | ~83.5 |
| Falcon3-1B | 32 | ~13.1 | ~76.3 | ~2440 |
| Falcon-H1-7B | 1 | ~81.3 | ~45.3 | ~45.3 |
| Falcon-H1-7B | 32 | ~64 | ~40.9 | ~1305 |

### Validation and notes

- Unit tests add coverage for Falcon weight path normalization and fused QKV/MLP splits.
- Demo results log per-user decoded text for sanity; token clamping only affects printing, not inference.
- For best performance, allow compile artifacts to warm and re-run.

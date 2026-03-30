# Speculative decoding from a recorded base (minimal bundle)

Self-contained **record replay + NextN MTP draft** under `specfr/`. Imports **`specfr` only**—not `models.demos.speculative_deepseek_r1_broad`.

## Scope (read this)

- **Edits for this workflow belong in this folder** (`models/demos/speculative_from_record_deepseek_r1/`). Do **not** change `models/demos/speculative_deepseek_r1_broad/` unless you explicitly intend to work in the full demo tree.
- The canonical copy of shared logic lives under `speculative_deepseek_r1_broad`; this bundle is a **trimmed** slice for portability.

## Compute server (reservation)

Use your reserved host when running heavy jobs:

`yzc-swc08-special-dchrysostomou-for-reservation-63054`

(SSH from your environment; DNS from this workspace may not resolve the name.)

## Layout

| Path | Purpose |
|------|---------|
| `run_mtp_from_record_cpu.py` | Entry script: trace replay + default **full MTP CPU** draft (`NextNSglangCPUDraftAdapter`); optional `--sglang-draft-structure`. |
| `scripts/materialize_nextn_embed_head_aux_from_r1_shards.py` | One-time download of two R1 Hub shards → `embed_tokens` + `lm_head` aux `.safetensors` when the NextN snapshot omits them (see below). |
| `specfr/` | Minimal modules: config, engine, trace replay, verification, HF cache helpers, `models_draft` (branching helpers only), NextN CPU draft + structure draft (HF NextN loader in `nextn_hf_nextn_model.py`), FP8 `dequantize`. |

**Removed as unused in this bundle:** `specfr/models_base.py` and `specfr/reference/` (HF R1 reference modeling). The record run uses `TraceReplayBaseAdapter` only and does not load the full base model from `reference/`.

## Run

```bash
cd models/demos/speculative_from_record_deepseek_r1
python run_mtp_from_record_cpu.py --max-new-tokens 32
```

Default `--record` is `DEFAULT_MTP_RECORD_PATH` in `specfr/default_paths.py`.

### Embed / LM head auxiliary file (optional)

The NextN Hub snapshot often does **not** ship standalone `embed_tokens` and final `shared_head.head` tensors. If loaders warn about missing embed/head, build a small aux file once (downloads two shards from `deepseek-ai/DeepSeek-R1-0528`, ~12 GB download; output ~4 GB bf16):

```bash
cd models/demos/speculative_from_record_deepseek_r1
python scripts/materialize_nextn_embed_head_aux_from_r1_shards.py --out /path/to/embed_head_aux.safetensors
```

Use `--hf-home` if Hugging Face cache should not use `DEFAULT_HF_HOME` from `specfr/default_paths.py`. Then pass `--embed-head-aux-safetensors /path/to/embed_head_aux.safetensors` to `run_mtp_from_record_cpu.py`, or place the file at `DEFAULT_EMBED_HEAD_AUX_PATH` so the default path is picked up automatically.

## Dependencies

PyTorch, transformers, huggingface_hub, safetensors; Hub access for `lmsys/DeepSeek-R1-NextN`.

## SPDX

Files under `specfr/` retain original SPDX headers from their source copies.

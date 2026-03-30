# Case study: full Hugging Face NextN as draft (record base)

This document is an **additional** track alongside the default **full MTP CPU** NextN + record flow (`run_nextn_mtp_from_record_cpu.py` uses `NextNSglangCPUDraftAdapter`). Nothing here replaces that script.

## What exists today (unchanged)

- The main record script uses **`NextNSglangCPUDraftAdapter`** (full MTP layer). **`NextNMTPHeadDraftAdapter`** (fusion + head only) remains in `models_draft.py` but is not selected by that CLI.
- Drafting runs the **MTP fusion subgraph** (`enorm` / `hnorm` / `eh_proj` / `shared_head`) on **`decode_state.last_hidden_state`** from the **record** — i.e. the same residual the full R1 model had at that step, without running the base transformer.

## What this case study adds

**Default (`case_study_nextn_full_layer_draft_from_record_cpu.py`, no `--ids-only-draft`):**

- **`NextNSglangStructureDraftAdapter`** (`nextn_sglang_structure_draft.py`) — same order as SGLang **`DeepseekModelNextN`**: **`eh_proj(cat(enorm(embed), hnorm(hidden)))`** where `hidden` is **`decode_state.last_hidden_state`** from the trace on the first draft step (then draft post-norm hidden for deeper steps), then **`DeepseekV3DecoderLayer`** + **`model.norm`** + **`lm_head`**. Loads fusion mats from **`nextn_layer_parameters.safetensors`** plus the full Hub **`AutoModelForCausalLM`**.

**Legacy (`--ids-only-draft`):**

- **`NextNFullHuggingfaceDraftAdapter`**: **prefill** on `committed` token ids, then **incremental decode** with **KV cache** — **no** `eh_proj` / `enorm` / `hnorm` on the record hidden (those tensors stay unused in HF forward).

### Why acceptance differed for the old ids-only path (not “bad init”)

The file **`nextn_layer_parameters.safetensors`** holds **two** kinds of weights:

| Subgraph | Tensors (examples) | Used by |
|----------|-------------------|---------|
| **MTP fusion** | `eh_proj`, `enorm`, `hnorm`, `shared_head.*` | **`NextNSglangCPUDraftAdapter`**, **`NextNSglangStructureDraftAdapter`**, + record `last_hidden_state` |
| **One decoder layer** | MLA, MoE experts, `shared_experts`, … | **`NextNSglangStructureDraftAdapter`** and **`NextNFullHuggingfaceDraftAdapter`** (HF `DeepseekV3Model`) |

Stock **`modeling_deepseek.py`** does not wire `eh_proj` / `enorm` / `hnorm` into `forward`. The **legacy ids-only** draft therefore ignored fusion mats. **Verification** compares draft tokens to **full R1** logits from the trace; ids-only MoE+MLA on **tokens alone** often **did not** match that distribution. The **default SGLang-order** draft in this script is meant to align the **draft** side with how SGLang uses NextN (fusion + layer).

| Aspect | MTP-head (`run_nextn_mtp…` default) | This case study (default) | This study `--ids-only-draft` |
|--------|-------------------------------------|---------------------------|--------------------------------|
| Draft adapter | `NextNSglangCPUDraftAdapter` (main record script) | `NextNSglangStructureDraftAdapter` | `NextNFullHuggingfaceDraftAdapter` |
| Decoder layer | No (fusion + `shared_head` only) | Yes (Hub layer 0) | Yes |
| Record `last_hidden` | Yes | Yes | No |
| Cost | Lower | Full MoE forward / step | Full MoE forward / step |

## How to run

```bash
export PYTHONPATH=/path/to/tt-metal:$PYTHONPATH

# CPU (defaults: --device cpu --dtype float32). Keep depth / tokens small; MoE is RAM-heavy.
python models/demos/speculative_deepseek_r1_broad/scripts/case_study_nextn_full_layer_draft_from_record_cpu.py \
  --draft-mtp-greedy --depth 1 --max-new-tokens 8 -q

# GPU + low precision (recommended if you have the VRAM)
python models/demos/speculative_deepseek_r1_broad/scripts/case_study_nextn_full_layer_draft_from_record_cpu.py \
  --device cuda --dtype bfloat16 \
  --depth 2 --top-k 4 --max-paths 16

# Greedy draft on GPU
python models/demos/speculative_deepseek_r1_broad/scripts/case_study_nextn_full_layer_draft_from_record_cpu.py \
  --device cuda --dtype bfloat16 --draft-mtp-greedy --depth 2

# Legacy ids-only draft (no eh_proj/enorm/hnorm on record hidden)
python models/demos/speculative_deepseek_r1_broad/scripts/case_study_nextn_full_layer_draft_from_record_cpu.py \
  --ids-only-draft --draft-mtp-greedy --depth 1 --max-new-tokens 8 -q
```

Optional: `--nextn-local-only` with `--nextn-repo-id /path/to/local/snapshot` after a manual `snapshot_download`.

## Disk / cache notes

`transformers` may write **dynamic modules** under `HF_MODULES_CACHE`. If `/home` is small, run with `HF_HOME` (and this script’s `--hf-home`) on a large filesystem; the case-study script also sets `HF_MODULES_CACHE` under that root when possible.

## Limitations

- **Default vs legacy:** with **SGLang-order** draft (default), the draft **does** use **record `last_hidden_state`** in `eh_proj` / `hnorm` on the first speculative step. **`--ids-only-draft`** reproduces the old “no record hidden” experiment; acceptance vs trace logits can still differ from **`NextNSglangCPUDraftAdapter`** (full layer) because the HF stack uses `from_pretrained` wiring and different numerics vs the manual CPU path.
- **Operational:** with **`first_k_dense_replace` aligned** to the Hub shard, layer 0 is built as **MoE** (256 routed experts + shared experts). Loading and **block-dequantizing** FP8 weights to `--dtype` needs very large **RAM** on CPU; prefer **`--device cuda --dtype bfloat16`** when possible.
- **FP8 in `config.json`:** Hub NextN often sets `quantization_config` with `quant_method: fp8`. On **CPU/MPS**, the adapter **removes** that attribute (see `nextn_full_layer_draft.py`) and **dequantizes** FP8 `*.weight` tensors using paired `*.weight_scale_inv` and `weight_block_size` from config so `nn.Linear` matches activation dtype.
- **Config vs shard:** `num_hidden_layers: 1` with `first_k_dense_replace: 3` would instantiate a **dense** MLP for layer 0 while the checkpoint only has **MoE** keys; the adapter forces `first_k_dense_replace=0` for that case so expert weights load.
- **Embeddings / LM head:** the NextN index often omits `model.embed_tokens` and `lm_head`; use **`--embed-head-aux-safetensors`** or ensure `DEFAULT_EMBED_HEAD_AUX_PATH` exists (materialize script). **`model.norm`** is filled from **`model.layers.0.shared_head.norm.weight`** in `nextn_layer_parameters.safetensors` when present.
- **VRAM / production:** even on GPU, this path is for **experimentation / comparison** only, not tuned for production.

# NextN structure, SGLang speculative decoding, and this repo (CPU)

## What is in the Hugging Face `lmsys/DeepSeek-R1-NextN` repo?

The published checkpoint is **not** “one monolithic LM” in the sense SGLang uses for MTP. The shard `nextn_layer_parameters.safetensors` contains **at least two** distinct weight groups:

| Weight group | Example keys | Role |
|--------------|--------------|------|
| **MTP fusion** | `model.layers.0.eh_proj`, `enorm`, `hnorm` | Fuse **token embed** with **main-model hidden** before the draft transformer block. |
| **Draft transformer block** | MLA (`q_*`, `kv_*`, …), MoE experts, `shared_experts`, … | One **full** `DeepseekV3DecoderLayer` (attention + MLP/MoE). |
| **Head** | `shared_head.norm`, `shared_head.head` (and/or global `model.norm` / `lm_head` after load) | Final RMSNorm + vocab projection. |

The generic HF remote module **`modeling_deepseek.py`** wires the **decoder block** into `DeepseekV3Model`. It **does not** define modules for `eh_proj` / `enorm` / `hnorm`, so those fusion tensors are **unused** by `from_pretrained` + forward of the stock CausalLM class (they often show up as `unexpected_keys`).

## How does SGLang use NextN for speculative sampling?

From [SGLang speculative decoding](https://docs.sglang.ai/advanced_features/speculative_decoding.html) and their implementation (`python/sglang/srt/models/deepseek_nextn.py`, class `DeepseekModelNextN`):

1. **Embed** the current draft token (`embed_tokens`).
2. **Fusion:** `eh_proj(cat(RMSNorm_e(embed), RMSNorm_h(target_hidden)))` — target hidden comes from the **main model** on the first draft step; later steps use the draft model’s hidden (EAGLE worker updates `spec_info.hidden_states` from `logits_output.hidden_states`).
3. **One decoder layer:** `DeepseekV2DecoderLayer` (same role as HF `DeepseekV3DecoderLayer` — self-attn + MLP/MoE).
4. **Final norm** (`shared_head.norm` in SGLang; aligns with patching `model.norm` from the shard when loading HF snapshots here).
5. **LM head** → logits (wrapper applies `LogitsProcessor`).

So SGLang MTP is **not** “fusion + one linear”; it is **fusion + a full transformer layer + norm + head**.

Typical server flags (minimal MTP) look like:

```bash
--speculative-algorithm EAGLE   # NEXTN is documented as an alias in some versions
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

(Exact names may vary slightly by SGLang version; see their DeepSeek docs.)

## What in this repo matches which part of that stack?

| Component | Class / script | Notes |
|-----------|----------------|--------|
| **Full MTP layer (manual CPU)** | `NextNSglangCPUDraftAdapter` | **Default** for `run_nextn_mtp_from_record_cpu.py`: fusion → MLA + MoE decoder → `shared_head.norm` → `lm_head`. Matches the NextN shard’s full block without loading all of HF `from_pretrained`. |
| **Fusion + `shared_head` shortcut (no decoder layer)** | `NextNMTPHeadDraftAdapter` / `MTPDraftAdapter._mtp_forward_batch` | Still defined in `models_draft.py` for advanced use; **not** selected by the record CLI scripts. |
| **SGLang NextN block (fusion → layer → norm → head)** | `NextNSglangStructureDraftAdapter` | **Logits** = ``lm_head(model.norm(layer(fused)))``. **Recurrence:** next step’s ``hnorm`` input is **post-``model.norm``** hidden (same tensor ``lm_head`` sees), like SGLang draft ``hidden_states``. Heavy on **CPU**; prefer **GPU** + bf16. |
| **Embed → layer(s) → norm → head on token ids only (no fusion)** | `NextNFullHuggingfaceDraftAdapter` | Ignores `eh_proj` / `enorm` / `hnorm` in forward — different graph from SGLang MTP. |

**`run_nextn_mtp_from_record_cpu.py`**

- Default: `NextNSglangCPUDraftAdapter` (full MTP layer on CPU).
- **`--sglang-draft-structure`:** `NextNSglangStructureDraftAdapter` (fusion + HF decoder layer + norm + lm_head).

In **live** SGLang, “main hidden” comes from the **target** forward. On record replay it comes from **`decode_state.last_hidden_state`** — valid if the trace matches the same R1-class stack.

### If `NextNSglangStructureDraftAdapter` logits differ from `NextNSglangCPUDraftAdapter`

Both run a **full** decoder block, but the HF stack (`from_pretrained`) vs the manual CPU path can still differ slightly (kernels, dequant, dtype). For **structure** debugging (norm copy, FP8 dequant, KV/masks), see: **`model.norm`** copied from the shard (`_try_copy_shared_head_norm_to_model_norm`), **embed / `lm_head`** aux aligned with the Hub layout, trace **`last_hidden`** from the same R1 stack, and **`_hf_past_kv_seq_len`** / 4D masks / draft-local **`position_ids`** (already fixed in-tree for single-layer KV).

## Optional: swap the draft decoder block for a **main R1** layer (CPU)

NextN’s `model.layers.0` is the **same HF module class** as a layer in `deepseek-ai/DeepSeek-R1-0528`. If you want to try **main-checkpoint** attention+MoE instead of NextN-trained layer weights:

1. **Indexing:** `num_hidden_layers == 61` ⇒ decoder weights are `model.layers.0` … `model.layers.60`. The last **full** decoder+MoE layer is **`model.layers.60`**. Names like `model.layers.61.shared_head.*` are **MTP/aux**, not a substitute for a full layer.
2. **Materialize:** `scripts/materialize_r1_decoder_layer_as_nextn_layer0.py` copies `model.layers.<K>.*` → `model.layers.0.*` into one `.safetensors` (default `--layer-index 60`). Same Hub shard download pattern as the embed/head materializer; **very large** RAM/disk.
3. **Load:** `NextNFullHuggingfaceDraftAdapter` / `NextNSglangStructureDraftAdapter` accept `decoder_layer0_override_safetensors=...` (applied after NextN `from_pretrained` + FP8 dequant). Case study: `--decoder-layer0-override-safetensors /path/to/file.safetensors`.

**Caveat:** Record `last_hidden` is still the residual **after the full 61-layer base**, while the draft applies **one** layer on `fused(embed, hnorm(record))`. Swapping weights to R1 layer 60 does **not** make the draft equivalent to that full forward; it only changes which single-layer map you use.

## Flag mapping (rough)

| SGLang (server) | This repo (`run_nextn_mtp_from_record_cpu.py`) |
|-----------------|-----------------------------------------------|
| `speculative-num-steps` | `--depth` (draft autoregressive steps per round) |
| `speculative-eagle-topk` | `--top-k` (when not `--draft-mtp-greedy`) |
| `speculative-num-draft-tokens` | Related to tree size; we use `--max-paths` + depth/top-k to cap beams |
| Target forward | **Trace replay** base (`TraceReplayBaseAdapter`) — not a live 671B run |
| Full SGLang NextN block order | `--sglang-draft-structure` |

## Discoverability alias (`SglangStyleNextNMTPDraftAdapter`)

`:class:`SglangStyleNextNMTPDraftAdapter`` in `models_draft.py` is an **alias** of `NextNMTPHeadDraftAdapter` — the **fusion + head shortcut** only. For the name **“SGLang”** in the sense of **`DeepseekModelNextN`**, use **`NextNSglangStructureDraftAdapter`** or **`--sglang-draft-structure`** above.

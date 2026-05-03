# Gemma 4 26B-A4B Source Of Truth

Target checkpoint: `google/gemma-4-26B-A4B`

Checkpoint revision: `64143b04706fadeb2f8ac198f7ecab57b94b1e0b`

Primary source implementation: HuggingFace Transformers `v5.5.0`, commit `c1c34249fa27deefbd4a377dfbf883a39baf5c6d`

Serving mode for this bringup: text-only prefill plus batch=1 decode, with decode under TTNN trace replay.

Out of first-milestone scope: image/video/audio towers, bidirectional multimodal masking, dynamic batching, device-side sampling beyond greedy/top-k already used by the demo, and packed BFP4 expert compute.

## Checkpoint Config

The selected checkpoint config is the authority over defaults.

Source: `https://huggingface.co/google/gemma-4-26B-A4B/blob/64143b04706fadeb2f8ac198f7ecab57b94b1e0b/config.json`

Text config:

| Field | Value |
| --- | --- |
| dtype | `bfloat16` |
| vocab_size | `262144` |
| hidden_size | `2816` |
| num_hidden_layers | `30` |
| num_attention_heads | `16` |
| num_key_value_heads | `8` for sliding attention |
| num_global_key_value_heads | `2` for full attention |
| head_dim | `256` for sliding attention |
| global_head_dim | `512` for full attention |
| sliding_window | `1024` |
| layer_types | `[sliding_attention x5, full_attention] x5` |
| rms_norm_eps | `1e-6` |
| hidden_activation | `gelu_pytorch_tanh` |
| intermediate_size | `2112` |
| enable_moe_block | `true` |
| num_experts | `128` |
| top_k_experts | `8` |
| moe_intermediate_size | `704` |
| attention_k_eq_v | `true` |
| num_kv_shared_layers | `0` |
| final_logit_softcapping | `30.0` |
| tie_word_embeddings | `true` |

Checkpoint index metadata: `26,544,131,376` total parameters, `51,611,872,412` bytes across `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`.

## Semantic Sources

| Area | Source lines | Contract |
| --- | --- | --- |
| Text config defaults | `configuration_gemma4.py` lines 87-201 | Text config fields, 5:1 sliding/full pattern, default RoPE dictionaries. |
| Top-level config | `configuration_gemma4.py` lines 264-347 | `Gemma4Config` contains text, vision, and optional audio sub-configs. |
| RMSNorm | `modeling_gemma4.py` lines 157-175 | `x.float() * (mean(x^2)+eps)^-0.5`, optional scale, cast to input dtype. |
| RoPE helper | `modeling_gemma4.py` lines 727-754 | `rotate_half`, then `x*cos + rotate_half(x)*sin`. |
| Text RoPE | `modeling_gemma4.py` lines 1035-1122 | Per-layer-type RoPE; full attention passes `head_dim_key="global_head_dim"` for proportional RoPE. |
| Proportional RoPE | `modeling_rope_utils.py` lines 187-254 | Full-layer partial rotation: zero inverse frequencies leave non-rotary channels unchanged. |
| Attention | `modeling_gemma4.py` lines 1125-1240 | Sliding/full attention, per-head Q/K/V norms, K=V on global layers, cache update, SDPA. |
| Eager attention math | `modeling_gemma4.py` lines 768-799 | Repeat KV groups, score scale `1.0`, add mask, fp32 softmax, matmul V. |
| Experts | `modeling_gemma4.py` lines 1243-1280 | Packed `gate_up_proj`, GeGLU expert compute, top-k weighted `index_add_`. |
| Router | `modeling_gemma4.py` lines 1283-1316 | RMSNorm without weight, learned scale, softmax over all experts, top-k, sum-normalize, per-expert scale. |
| Decoder layer | `modeling_gemma4.py` lines 1319-1403 | Attention residual, shared dense MLP, routed MoE branch from pre-FF residual, layer scalar. |
| Text model | `modeling_gemma4.py` lines 1490-1620 | Embedding scale, DynamicCache creation, mask mapping, per-layer RoPE, final norm. |
| Output head | `modeling_gemma4.py` lines 2400-2451 | Tied LM head and `tanh(logits / 30.0) * 30.0` softcap. |

## Red-Zone Inventory

| Semantic op | Why red-zone | Required source behavior |
| --- | --- | --- |
| `attention_decode` | Mixed sliding/full layers with different head dims and KV head counts | Sliding: 16 Q heads, 8 KV heads, head_dim 256, sliding window 1024. Full: 16 Q heads, 2 KV heads, head_dim 512, K=V. |
| `rope_by_layer_type` | Full attention uses proportional partial RoPE | Full layers rotate only 25% of 512-dim head channels; remaining channels must be identity. |
| `kv_cache_update` | Decode trace mutates persistent cache | Non-shared layers update by `layer_idx`; this checkpoint has no KV-shared layers. |
| `router` | Gemma is softmax-first, not GPT-OSS topk-first | Top-k weights are selected from full softmax probabilities and normalized by sum, not softmaxed again. |
| `expert_compute` | Experts are active-only and packed differently from dense MLP | `gate_up_proj` first half is gate, second half up; activation is GELU tanh approximation. |
| `shared_mlp_plus_moe` | Dense MLP and MoE both run in every layer | Dense branch and expert branch are separately normed, summed, normed again, then residual-added. |
| `lm_head_softcap` | Full vocab is large and logits are softcapped | Use tied embedding weight, BF16 quality path, softcap before sampling. |
| `trace_decode` | TTNN trace requires static tensor addresses | Host creates fixed-shape decode embeddings/positions; trace copies into persistent device buffers and replays. |

## Source Disagreements

The local Python environment initially had Transformers `4.57.1`, which does not recognize `model_type: gemma4`. The checkpoint config declares `transformers_version: 5.5.0.dev0`; the pinned public `v5.5.0` tag contains the Gemma4 source used above. The TTNN runtime must not rely on importing `transformers.models.gemma4` from the installed environment.

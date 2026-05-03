# Gemma 4 26B-A4B Model Dossier

## Source Selection

| Item | Selection |
| --- | --- |
| Target model | `google/gemma-4-26B-A4B` |
| Checkpoint | HuggingFace BF16 safetensors |
| Checkpoint config | HF commit `64143b04706fadeb2f8ac198f7ecab57b94b1e0b` |
| Primary source implementation | Transformers `v5.5.0` commit `c1c34249fa27deefbd4a377dfbf883a39baf5c6d` |
| Secondary source implementation | Local `models/demos/gemma4` TTNN scaffold |
| Serving mode | Text-only single-user prefill plus traced decode, batch=1 |
| First target | Full 30-layer decode path on 8x Blackhole p150b, correctness-first BF16 weights where practical |

## Architecture Skeleton

| Area | Source lines | Contract summary | First milestone scope |
| --- | --- | --- | --- |
| Embedding | `modeling_gemma4.py` 1406-1417, 1504-1507 | Token embedding scaled by `sqrt(hidden_size)`; tied to LM head. | Required. |
| Attention | `modeling_gemma4.py` 1125-1240 | Mixed 5 sliding + 1 full. Per-head Q/K/V norms. Full layers use K=V and 2 KV heads. | Required for decode. |
| Cache | `modeling_gemma4.py` 1197-1220, 1571-1577 | DynamicCache update by layer; position ids advance from past length. | Required for long decode. |
| RoPE | `modeling_gemma4.py` 1035-1122; `modeling_rope_utils.py` 187-254 | Sliding default RoPE, full proportional partial RoPE. | Required. |
| MLP/MoE | `modeling_gemma4.py` 1016-1032, 1243-1316 | Shared GeGLU MLP plus 128-expert top-8 routed branch. | Required. |
| Norm/residual | `modeling_gemma4.py` 157-175, 1319-1403 | RMSNorms before/after attention and feedforward; layer scalar after all layer work. | Required. |
| Output head | `modeling_gemma4.py` 2400-2451 | Tied LM head, optional logits slice, softcap 30.0. | Required. |
| Optional heads | `modeling_gemma4.py` 2052-2307 | Vision/video/audio soft-token paths and bidirectional multimodal masks. | Out of scope for text-only milestone. |
| Quantization | HF index metadata | Source weights BF16; local TT path may store BFP8/BFP4 later. | BF16/dequant correctness path first. |
| Distributed reference behavior | `configuration_gemma4.py` 125-135 | HF TP plan is colwise Q/K/V/MLP gate/up, rowwise O/down. | Use as topology hint only. |

## Red-Zone Decisions

| Semantic op | Why unusual | Reference function | First lowering | Final-risk guess |
| --- | --- | --- | --- | --- |
| `attention_decode` | Sliding/full variants, K=V global layers, per-head norms | `Gemma4TextAttention.forward` | TTNN decomposed | Medium: Blackhole SDPA program configs and head_dim 512 L1 pressure. |
| `rope_cache` | Proportional partial full-layer RoPE | `Gemma4TextRotaryEmbedding.forward` | CPU cache generation plus TTNN embedding lookup | Medium: identity channels must stay exact. |
| `router` | Full-softmax then top-k sum-renormalization | `Gemma4TextRouter.forward` | TTNN decomposed | High: current generic fused GPT router is semantically wrong. |
| `expert_compute` | Packed expert weights and active-only routing | `Gemma4TextExperts.forward` | `ttnn.sparse_matmul` | High: replication vs TP sharding is memory/perf critical. |
| `decoder_layer` | Shared dense MLP and MoE both contribute | `Gemma4TextDecoderLayer.forward` | TTNN decomposed | Medium: many norms and deallocations inside trace. |
| `trace_decode` | Mutates KV cache under static replay | TTNN trace APIs plus demo loop | Existing demo trace pattern | High until validated on Blackhole. |

## First-Target Choice

Decode first is selected because the acceptance criteria prioritize single-user latency and traced long decode. Prefill remains required for TTFT and cache initialization, but the optimization target is decode replay.

Excluded from the first target: multimodal tokens, BFP4 packed expert compute, dynamic batching, non-greedy sampling quality work, and full max context. These exclusions do not change the text layer function for ordinary prompts.

## Cache And State Contracts

| State | Producer | Consumer | Prefill shape/meaning | Decode shape/meaning | Placement candidate | Trace mutability |
| --- | --- | --- | --- | --- | --- | --- |
| KV cache | Attention layer | SDPA decode | Per non-shared layer K/V cache over prompt positions | Persistent paged cache updated at `current_pos` | Device DRAM, per-layer, replicated/TP-local according to KV ownership | Mutable in trace. |
| Position ids | Host/demo | RoPE embedding lookup and cache update | `[1, seq]` logical positions | `[1, 32]` uint32 padded for RoPE lookup plus `[1]` int32 for cache update | Host-created, copied to trace buffers | Contents mutable, address fixed. |
| Page table | Demo | Paged attention | Identity table for batch=1 pages | Same persistent tensor | Device DRAM replicated | Stable. |
| Router probabilities | Router | Top-k and expert dispatch | `[1,1,S,128]` transient | `[1,1,1,128]` transient | Device L1/DRAM as op requires | Allocated inside trace unless fused. |
| Expert routing weights | Router scatter | Sparse matmul | Dense top-k weights over 128 experts | Dense top-k weights over 128 experts | Device tensor, future active-only packed metadata | Transient. |

## Weight Map

| Group | Source key pattern | Logical shape | Physical TTNN layout | Quantization | First milestone handling |
| --- | --- | --- | --- | --- | --- |
| Embedding/LM head | `model.language_model.embed_tokens.weight` | `[262144, 2816]` | Embedding hidden-sharded; LM head vocab-sharded | BF16 | Required. |
| Attention Q | `layers.*.self_attn.q_proj.weight` | Sliding `[4096,2816]`, full `[8192,2816]` | Fused QKV colwise | BF16 first | Required. |
| Attention K/V | `k_proj.weight`, optional `v_proj.weight` | Sliding K/V `[2048,2816]`, full K `[1024,2816]`, V tied to K | Fused with Q; global duplicates K as V | BF16 first | Required. |
| Attention output | `o_proj.weight` | Sliding `[2816,4096]`, full `[2816,8192]` | Rowwise sharded | BF16 first | Required. |
| Norms | `*.weight` | Hidden or head-dim vectors | Replicated | BF16 | Required. |
| Shared MLP | `mlp.gate_proj/up_proj/down_proj.weight` | `[2112,2816]`, `[2112,2816]`, `[2816,2112]` | Gate/up colwise, down rowwise | BF16 first | Required. |
| Router | `router.scale`, `router.proj.weight`, `router.per_expert_scale` | `[2816]`, `[128,2816]`, `[128]` | Replicated | BF16 | Required. |
| Experts | `experts.gate_up_proj`, `experts.down_proj` | `[128,1408,2816]`, `[128,2816,704]` | Sparse-matmul expert layout | BF16 first; BFP8/BFP4 later | Required. |
| Layer scalar | `layers.*.layer_scalar` | `[1]` | Scalar constant | BF16/source dtype | Required. |

## Common Reuse Decision

| Helper | Reuse directly | Use as pattern only | Do not use | Reason |
| --- | --- | --- | --- | --- |
| `attention_1d.py` |  | yes |  | Llama/Meta RoPE and weight assumptions do not match Gemma4. |
| `mlp_1d.py` |  | yes |  | Gemma has shared GeGLU plus MoE in every layer. |
| `tt_ccl.py` |  | yes |  | CCL patterns are useful, but Gemma current code has explicit TP allreduce helpers. |
| `LazyWeight` | yes |  |  | Good fit for future manifest/cache cleanup. |
| validation helpers | yes |  |  | PCC and tensor-contract checks should be reused. |
| distribution helpers | yes |  |  | Mesh mapping should stay explicit and testable. |

## Custom-Op Candidates

| Semantic op | Decomposed reference exists? | Contract-changing? | Existing TTNN op to study | Ops-team question |
| --- | --- | --- | --- | --- |
| Router softmax/topk/sum-normalize/scatter | yes | no initially | GPT-OSS `topk_router_gpt` | Can a Gemma router op support softmax-first and sum renorm? |
| Expert dispatch/compute/combine | yes | maybe | `ttnn.sparse_matmul`, GPT-OSS MoE fused ops | Should decode use dense routing tensor or compact active-expert metadata? |
| Global attention K=V/head_dim512 decode | yes | no | SDPA decode program configs | What Blackhole grid keeps head_dim512 within L1 without underutilizing? |
| Trace allocation planning | partial | no | TTNN mesh trace allocator | Does Gemma decode need nonzero trace region size on 8x p150b? |

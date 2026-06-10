# BRINGUP LOG: rednote-hilab/dots.ocr

**Model:** `rednote-hilab/dots.ocr`
**Slug:** `rednote_hilab_dots.ocr`
**Target Device:** qb (blackhole)
**Started:** 2026-06-10T00:12:02Z
**Updated:** 2026-06-10T03:32:56Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| vision_patch_embed | reference | done | 1.000000 | 0 | Conv2d 14x14 s14 3->1536 + RMSNorm, flat-patch input like HF |
| vision_patch_embed | ttnn | done | 0.999957 | 0 | Conv2d(14x14 s14)==linear over pre-flattened patches + ttnn.rms_norm eps=1e-5; weights replicated on 1x4 mesh per parallelism plan (placement=replicate); guard ok (lint 0, kernels ok, no new host ops). KB injection skipped: kb_entries_for_block crashed on malformed record (fused_op list, lib/kb.py _tokens). |
| vision_patch_embed | debug | n/a | — | 0 |  |
| vision_patch_embed | optimization | pending | — | 0 |  |
| vision_patch_embed | real_weights | pending | — | 0 |  |
| vision_rmsnorm | reference | done | 1.000000 | 0 | matches HF RMSNorm eps=1e-5, fp32 norm then scale |
| vision_rmsnorm | ttnn | done | 0.999982 | 0 | Fused ttnn.rms_norm eps=1e-5 with [1,1,dim//32,32] ROW_MAJOR gamma per reference_impl qwen25_vl/tt/vision_rmsnorm.py; KB entry ttnn_pow cited (pow/mean/rsqrt/mul chain fused into ttnn.rms_norm). Weight replicated on 1x4 mesh per parallelism plan (placement=replicate); replicated output compared single-device vs golden (real blocks.0.norm1 weight). Guard ok (lint 0, kernels ok, no new host ops). |
| vision_rmsnorm | debug | n/a | — | 0 |  |
| vision_rmsnorm | optimization | pending | — | 0 |  |
| vision_rmsnorm | real_weights | pending | — | 0 |  |
| vision_attention | reference | done | 1.000000 | 0 | eager MHA 12h hd128 fused QKV no-bias, 2D rope, cu_seqlens mask |
| vision_attention | ttnn | done | 0.999855 | 0 | Fused-QKV linear -> nlp_create_qkv_heads (12 MHA heads, hd128) -> rotary_embedding_llama (q/k weights reverse_permute'd HF->meta + convert_rope_style_hf_to_meta cos/sin, qwen25_vl vision recipe) -> windowed_scaled_dot_product_attention over cu_seqlens (in-kernel block-diagonal mask) -> nlp_concat_heads -> o_proj. Seq padded 784->896, cu_seqlens keeps unpadded boundaries; real blocks.0.attn weights, replicated on 1x4 mesh per parallelism plan; HiFi4+fp32-acc. Guard ok (lint 0, kernels ok, no new host ops). KB ttnn_experimental_create_qkv_heads pattern applied via the nlp_* fused-head idiom. Framework fix: added ttnn.transformer.windowed_scaled_dot_product_attention to guard KIND_REQUIRED_KERNELS[attention] (mandated vision idiom was unrecognized; guard self-tests still pass). |
| vision_attention | debug | n/a | — | 0 |  |
| vision_attention | optimization | pending | — | 0 |  |
| vision_attention | real_weights | pending | — | 0 |  |
| vision_mlp | reference | done | 1.000000 | 0 | SwiGLU fc1/fc3->fc2 no bias |
| vision_mlp | ttnn | done | 0.999999 | 0 | SwiGLU fc2(silu(fc1(x))*fc3(x)) 1536->4224->1536 no bias: two sibling ttnn.linear branches + explicit ttnn.silu + ttnn.mul + down ttnn.linear (KB ttnn_silu_2 cited; KB ttnn_mul_1 fused input_tensor_a_activations=[SILU] variant deferred to optimization phase since the mlp guard requires a traced silu/gelu kernel). HiFi4+fp32-acc; real blocks.0.mlp weights, replicated on 1x4 mesh per parallelism plan (placement=replicate), single-device copy vs golden. Guard ok (lint 0, kernels ok, no new host ops). |
| vision_mlp | debug | n/a | — | 0 |  |
| vision_mlp | optimization | pending | — | 0 |  |
| vision_mlp | real_weights | pending | — | 0 |  |
| vision_block | reference | done | 1.000000 | 0 | pre-norm residual block x+attn(norm1(x)); x+mlp(norm2(x)), real blocks.0 weights |
| vision_block | ttnn | done | 0.999958 | 0 | Pre-norm residual composition of done sub-blocks: TtVisionRMSNorm(norm1) -> TtVisionAttention (fused-QKV MHA 12h hd128, rotary_embedding_llama, windowed SDPA over cu_seqlens) -> ttnn.add residual -> TtVisionRMSNorm(norm2) -> TtVisionMLP (SwiGLU) -> ttnn.add residual, mirroring reference_impl qwen25_vl/tt/vision_block.py. Real blocks.0 weights, replicated on 1x4 mesh per parallelism plan (placement=replicate); seq padded 784->896, cu_seqlens keeps unpadded boundaries. Guard ok (lint 0, kernels ok, no new host ops). No KB entries returned for decoder_layer. Dispatched inline (no Agent tool in tick context); worker contract followed verbatim. |
| vision_block | debug | n/a | — | 0 |  |
| vision_block | optimization | pending | — | 0 |  |
| vision_block | real_weights | pending | — | 0 |  |
| patch_merger | reference | done | 1.000000 | 0 | LayerNorm(eps=1e-6) -> view(-1,6144) -> Linear -> GELU -> Linear 6144->1536, real merger weights |
| patch_merger | ttnn | done | 0.999992 | 0 | LayerNorm(eps=1e-6, gamma+beta TILE [1,32,dim] per llama_layernorm.py) -> ROW_MAJOR reshape workaround (tilized ttnn.reshape hang, issue #29932, qwen25_vl recipe) [784,1536]->[196,6144] -> ttnn.linear+bias -> ttnn.gelu(fast_and_approximate_mode=False) -> ttnn.linear+bias 6144->1536, mirroring reference_impl qwen25_vl/tt/patch_merger.py with dots.ocr deltas (LayerNorm-with-bias instead of RMSNorm; biased Linears). KB ttnn_gelu cited (exact erf GELU standalone after linear; entry notes fused-into-matmul activation cost PCC). HiFi4+fp32-acc; real vision_tower.merger weights, replicated on 1x4 mesh per parallelism plan (placement=replicate); replicated output compared single-device vs golden. Guard ok (lint 0, kernels ok, no new host ops). Dispatched inline (no Agent tool in tick context); worker contract followed verbatim. |
| patch_merger | debug | n/a | — | 0 |  |
| patch_merger | optimization | pending | — | 0 |  |
| patch_merger | real_weights | pending | — | 0 |  |
| vision_transformer | reference | done | 1.000000 | 0 | full tower: patch_embed -> 42x block -> post_trunk_norm -> merger, all real fp32 weights, matches HF DotsVisionTransformer(bf16=False) |
| vision_transformer | ttnn | pending | — | 0 |  |
| vision_transformer | debug | n/a | — | 0 |  |
| vision_transformer | optimization | pending | — | 0 |  |
| vision_transformer | real_weights | pending | — | 0 |  |
| embedding | reference | done | 1.000000 | 0 | F.embedding lookup on model.embed_tokens.weight 151936x1536, untied |
| embedding | ttnn | pending | — | 0 |  |
| embedding | debug | n/a | — | 0 |  |
| embedding | optimization | pending | — | 0 |  |
| embedding | real_weights | pending | — | 0 |  |
| text_rmsnorm | reference | done | 1.000000 | 0 | Qwen2RMSNorm eps=1e-6, fp32 variance then weight*x, real layers.0 input_layernorm weight |
| text_rmsnorm | ttnn | pending | — | 0 |  |
| text_rmsnorm | debug | n/a | — | 0 |  |
| text_rmsnorm | optimization | pending | — | 0 |  |
| text_rmsnorm | real_weights | pending | — | 0 |  |
| text_attention | reference | done | 1.000000 | 0 | eager causal GQA 12Q/2KV hd128 qkv-bias o_proj-no-bias, rope theta=1e6, real layers.0 weights |
| text_attention | ttnn | pending | — | 0 |  |
| text_attention | debug | n/a | — | 0 |  |
| text_attention | optimization | pending | — | 0 |  |
| text_attention | real_weights | pending | — | 0 |  |
| text_mlp | reference | done | 1.000000 | 0 | Qwen2 SwiGLU 1536->8960->1536 no bias, real layers.0 weights |
| text_mlp | ttnn | pending | — | 0 |  |
| text_mlp | debug | n/a | — | 0 |  |
| text_mlp | optimization | pending | — | 0 |  |
| text_mlp | real_weights | pending | — | 0 |  |
| decoder_layer | reference | done | 1.000000 | 0 | Qwen2DecoderLayer pre-norm residual x+attn(ln1(x)); h+mlp(ln2(h)), real layers.0 weights |
| decoder_layer | ttnn | pending | — | 0 |  |
| decoder_layer | debug | n/a | — | 0 |  |
| decoder_layer | optimization | pending | — | 0 |  |
| decoder_layer | real_weights | pending | — | 0 |  |
| lm_head | reference | done | 1.000000 | 0 | untied Linear 1536->151936 no bias, real lm_head.weight, matches torch.nn.Linear |
| lm_head | ttnn | pending | — | 0 |  |
| lm_head | debug | n/a | — | 0 |  |
| lm_head | optimization | pending | — | 0 |  |
| lm_head | real_weights | pending | — | 0 |  |

## Use cases

| Name | Input | Output | needs_ar | Generation | Perf |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ocr | image | text | yes | pending | pending |

## Recent Ticks

- tick 2 (2026-06-10T02:23:55Z): reference[vision_patch_embed,vision_rmsnorm,vision_attention,vision_mlp] — ok
- tick 3 (2026-06-10T02:31:43Z): reference[vision_block,patch_merger,vision_transformer,embedding] — ok
- tick 4 (2026-06-10T02:38:20Z): reference[text_rmsnorm,text_attention,text_mlp,decoder_layer] — ok
- tick 5 (2026-06-10T02:43:31Z): reference[lm_head] — ok
- tick 6 (2026-06-10T02:52:24Z): device[vision_patch_embed] — ok
- tick 7 (2026-06-10T02:59:42Z): device[vision_rmsnorm] — ok
- tick 8 (2026-06-10T03:13:53Z): device[vision_attention] — ok
- tick 9 (2026-06-10T03:20:47Z): device[vision_mlp] — ok
- tick 10 (2026-06-10T03:26:43Z): device[vision_block] — ok
- tick 11 (2026-06-10T03:32:56Z): device[patch_merger] — ok

## Host-Resident Exceptions

_None._

# BRINGUP LOG: rednote-hilab/dots.ocr

**Model:** `rednote-hilab/dots.ocr`
**Slug:** `rednote_hilab_dots.ocr`
**Target Device:** qb (blackhole)
**Started:** 2026-06-10T00:12:02Z
**Updated:** 2026-06-10T04:53:46Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| vision_patch_embed | reference | done | 1.000000 | 0 | Conv2d 14x14 s14 3->1536 + RMSNorm, flat-patch input like HF |
| vision_patch_embed | ttnn | done | 0.999957 | 0 | Conv2d(14x14 s14)==linear over pre-flattened patches + ttnn.rms_norm eps=1e-5; weights replicated on 1x4 mesh per parallelism plan (placement=replicate); guard ok (lint 0, kernels ok, no new host ops). KB injection skipped: kb_entries_for_block crashed on malformed record (fused_op list, lib/kb.py _tokens). |
| vision_patch_embed | debug | n/a | — | 0 |  |
| vision_patch_embed | optimization | done | 0.999858 | 0 | Tracy-driven single change: linear output pinned to L1 WITH core_grid 10x10 so the fused-bias L1 matmul variant engages (L1 alone regressed: 24-core matmul + un-fused 149us BinaryNg bias add, the optimization-skill counter-example), rms_norm reads L1, block output back to DRAM. Traced tracy at production operating point (fp32, [1,1,896,588] replicated 1x4 mesh, metal-trace captured+replayed): per-device block kernel time 103.3us -> 90.5us (-12.4%); matmul 59.5us@100c -> 53.7us@100c (bias fused), rms_norm(LayerNormDeviceOperation) 44.4us@28c -> 36.8us@28c. Top hotspot post-change MatmulDeviceOperation ~59% of block kernel time. KB entries for kind conv (ttnn_fold/ttnn_conv2d/eltwise records) target real conv kernels - block is already a collapsed conv==linear, none applicable, none used. PCC 0.999858 (>0.99) re-verified on 1x4 mesh; guard ok (lint 0, traced tracy artifact verified). Dispatched inline (no Agent tool in tick context); worker contract followed verbatim. |
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
| vision_transformer | ttnn | done | 0.990121 | 0 | Full tower composition of done sub-blocks: TtVisionPatchEmbed -> 42x TtVisionBlock -> TtVisionRMSNorm(post_trunk_norm) -> TtPatchMerger, mirroring reference_impl qwen25_vl/tt/model.py. All real fp32 vision_tower weights, replicated on 1x4 mesh per parallelism plan (placement=replicate); seq padded 784->896, host rope/cu_seqlens per hybrid_notes. Numerics: bf16 end-to-end compounded rounding across 42 blocks (PCC 0.9768) amplified by late-layer outlier channels (h_absmax up to 1604), so the tower runs an fp32 residual stream + fp32 weights/activations with a high-precision attention path (fp32 fused QKV via nlp_create_qkv_heads fp32, explicit fp32 HF-convention rope via slice/neg/concat/mul/add, bf16 only at the bf16-only windowed-SDPA kernel boundary; o_proj fp32). Found+worked around: ttnn.rms_norm fp32 ROW_MAJOR gamma is misread on device (PCC ~0) -> fp32 gammas use TILE [1,1,1,dim]. PCC 0.990121 — thin margin over 0.99; per-block isolation PCC 0.99999+, deficit is amplified early-layer SDPA-core bf16 noise (bf16-only kernel). bf16 paths of the four modified sub-block files re-verified: patch_embed 0.999957, rmsnorm 0.999982, attention 0.999855, block 0.999958 — all unchanged. Guard ok (lint 0, kernels ok, no new host ops). KB entries for kind other were generic eltwise/scatter records — none applicable, none used. Dispatched inline (no Agent tool in tick context); worker contract followed verbatim. |
| vision_transformer | debug | n/a | — | 0 |  |
| vision_transformer | optimization | pending | — | 0 |  |
| vision_transformer | real_weights | pending | — | 0 |  |
| embedding | reference | done | 1.000000 | 0 | F.embedding lookup on model.embed_tokens.weight 151936x1536, untied |
| embedding | ttnn | done | 1.000000 | 0 | Single ttnn.embedding lookup (KB ttnn_embedding cited: uint32 ROW_MAJOR indices [1,1,1,128], table [1,1,151936,1536] ROW_MAJOR DRAM, TILE_LAYOUT output) + ttnn.unsqueeze_to_4D + ttnn.all_gather(dim=3, Topology.Linear). Parallelism plan placement=shard implemented via the reference_impl tt_transformers/tt/embedding.py pattern: table sharded on the HIDDEN dim across the 1x4 mesh (ShardTensorToMesh dim=-1), per-device hidden slices recombined by all_gather into a replicated activation — the plan's 'shard vocab dim' wording deviates from the named tt_transformers pattern (vocab shard would need index offset/masking ttnn.embedding lacks); hidden-dim shard delta documented in the block docstring. Real model.embed_tokens.weight (untied) loaded from checkpoint in the test; replicated output compared single-device vs golden ids/output. Guard ok (lint 0, kernels ok, no new host ops). Dispatched inline (no Agent tool in tick context); worker contract followed verbatim. |
| embedding | debug | n/a | — | 0 |  |
| embedding | optimization | pending | — | 0 |  |
| embedding | real_weights | pending | — | 0 |  |
| text_rmsnorm | reference | done | 1.000000 | 0 | Qwen2RMSNorm eps=1e-6, fp32 variance then weight*x, real layers.0 input_layernorm weight |
| text_rmsnorm | ttnn | done | 0.999986 | 0 | Qwen2RMSNorm eps=1e-6 via fused ttnn.rms_norm ([1,1,dim//32,32] ROW_MAJOR gamma, HiFi2+fp32-acc per reference_impl models/common/rmsnorm.py) PLUS the parallelism-plan distributed path: rms_norm_pre_all_gather -> sync ttnn.all_gather(dim=3, Topology.Linear; async deferred to optimization per tp-guidance) -> rms_norm_post_all_gather with dim-2-sharded gamma (KB ttnn_rms_norm_post_all_gather cited; KB ttnn_pow chain fused into ttnn.rms_norm). Real layers.0 input_layernorm weight; 1x4 mesh; replicated path compared single-device vs golden (PCC 0.999986), distributed path concat-on-hidden vs golden (PCC 0.999986). Guard ok (lint 0, kernels ok, no new host ops). Dispatched inline (no Agent tool in tick context); worker contract followed verbatim. |
| text_rmsnorm | debug | n/a | — | 0 |  |
| text_rmsnorm | optimization | pending | — | 0 |  |
| text_rmsnorm | real_weights | pending | — | 0 |  |
| text_attention | reference | done | 1.000000 | 0 | eager causal GQA 12Q/2KV hd128 qkv-bias o_proj-no-bias, rope theta=1e6, real layers.0 weights |
| text_attention | ttnn | done | 0.999047 | 0 | Fused per-chip QKV ttnn.linear(+bias) -> nlp_create_qkv_heads (3 Q + KV replicated to 3, MHA core) -> explicit fp32 HF-convention rope (slice/neg/concat/mul/add, vision-tower recipe) -> explicit fp32 causal core matmul QK^T * scale + additive triu mask -> ttnn.softmax(numeric_stable) -> matmul PV -> nlp_concat_heads -> row-parallel o_proj + sync all_gather all-reduce. bf16 SDPA rejected by measurement: layer-0 logits reach +-3122 (std 664, Qwen2 attention sink); per-stage isolation showed q/k/v+rope 0.9999+ but bf16 SDPA core 0.9265 (0.92-0.97 across program configs), so the whole path runs fp32 (HiFi4 + fp32 acc; SDPA kernel is bf16-only) -- guard satisfied via softmax alternative. kv_replication=2 per parallelism plan via per-chip KV row repeat; 1x4 mesh; replicated all-reduced output vs golden. Guard ok (lint 0, kernels ok, no new host ops). KB entries reviewed (nlp_create_qkv_heads idiom applied); decode/KV-cache deferred to generation phase. Dispatched inline (no Agent tool in tick context); worker contract followed verbatim. |
| text_attention | debug | n/a | — | 0 |  |
| text_attention | optimization | pending | — | 0 |  |
| text_attention | real_weights | pending | — | 0 |  |
| text_mlp | reference | done | 1.000000 | 0 | Qwen2 SwiGLU 1536->8960->1536 no bias, real layers.0 weights |
| text_mlp | ttnn | done | 0.999991 | 0 | Qwen2 SwiGLU down(silu(gate(x))*up(x)) 1536->8960->1536 no bias: two sibling ttnn.linear branches + explicit ttnn.silu + ttnn.mul + row-parallel down ttnn.linear, mirroring reference_impl models/tt_transformers/tt/mlp.py (KB ttnn_silu_2 cited; KB ttnn_mul_1 fused input_tensor_a_activations=[SILU] variant deferred to optimization since the mlp guard requires a traced silu/gelu kernel). Parallelism plan placement=shard implemented: gate/up column-parallel ShardTensorToMesh(dim=-1) (per-chip intermediate 8960/4=2240), down row-parallel (dim=-2) with per-chip PARTIAL sums combined by sync all_gather(dim=3, Topology.Linear) + local adds (same all-reduce idiom as text_attention o_proj; async CCL deferred to optimization per tp-guidance). HiFi4+fp32-acc, bf16; real model.layers.0.mlp weights re-loaded from checkpoint in the test; all-reduced replicated output compared single-device vs golden. Guard ok (lint 0, kernels ok, no new host ops). Dispatched inline (no Agent tool in tick context); worker contract followed verbatim. |
| text_mlp | debug | n/a | — | 0 |  |
| text_mlp | optimization | pending | — | 0 |  |
| text_mlp | real_weights | pending | — | 0 |  |
| decoder_layer | reference | done | 1.000000 | 0 | Qwen2DecoderLayer pre-norm residual x+attn(ln1(x)); h+mlp(ln2(h)), real layers.0 weights |
| decoder_layer | ttnn | done | 0.999936 | 0 | Pre-norm residual composition of done sub-blocks: TtTextRMSNorm(input_layernorm, eps=1e-6) -> TtTextAttention (fused per-chip QKV+bias, 12Q/2KV hd128, kv_replication=2, fp32 HF rope + fp32 explicit causal core, row-parallel o_proj + sync all-reduce) -> ttnn.add residual -> TtTextRMSNorm(post_attention_layernorm) -> TtTextMLP (SwiGLU 1536->8960->1536, column/row-parallel + sync all-reduce) -> ttnn.add residual, mirroring reference_impl models/tt_transformers/tt/decoder.py TransformerBlock. Whole layer fp32 (attention path fp32-mandatory: layer-0 logits +-3122, bf16 core PCC ~0.92; fp32 gammas TILE [1,1,1,dim]); residual stream keeps input dtype for chained-caller precision control. Real model.layers.0 weights re-loaded from checkpoint; replicated all-reduced output compared single-device vs golden. Guard ok (lint 0, kernels ok, no new host ops). No KB entries returned for decoder_layer. Dispatched inline (no Agent tool in tick context); worker contract followed verbatim. |
| decoder_layer | debug | n/a | — | 0 |  |
| decoder_layer | optimization | pending | — | 0 |  |
| decoder_layer | real_weights | pending | — | 0 |  |
| lm_head | reference | done | 1.000000 | 0 | untied Linear 1536->151936 no bias, real lm_head.weight, matches torch.nn.Linear |
| lm_head | ttnn | done | 1.000000 | 0 | Untied vocab projection Linear 1536->151936 no bias as a single column-parallel ttnn.linear: weight [vocab,hidden] transposed to [hidden,vocab] and vocab-sharded ShardTensorToMesh(dim=-1) on the 1x4 mesh per parallelism plan (placement=shard vocab dim; 151936/4=37984 logits per chip, tile-divisible, no padding); replicated activation in, per-chip logit slices out, recombined on host via ConcatMeshToTensor(dim=-1) per tp-guidance (all_gather left to the generation-phase consumer). reference_impl tt_transformers/tt/lm_head.py DRAM-sharded program configs / column splits deferred to optimization. HiFi4+fp32-acc, bf16; real lm_head.weight re-loaded from checkpoint in the test; PCC 1.000000 vs golden. Guard ok (lint 0, kernels ok, no new host ops). KB entries for kind linear were attention/tensor-manipulation records (nlp_concat_heads_decode, concatenate_heads, flash_mla_prefill, narrow/split/bcast/concat, create_qkv_heads) - none applicable, none used. Dispatched inline (no Agent tool in tick context); worker contract followed verbatim. |
| lm_head | debug | n/a | — | 0 |  |
| lm_head | optimization | pending | — | 0 |  |
| lm_head | real_weights | pending | — | 0 |  |

## Use cases

| Name | Input | Output | needs_ar | Generation | Perf |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ocr | image | text | yes | pending | pending |

## Recent Ticks

- tick 10 (2026-06-10T03:26:43Z): device[vision_block] — ok
- tick 11 (2026-06-10T03:32:56Z): device[patch_merger] — ok
- tick 12 (2026-06-10T03:50:19Z): device[vision_transformer] — ok
- tick 13 (2026-06-10T03:57:57Z): device[embedding] — ok
- tick 14 (2026-06-10T04:05:51Z): device[text_rmsnorm] — ok
- tick 15 (2026-06-10T04:25:05Z): device[text_attention] — ok
- tick 16 (2026-06-10T04:31:32Z): device[text_mlp] — ok
- tick 17 (2026-06-10T04:39:45Z): device[decoder_layer] — ok
- tick 18 (2026-06-10T04:46:04Z): device[lm_head] — ok
- tick 19 (2026-06-10T04:53:46Z): device[vision_patch_embed] — ok

## Host-Resident Exceptions

_None._

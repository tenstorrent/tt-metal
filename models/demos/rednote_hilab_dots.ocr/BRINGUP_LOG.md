# BRINGUP LOG: rednote-hilab/dots.ocr

**Model:** `rednote-hilab/dots.ocr`
**Slug:** `rednote_hilab_dots.ocr`
**Target Device:** p150 (blackhole)
**Started:** 2026-05-29T00:11:46Z
**Updated:** 2026-05-29T03:19:23Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| vision_patch_embed | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_patch_embed | ttnn | skipped | — | 0 |  |
| vision_patch_embed | debug | n/a | — | 0 |  |
| vision_patch_embed | optimization | pending | — | 0 |  |
| vision_patch_embed | real_weights | pending | — | 0 |  |
| vision_rmsnorm | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_rmsnorm | ttnn | done | 0.999995 | 0 | ttnn.rms_norm eps=1e-5 HiFi4+fp32_dest_acc bf16 DRAM TILE; PCC 0.99999 vs seed-0 golden on p150 |
| vision_rmsnorm | debug | n/a | — | 0 |  |
| vision_rmsnorm | optimization | done | 0.999995 | 0 | tracy attached (traced session, 1 op); rms_norm is the entire block (100% of 30.2us block kernel time), already multi-core on 32 cores (1 row-tile/core), HiFi4+fp32_dest_acc+packer_l1_acc, bf16 DRAM TILE. A/B vs width-sharded LayerNormShardedMultiCoreProgramConfig: K_t=48 does not divide 32/64 cores; compiling variants (6-8 cores, block_h=32) overflow static L1 CBs. No op-level optimization warranted - at ceiling. PCC 0.99999 held. |
| vision_rmsnorm | real_weights | pending | — | 0 |  |
| vision_attention | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_attention | ttnn | done | 0.999988 | 0 | manual SDPA chain (qkv linear -> nlp_create_qkv_heads -> 2D RoPE -> matmul+softmax+matmul w/ block-diagonal additive mask -> nlp_concat_heads -> proj). HiFi4+fp32_dest_acc, bf16. PCC=0.999988 vs golden. Guard ok. |
| vision_attention | debug | n/a | — | 0 |  |
| vision_attention | optimization | done | — | 1 | Tracy captured under --traced (metal trace replay session) at production shapes seq=256 cu_seqlens=[0,96,256]. Top hotspot ReshapeViewDeviceOperation 33.1%. Fix: L1-pinned the head-split (qkv->[seq,3,nh,hd]) and head-merge (->[seq,nh*hd]) reshape outputs via memory_config=L1_MEMORY_CONFIG. Block kernel time 970.84->739.34us (-23.8%). The reshape op itself unchanged (~253us) but pinning its output to L1 cut the downstream slice/RoPE/transpose chain reading from DRAM (Slice 90.7->24.0us, Binary 180->130.7us, Transpose 80.6->52.7us). PCC 0.99998858 held. |
| vision_attention | real_weights | pending | — | 0 |  |
| vision_mlp | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_mlp | ttnn | done | 0.999986 | 0 | fused gate(fc1)/up(fc3) ttnn.linear -> ttnn.silu(gate)*up -> down(fc2) ttnn.linear. No bias. HiFi4+fp32_dest_acc bf16 DRAM TILE. PCC=0.9999855 vs golden. Guard ok. |
| vision_mlp | debug | n/a | — | 0 |  |
| vision_mlp | optimization | done | 0.999986 | 0 | Tracy captured under --traced (metal trace replay session) at production shapes seq=256 dim=1536 intermediate=4224. Top hotspot MatmulDeviceOperation 74.5% (two linears, matmul-bound, at ceiling). Optimization budget was the gate/up split + SwiGLU elementwise chain (Slice+silu+mul=25.5%) landing DRAM-interleaved. Fix: replaced python-getitem split with ttnn.slice+L1_MEMORY_CONFIG and pinned silu/mul outputs to L1. Block kernel time 293.81->253.26us (-13.8%). Slice 30.82->18.32, silu 18.68->10.03, mul 25.52->8.04us; activation chain 75.0->36.4us (-51%), now INTERLEAVED/L1. Matmuls unchanged ~217us (real floor). PCC 0.9999855 held. |
| vision_mlp | real_weights | pending | — | 0 |  |
| vision_block | reference | done | 1.000000 | 0 | DotsVisionBlock: pre-norm RMSNorm -> fused-QKV 2D-RoPE bidirectional attn -> residual -> RMSNorm -> SwiGLU -> residual. PCC=1.0 vs HF eager. |
| vision_block | ttnn | done | 0.999996 | 0 | First composite: pre-norm residual h=h+attn(norm1(h)); h=h+mlp(norm2(h)). Reuses TtVisionRMSNorm/Attention/MLP leaves by file-path import. HiFi4+fp32_dest_acc, bf16. Guard ok. |
| vision_block | debug | n/a | — | 0 |  |
| vision_block | optimization | done | 0.999996 | 0 | Profiled TtVisionBlock under metal trace (tracy --traced). Composite total device kernel 1055.5->1050.2us. Top hotspot MatmulDeviceOperation 35.8% + ReshapeViewDeviceOperation 30.6% both inside already-optimized leaves (vision_attention -23.8%, vision_mlp -13.8%) -- inherited wins. Applied one composite-boundary lever: pinned both residual ttnn.add ops to L1 (only block-internal ops landing DRAM-interleaved). Residual adds 12.63->8.76us (-30.6%) DRAM->L1. PCC 0.9999956 unchanged. |
| vision_block | real_weights | pending | — | 0 |  |
| vision_patch_merger | reference | done | 1.000000 | 0 | PatchMerger: LayerNorm(eps=1e-6,bias) -> view(merge**2) -> Linear -> GELU -> Linear (all biased). PCC=1.0 vs HF. |
| vision_patch_merger | ttnn | done | 0.999990 | 0 | PatchMerger: ttnn.layer_norm (weight+bias, eps=1e-6) -> ttnn.reshape group 4 patches (1536->6144) -> ttnn.linear+bias -> ttnn.gelu -> ttnn.linear+bias. LN gamma/beta laid out [1,1,dim//32,32] row-major. HiFi4+fp32_dest_acc bf16 DRAM TILE. PCC=0.99999 vs golden. Guard ok. |
| vision_patch_merger | debug | n/a | — | 0 |  |
| vision_patch_merger | optimization | done | 0.999990 | 0 | traced tracy at production shape (SEQ=256->64 merged): matmuls 81.1% (303us/96 cores, K=6144, fused bias), LayerNorm 11.6%/8c, reshape 4.9%, GELU 2.3%; all DRAM-interleaved. Tried L1-pinning the LN/reshape/fc1/GELU/fc2 chain: net-neutral (368 vs 372us) since matmuls are compute-bound and the explicit-L1 output split the matmul fused-bias into a separate +6us BinaryNg. Reverted to fused-bias DRAM path. At-ceiling for per-block tuning; matmuls already 96 cores. PCC 0.99999 unchanged. |
| vision_patch_merger | real_weights | pending | — | 0 |  |
| vision_tower | reference | done | 1.000000 | 0 | Full DotsVisionTransformer tested at REDUCED 2 layers (full=42, grid 1x4x4=16 patches, bf16=False fp32 path). PCC=1.0 vs HF. Full-depth check deferred to real_weights. |
| vision_tower | ttnn | done | 0.999982 | 0 | Full DotsVisionTransformer assembly at REDUCED 2 layers (full=42), grid 1x4x4=16 patches. Composes verified TtVisionBlock x2 (pre-norm residual, 2D vision RoPE theta 1e4, block-diagonal cu_seqlens attn) -> post_trunk TtVisionRMSNorm (eps 1e-5) -> TtVisionPatchMerger (LayerNorm eps 1e-6 + GELU MLP, merge 2x2) by file-path import. patch_embed Conv2d+RMSNorm run on host (documented host-resident boundary); 2D RoPE + cu_seqlens precomputed on host, threaded through blocks. HiFi4+fp32_dest_acc bf16 DRAM TILE. PCC=0.99998 vs golden on p150. Guard ok. |
| vision_tower | debug | n/a | — | 0 |  |
| vision_tower | optimization | done | 0.999982 | 0 | Traced tracy on the composite assembly (golden grid 1x4x4=16 patches, reduced 2-layer trunk standing in per-layer for full 42). Total device kernel 1084.8us. Top hotspot MatmulDeviceOperation 64.2% (696.5us/14 ops: QKV+proj+MLP fc1/fc2/fc3 + merger fc1/fc2) -- all inside already-optimized leaves (vision_attention -23.8%, vision_mlp -13.8%, merger matmuls 96c at-ceiling). LayerNorm 15.4% all single-core: post_trunk_norm is a TtVisionRMSNorm instance and inherits that leaf's at-ceiling verdict (K_t=48 doesn't divide 32/64 cores; width-sharded variants overflow static L1 CBs). Reshape 6.4% + Binary 5.4% + Transpose 2.1% are leaf-internal (RoPE/head ops + vision_block residual adds already L1-pinned). The vision_tower.py assembly layer itself adds NO inter-block reshard/transpose and NO DRAM hotspot: block->block boundary lands L1 (residual adds), and post_trunk_norm->merger handoff is already L1. The 72.2% DRAM share is entirely leaf-owned matmul output (intentional: merger fused-bias DRAM path proven net-best in the patch_merger tick; L1-pinning leaf matmuls was net-neutral). No composite-boundary lever available -- at-ceiling, inherits leaf wins. PCC 0.9999818 unchanged. |
| vision_tower | real_weights | pending | — | 0 |  |
| embedding | reference | done | 1.000000 | 0 | Qwen2 token embedding (vocab 151936, hidden 1536) lookup. PCC=1.0 vs nn.Embedding. |
| embedding | ttnn | done | 0.999999 | 0 | ttnn.embedding gather; weight [151936,1536] bf16 ROW_MAJOR in DRAM; uint32 row-major ids -> TILE output. HiFi4+fp32_dest_acc preset (no matmul; exact gather). PCC=0.9999986 vs golden on p150. Guard ok. |
| embedding | debug | n/a | — | 0 |  |
| embedding | optimization | done | 0.999999 | 0 | tracy attached (traced replay session); top op EmbeddingsDeviceOperation = 100% of the 21.6us block kernel time. Single ttnn.embedding gather: no matmul to shard, no reshape chain to L1-pin, no activation to fuse. At-ceiling — the gather is the optimal kernel for a 151936x1536 DRAM table lookup. No targeted optimization warranted. PCC=0.9999986 unchanged. |
| embedding | real_weights | pending | — | 0 |  |
| rmsnorm | reference | done | 1.000000 | 0 | Qwen2RMSNorm (eps 1e-6): fp32 normalize -> cast -> weight*x. PCC=1.0 vs HF Qwen2RMSNorm. |
| rmsnorm | ttnn | done | 0.999995 | 0 | Qwen2 LM RMSNorm eps=1e-6 (vs vision 1e-5). ttnn.rms_norm HiFi4+fp32_dest_acc bf16 DRAM TILE; weight reshaped [1,1,dim//32,32] row-major. Mirror of TtVisionRMSNorm. PCC=0.99999 vs golden on p150. Guard ok. |
| rmsnorm | debug | n/a | — | 0 |  |
| rmsnorm | optimization | done | 0.999995 | 0 | tracy attached (traced session, 1 op); rms_norm (LayerNormDeviceOperation) is the entire block (100% of 30.5us block kernel time), already multi-core on 32 cores, HiFi4+fp32_dest_acc+packer_l1_acc, bf16 DRAM TILE. K_t=48 divides only 6/8/16/24/48 cores (not the 32/64 grid); width-sharded variants reduce parallelism vs the 32-core height path and overflow static L1 CBs at compile (same as vision_rmsnorm tick 22). No op-level optimization warranted - at ceiling. PCC 0.99999 held. |
| rmsnorm | real_weights | pending | — | 0 |  |
| rope | reference | done | 1.000000 | 0 | Qwen2RotaryEmbedding (theta 1e6, head_dim 128, default rope): position_ids -> (cos,sin). PCC cos=1.0 sin=1.0 vs HF. |
| rope | ttnn | done | 1.000000 | 0 | Qwen2 LM RoPE cos/sin tables (theta 1e6, head_dim 128). On-device outer(pos,inv_freq)->cat->cos/sin, fp32. PCC cos=1.0 sin=1.0 vs golden. |
| rope | debug | n/a | — | 0 |  |
| rope | optimization | done | 1.000000 | 0 | Tracy traced: 4 ops (multiply 64->70%, cos/sin Unary 25->22%, concat 10->7%). All output was 100% DRAM-interleaved; pinned the whole chain to L1 (memory_config=L1 on multiply/concat/cos/sin). Layout flipped 100% DRAM->100% L1; total kernel 27686->25366 ns (-8.4%); concat -36%, cos/sin -18%. PCC cos=1.0 sin=1.0 (min 0.9999999999999976) unchanged. |
| rope | real_weights | pending | — | 0 |  |
| attention | reference | done | 1.000000 | 0 | Qwen2Attention (GQA 12/2, head_dim 128, QKV bias, o_proj no bias, 1D RoPE, causal eager). PCC=1.0 vs HF eager. |
| attention | ttnn | done | 0.999989 | 0 | GQA 12/2 head_dim 128, QKV bias, o_proj no bias, 1D RoPE theta 1e6, causal additive mask. Manual head split (fused cat[Wq,Wk,Wv]), repeat_kv n_rep=6, HiFi4+fp32_dest_acc. |
| attention | debug | n/a | — | 0 |  |
| attention | optimization | done | 0.999989 | 0 | Tracy captured under --traced (metal trace replay session) at production shapes seq=128 GQA 12/2 head_dim=128 causal. Top hotspot ReshapeViewDeviceOperation 45.5%. Fix: L1-pinned the head-split (qkv slices -> reshape [1,seq,nh/nkv,hd]) and head-merge (-> [seq,nh*hd]) reshape outputs via memory_config=L1_MEMORY_CONFIG. Block kernel time 445.68->348.43us (-21.8%); 60.2% of output now L1 vs 100% DRAM before. Downstream chain shrank: Matmul 148.08->92.90us, Transpose 37.58->22.64us, Binary 29.00->25.33us. PCC 0.999989322840819 held. |
| attention | real_weights | pending | — | 0 |  |
| mlp | reference | done | 1.000000 | 0 | Qwen2MLP SwiGLU SiLU (gate/up/down 1536<->8960, no bias). PCC=1.0 vs HF. |
| mlp | ttnn | done | 0.999985 | 0 | Qwen2 SwiGLU MLP 1536<->8960 no bias; fused gate/up linear, on-device silu+mul; HiFi4+fp32_dest_acc. Guard ok. |
| mlp | debug | n/a | — | 0 |  |
| mlp | optimization | done | 0.999985 | 0 | Tracy captured under --traced (metal trace replay session) at production shapes seq=128 dim=1536 intermediate=8960. Top hotspot MatmulDeviceOperation 90.4% (two linears, matmul-bound, real floor). Optimization budget was the gate/up split + SwiGLU elementwise chain (Slice+silu+mul) landing DRAM-interleaved. Fix: replaced python-getitem split with ttnn.slice+L1_MEMORY_CONFIG and pinned silu/mul outputs to L1. Block kernel time 424.71->392.92us (-7.5%). Slice 31.49->19.18, silu 15.48->10.81, mul 19.84->7.68us; activation chain 66.8->37.7us (-44%), now INTERLEAVED/L1. Matmuls unchanged ~355us. PCC 0.9999850 held. |
| mlp | real_weights | pending | — | 0 |  |
| decoder_layer | reference | done | 1.000000 | 0 | Qwen2DecoderLayer: input RMSNorm -> GQA self-attn (12q/2kv, head_dim128, RoPE theta 1e6, QKV bias) -> residual -> post-attn RMSNorm -> SwiGLU MLP -> residual. PCC=1.0 vs HF Qwen2DecoderLayer. |
| decoder_layer | ttnn | done | 0.999996 | 0 | Qwen2 LM decoder layer composite (pre-norm residual). Composes verified leaves TtRMSNorm x2 (eps 1e-6) + TtAttention (GQA 12/2, QKV bias, 1D RoPE theta 1e6, causal) + TtMLP (SwiGLU) by file-path import. HiFi4+fp32_dest_acc, bf16 DRAM TILE. Guard ok. |
| decoder_layer | debug | n/a | — | 0 |  |
| decoder_layer | optimization | done | 0.999996 | 0 | Composite LM decoder layer. Tracy traced (decoder_layer.traced_profile.csv). Top hotspot MatmulDeviceOperation 55.9% + ReshapeView 19.9% = 75.8% device kernel time, all INHERITED from optimized leaves (attention -21.8%, mlp -7.5%). Composite-boundary lever = two residual ttnn.add ops, which landed DRAM-interleaved; pinned both to L1_MEMORY_CONFIG (vision_block pattern). Residual adds 4.33/4.55/4.41us DRAM -> 4.08/2.79us L1. Total 803.4 -> 799.5us (-0.5%); small because composite is dominated by inherited matmul/reshape. PCC 0.99999559 unchanged. At-ceiling on inheritance + small residual L1 win. |
| decoder_layer | real_weights | pending | — | 0 |  |
| lm_head | reference | done | 1.000000 | 0 | Untied Linear hidden 1536 -> vocab 151936, no bias. PCC=1.0 vs nn.Linear. |
| lm_head | ttnn | done | 0.999996 | 0 | untied Linear hidden 1536 -> vocab 151936, no bias; HF weight [vocab,hidden] transposed on host. ttnn.linear HiFi4+fp32_dest_acc bf16 DRAM weight. Matches lm_head_forward (bare linear, no final norm). PCC=0.9999958 vs golden. Guard ok. |
| lm_head | debug | n/a | — | 0 |  |
| lm_head | optimization | pending | — | 0 |  |
| lm_head | real_weights | pending | — | 0 |  |
| language_model | reference | done | 1.000000 | 0 | Full Qwen2ForCausalLM tested at REDUCED 2 layers (full=28), seq 64: embed -> 2x decoder_layer -> final RMSNorm -> lm_head. PCC=1.0 vs HF. Full 28-layer check deferred to real_weights. |
| language_model | ttnn | done | 0.999911 | 0 | Full Qwen2ForCausalLM assembly at REDUCED 2 layers (full=28), seq 64. Composes verified TtEmbedding -> 2x TtDecoderLayer (GQA 12/2, QKV bias, 1D RoPE theta 1e6, causal) -> final TtRMSNorm (eps 1e-6, applied here before lm_head) -> TtLMHead (untied hidden->vocab no bias) by file-path import. Shared cos/sin + causal mask precomputed on host, threaded through layers. HiFi4+fp32_dest_acc bf16 DRAM TILE. PCC=0.99991 vs golden on p150. Guard ok. |
| language_model | debug | n/a | — | 0 |  |
| language_model | optimization | pending | — | 0 |  |
| language_model | real_weights | pending | — | 0 |  |

## Use cases

| Name | Input | Output | needs_ar | Generation | Perf |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ocr | image | text | yes | pending | pending |

## Recent Ticks

- tick 24 (2026-05-29T02:14:00Z): device[vision_mlp] — ok
- tick 25 (2026-05-29T02:20:39Z): device[vision_block] — ok
- tick 26 (2026-05-29T02:32:56Z): device[vision_patch_merger] — ok
- tick 27 (2026-05-29T02:40:41Z): device[vision_tower] — ok
- tick 28 (2026-05-29T02:46:22Z): device[embedding] — ok
- tick 29 (2026-05-29T02:52:35Z): device[rmsnorm] — ok
- tick 30 (2026-05-29T02:58:59Z): device[rope] — ok
- tick 31 (2026-05-29T03:05:15Z): device[attention] — ok
- tick 32 (2026-05-29T03:11:57Z): device[mlp] — ok
- tick 33 (2026-05-29T03:19:23Z): device[decoder_layer] — ok

## Host-Resident Exceptions

- **vision_patch_embed**: DotsPatchEmbed is a single Conv2d(3,1536,k=14,s=14) over patchified pixels followed by RMSNorm; Conv2d patchify is a one-shot host-side im2col+matmul in the Qwen-VL TTNN demos (qwen25_vl runs patch embed on host then moves tokens to device). Cheap relative to the 42-layer trunk and runs once per image. (ref: models/demos/qwen25_vl/tt/model.py)

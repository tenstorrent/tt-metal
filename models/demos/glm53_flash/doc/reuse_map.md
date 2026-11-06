# GLM-5.3-Flash reuse map (tt-metal main 8cfcaf754d3 + arg/glm53_mesh 253f593bf2f, 2026-10-21)

Directive: the first implementation uses the reference layer implementations other models already carry
("best layer implementation per shape") instead of inventing nodes. Filter (coordinator, 2026-10-21): ORCHESTRATION
(mesh axes, TP/DP/EP/SP placement, CCL patterns, cache partitioning, trace/dispatch structure) is taken only from
models built for the 8x4 Blackhole Galaxy (`models/demos/deepseek_v3`, `models/demos/llama3_70b_galaxy`; the
single-Galaxy prefill engine `models/demos/deepseek_v3_d_p`); PER-LAYER KERNELS and program configs may come from any
Blackhole model whose per-chip shape matches ours. QB2 (8-chip) and Wormhole mesh layouts are not imported. Every row
names the device/mesh the reference was tuned for and whether the reuse is `orchestration` or `kernel`.

Our per-chip shapes (8x4, TP4 over the columns, DP8 rows; one user group per row; sequence-parallel prefill over
the rows): KDA 16 heads x K = V = 128 per chip, DSA 16 MLA heads (q 256, latent 512, no rope) + 32 index heads x 128,
MoE 288 experts top-8 (I = 2048, hidden 4096) + 1 shared, mHC 4 streams x 4096 fp32, vocab 154 880.
Measured today (`doc/mesh/README.md`, `doc/mesh/tracy/README.md`, `doc/functional_decoder/README.md`): 8x4 decode
76.45 ms/step at 32 users (13.1 tok/s/user), 121.56 ms at 256 users; TP4 prefill 16.9 ms (KDA+dense) / 168-169 ms
(MoE layers) per layer at S = 1024 on one row; 128-token TTFT 919 ms eager.

## 1. The KDA question (deliverable 2): device check on chip 3

Test: `tests/test_kda_vs_gdn_ops.py` (random inputs, HF `transformers.models.glm5_next.modeling_glm5_next`
`recurrent_kimi_delta_attention` as the exact fp32 reference; B = 1, H = 16, K = V = 128, gate
`g = -5 sigmoid(.)` per key channel, two gate regimes: mild = mean -0.3 / token, strong = mean -2.5 / token).
Logs: `/home/ttuser/ashai/glm53/_logs/agents/reuse/kda_vs_gdn_run2.log` (prefill + gauge), `..._run4.log` (decode).
Chip 3 grid 12x10 = 120 compute cores. The compiled binary (`build/lib/_ttnncpp.so` 2026-10-19 19:25) is the
upstream op: `g [B, T, HV]`, one log decay per head and token (`ttnn.transformer.fused_recurrent_gated_delta_rule.__doc__`
checked at runtime; the per-key variant of 1b3697eae4c was dropped in ac9e77011db and never built here).

### 1.1 Prefill: `chunk_gated_delta_rule` vs the d_p ttKDA recurrence

| T | gate regime | `chunk_gated_delta_rule` (flat token-major path, chunk 32, cost-model path) | PCC vs per-channel KDA ref (out / state) | PCC vs uniform-gate ref | `KDARecurrence` (d_p, `ttnn.experimental.kda.*`) | PCC vs per-channel ref (out / state) |
|---|---|---|---|---|---|---|
| 2048 | mild | 0.52 ms | 0.9599 / 0.9597 | 1.0000 | 0.72 ms | 0.9999 / 0.99999 |
| 2048 | strong | 0.50 ms | 0.9887 / 0.9905 | 1.0000 | 0.70 ms | 1.0000 / 0.9999 |
| 8192 | mild | 1.61 ms | 0.9606 / 0.9583 | 1.0000 | 2.82 ms | 0.9999 / 0.99999 |
| 8192 | strong | 1.61 ms | 0.9891 / 0.9893 | 1.0000 | 2.80 ms | 0.9999 / 0.9999 |

* The GDN op computes exactly what it documents (PCC 1.0000 against the reference run with the same per-head gate);
  its mismatch to KDA (0.96 / 0.99) is identical to the reference-vs-reference gap of replacing the per-channel gate by
  its per-head mean (0.9598 / 0.9887): the op cannot express Kimi's rule, there is no scalar gate that does.
* Adapter analysis. KDA's decay is a row scale of the state, `S <- diag(exp(g_t)) S`. In the basis
  `S' = exp(-G) S` (G = cumulative log decay) the rule is an ungated delta rule with READ key `k exp(+G)` and WRITE key
  `k exp(-G)`. `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` use one k for both (no adapter);
  `gated_delta_attn_seq` takes separate `q_decay / k_bd_sc / k_decay_t` terms (the asymmetry IS expressible) but its
  state decay `dl_exp` is `[BH, NC, 1, 1]` (scalar per chunk), so the gauge has to span the whole sequence and
  `exp(+-G)` leaves fp32 (`test_gauge_fold_is_numerically_dead`: exact at T = 32 mild (min G -20), NaN at T = 32 strong
  (min G -104) and at every T >= 256). Verdict: no adapter; the per-channel kernels are the `ttnn.experimental.kda.*`
  family the d_p ttKDA already uses (prepare_chunk_recurrence takes `g [1, T, H*K]`, `final_decay [H, N, K, 1]` is a
  per-key column).
* Speed: even if it were exact, the GDN op is only 1.4-1.75x faster than the ttKDA recurrence, and the recurrence is a
  small share of the KDA layer prefill (the TP4 KDA+dense layer costs 16.9 ms at S = 1024; the recurrence at T = 1024
  x 16 heads is ~0.4 ms by the table above): no prefill gain is available from the GDN op.
* Batch / head counts the ops accept here: the phased scan needs one core per (user, head) and the fused path NV + NP
  cores per head, so `B * HV <= 120` per call on this grid (B = 1 x 64 heads fits; multi-user prefill at 16 heads
  = 7 users per call); K, V multiples of 32; the flat path needs K == V, chunk 32, T % 32 == 0. PR #60316's head-wave
  dispatch is not in this tree: a larger BH is refused, not waved (`qwen38_27b_qb2/tt/decoder.py::_delta` loops
  `scan_batch = cores // hv` slices for the same reason).

### 1.2 Decode (T = 1): `fused_recurrent_gated_delta_rule` adapter vs our `kda_decode`

Adapter = ONE `ttnn.multiply(S, exp(g)[.., :, None])` (per-channel decay as a row scale, fp32 `[B, H, K, V]`) then the
fused op with `g = 0` on the scaled state. Exact: out PCC 0.999999 / 1.000000, state 1.000000 at B = 8 / 32.
Cost: `120 // 16 = 7` users per fused call (one core per (user, head)), so 2 calls at B = 8 and 5 at B = 32:

| B (users) x 16 heads | adapter, eager ms/step (fused calls) | our `custom_ops/kda_decode` block, traced (`doc/functional_decoder/README.md`) |
|---|---|---|
| 8 | 2.563 (2) | 0.248 ms at B = 4 (program 70.7 us) |
| 32 | 6.241 (5) | 0.438 ms (program 257 us, state moved twice = 250 GB/s) |

Verdict: no sibling for KDA decode; keep `custom_ops/kda_decode` (it also folds the conv, L2 norms, gates and the
gated RMSNorm; the fused op would still leave ~60 composite ops around it). A T = K+1 speculative-verify window has
no exact path either (same gate form); the per-key kernel of 1b3697eae4c is the starting point if MTP is wanted.

## 2. Reuse map by layer type and mode

Columns: existing implementation (file:function, ops, configs / precision), tuned for (device, mesh, per-chip shape,
recorded perf), reuse type, what we use today, gap, verdict (reuse as-is / reuse with adapter / no sibling: keep ours).

| # | GLM-5.3 layer / mode | existing implementation | tuned for | type | ours today | gap | verdict |
|---|---|---|---|---|---|---|---|
| 1 | KDA prefill (recurrence) | `models/demos/deepseek_v3_d_p/tt/kda/kda.py::ttKDA.forward` + `recurrence.py::KDARecurrence`: `ttnn.experimental.kda.{prepare_chunk_recurrence, recurrent_chunk_scan, summarize_chunk_recurrence, affine_exclusive_scan, reduce_affine_transforms, chain_affine_transforms, select_final_carry}`; prep HiFi4 fp32-acc, scan / prefix HiFi2 fp32-acc, state fp32, chunk terms bf16/fp32 in L1 when they fit (`config.py`) | Kimi-K3 / DSv4 on the 8x4 BH Galaxy, SP8 x TP4, 128-dim heads, conv 4, bounded gate -5 (1.7 ms / layer at T = 5120 SP8xTP4 per the design doc); here 0.72 ms at T = 2048 / 2.82 ms at 8192 for 16 heads | kernel + orchestration (SP over rows, TP over columns: the same axes as ours) | `tt/kda.py::TtKDA.prefill_forward` (ttKDA with `KDAProgramConfig(HiFi2 projections, bf16 gated-rms out)`, grouped scan sized to the local chunk count on multi-row meshes) | none (section 1.1) | **reuse as-is (done)** |
| 2 | KDA prefill alternative | `ttnn.transformer.chunk_gated_delta_rule` as called by `models/demos/qwen38_27b_qb2/tt/decoder.py::_delta` (flat `[b, t, h*d]` q/k/v, chunk 32, `delta_constants` eye/tril/ones/masks, `scan_batch = cores // hv` slices, `output_head_major`) and `models/demos/blackhole/qwen36/tt/gdn/tp.py` | Qwen3.8-27B on QB2 (P300x2 / P150x8, 1D TP) and Qwen3.6-27B single BH chip; GDN heads Hk 16 / HV 16..128 x 128, scalar gate per head | kernel | not used | scalar gate (section 1.1: PCC 0.96-0.99 vs KDA); 1.4-1.75x faster than row 1 on a < 5 % share | **reject** |
| 3 | KDA conv + gated norm | `ttnn.experimental.kda.qkv_causal_conv1d_silu(_with_carry)` (`QkvCausalConv1dSiluProgramConfig(channel_chunk_size)`), `sigmoid_gated_rms_norm`; the same two ops are what qwen38 prefill calls | Kimi-K3 Galaxy (ttKDA) and Qwen3.8 QB2 | kernel | inside ttKDA for prefill; decode conv / norm fused in `custom_ops/kda_decode` | none | **reuse as-is (prefill, done)**; decode stays ours |
| 4 | KDA decode | `ttnn.transformer.fused_recurrent_gated_delta_rule` (`qwen36/tt/gdn/decode.py`); qwen38 runs the chunk op at T = 1 padded to 32 rows | Qwen3.6 single BH chip; one core per (user, head) | kernel | `custom_ops/kda_decode` (5 ops per layer, 0.25 / 0.44 ms at 16 heads B = 4 / 32) | exact only through the 2-op adapter, 2.6 / 6.2 ms (section 1.2) | **no sibling: keep ours** |
| 5 | DSA prefill, dense rows (< 2048) | `models/demos/deepseek_v3/tt/mla/mla1d.py` prefill: `ttnn.transformer.flash_mla_prefill` / `chunked_flash_mla_prefill` on the absorbed q over the latent (`SDPAProgramConfig(q_chunk Q_CHUNK_SIZE, k_chunk K_CHUNK_SIZE = 64 via DEEPSEEK_K_CHUNK_SIZE)`, HiFi4, compute config written for Wormhole, heads TP over columns, DP rows) | DeepSeek-V3 on the 8x4 Galaxy (latent 512 + rope 64) | kernel (Galaxy-tuned) | `tt/dsa.py`: `ttnn.transformer.scaled_dot_product_attention` on K/V materialized per head from the latent (q/k chunk 128) | ours has no rope part (576 vs 512 wide latent) and only dense rows below position 2048; absorbed-form flash MLA skips the per-head K/V materialization | **reuse with adapter** (small gain: dense rows are <= 2048 per prompt; measure before swapping) |
| 6 | DSA prefill, sparse rows | `ttnn.transformer.sparse_sdpa` + `models/demos/deepseek_v3_d_p/tt/mla/mla.py::_sparse_mla` (heads -> sequence `all_to_all_async_generic` so each chip runs all 64 heads on a sequence quarter when the TP shard is < 32 heads, inverted after) | GLM-5.3 (big) / DSv3.2 on the Galaxy prefill engine, SP x TP | kernel (op, done) + orchestration (the a2a trick) | `tt/dsa.py` sparse path: `sparse_sdpa(k_chunk 128)` on heads PADDED 16 -> 32 (`_pad_heads_rm`): 2x the attention work at TP4 | the pad | **reuse with adapter: the head -> sequence a2a of `_sparse_mla` instead of padding** (slabattn agent's slab-local SP design is the place) |
| 7 | DSA decode (dense <= 2050, sparse beyond) | `ttnn.transformer.paged_flash_multi_latent_attention_decode` as in `mla1d.py::_fwd_decode_flash_mla` (k_chunk 64, HiFi4, no fp32 acc, `cur_pos_tensor` per user) preceded by `_fwd_decode_all_to_all_pre_flash_mla` (`all_to_all_async_generic(cluster_axis=1)`: heads -> users, `[1, 32, 16, 576] -> [1, 4, 128, 576]`, each column chip attends ITS users with all heads on its own cache) | DeepSeek-V3 decode on the 8x4 Galaxy (user-partitioned latent cache over the TP columns) | orchestration (Galaxy) + kernel | same op (k_chunk 32) in `custom_ops/dsa_decode_state` block (0.86 / 0.97 ms at B = 4 / 32, 64 heads); user-partitioned cache + `GLMMesh.all_to_all_cols` already in `tests/test_mesh_sparse_partitioned.py` (OPT_SPARSE_1M) | k_chunk 32 vs 64 (sweep); the sparse decode gather (`custom_ops/paged_pool_gather`, `index_score_decode`, `topk_rows`) has no sibling (DSv3 is dense) | **reuse as-is (orchestration adopted; kernel done)** |
| 8 | Indexer prefill / decode | `ttnn.experimental.indexer_score_dsa(key_compression_ratio=4)`, `topk_large_indices`, `ring_indexer_score_dsa`; `deepseek_v3_d_p/tt/mla/indexer.py::TtIndexer` (own TP/SP collectives, device index-key cache) | GLM-5.3 big / DSv3.2 on the Galaxy prefill engine | kernel (ops) + orchestration (SP x TP indexer collectives) | ops used directly in `tt/indexer.py`; decode batched by `custom_ops/index_score_decode` + `topk_rows`; the k-pool gate composite is ours (Flash-only) | none for the ops | **reuse as-is (done)**; TtIndexer's SP collectives = reference for the slab-local prefill |
| 9 | MoE router | `models/demos/deepseek_v3/tt/deepseek_moe_gate/op.py` (unified kernel: 8 groups x 32 experts, top-2-sum group select, top-8), `ttnn.experimental.deepseek_prefill.moe_grouped_topk` (single-group path ranks at bf16 resolution), `models/common/modules/moe/tt_moe_gate.py` | DSv3 Galaxy (256 experts, n_group 8) | kernel | exact fp32 composite `tt/moe.py::TtRouter` (sigmoid + bias top-8, renorm x 2.5; sets 32/32, 255/256 vs HF) | n_group 1 / 288 experts / exactness (bf16 ranking swaps near ties) | **no sibling: keep ours** |
| 10 | Routed experts, decode | `models/demos/deepseek_v3/tt/moe.py::_fwd_moe` + `experts.py`: all-gather over the TP columns -> gate -> row-major tokens -> `ttnn.all_to_all_dispatch(cluster_axis=0)` with the `[E, D]` expert map -> each device runs its `E / 32` FULL experts as batched `ttnn.linear` (w1/w3 bfp4 LoFi, w2 bfp8 HiFi2, SiLU fused) on the dispatched rows -> `ttnn.all_to_all_combine` -> `ttnn.sum(dim 0)` or `deepseek_moe_fast_reduce_nc` (1D ring) -> `reduce_scatter_minimal_async(cluster_axis=1)`; `tt/ccl.py` persistent semaphores, `num_links = ccl.get_max_links(axis)` | DeepSeek-V3 decode on the 8x4 Galaxy (EP32, 8 experts per chip) | orchestration (Galaxy) | hybrid EP4 x expert-TP8 (`tt/moe.py::TtRoutedExperts`, `custom_ops/expert_stream` at 60-74 % of DRAM peak on full-width slices) and the EP path `TtEPExperts` (`tests/test_moe_ep.py`: 9 full experts 297 us at 32 tokens / 590 us at 256, exact) | dtype policy (bfp4 gate/up only under the eval gate), 288 / 32 = 9 experts per chip, our streaming expert op instead of batched linear | **reuse orchestration (in progress: `doc/mesh/orchestration_v2.md`, OPT_EXPERTS)** |
| 11 | Routed experts, prefill | `models/demos/deepseek_v3_d_p/tt/moe/` (`tt_moe.py`, `tt_dispatch.py`, `tt_combine.py`, `tt_routed_expert.py`): `deepseek_prefill.dispatch_fabric2d` -> `unified_routed_expert_moe` (has `ClampedSiluGlu` = our swiglu_limit) -> `combine_fabric2d` -> `post_combine_reduce`; capacity-factor buffers `[chips, experts_per_chip, tokens, emb]`, dense per-expert matmuls, bfp4 default / bfp8 option | DeepSeek-V3/V4 and GLM-5.3 big prefill on the 8x4 BH torus (EP32, SP x TP) | orchestration + kernel (Galaxy) | 32-token blocks through `sparse_matmul` (`TtRoutedExperts`) or the stream dispatch; 168-169 ms per MoE layer at S = 1024 on one TP4 row (prefill at ~0.3 % of the compute roofline) | adapter: 288 experts / top-8 / scale 2.5 / bias router / shared expert / our weight cache; the gate stays ours | **reuse with adapter: the single largest prefill item** |
| 12 | Shared expert / dense MLP | `deepseek_v3/tt/mlp/shared_expert.py`, `deepseek_v3_d_p/tt/moe/tt_shared_expert.py`, `models/common/modules/mlp` | DSv3 Galaxy TP | kernel | `tt/mlp.py::TtClampedSwiGLU` (`ttnn.clamped_silu_glu`, `DecodeLinear` 1D mcast configs: `tt/matmul_configs.py`), 89 us / step | the clamp (swiglu_limit) is ours; already at 1 % of the step | **keep ours** |
| 13 | mHC | `models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py::TtMHCWrap` (fused `deepseek_prefill.mhc_split_sinkhorn` parametrization in fp32, composite apply) | DSv4 prefill on the Galaxy (rows = tokens) | kernel (Sinkhorn) | Sinkhorn kernel reused inside `tt/mhc.py::TtMHCFast` + `custom_ops/mhc_pre_block` / `mhc_post_mix` (44 ops, 0.46 ms vs the wrap's 222 ops, 1.96 ms at B = 4) | the wrap's column slices cost 1.9 ms in decode; for prefill rows (T >= 1024) the wrap is the reference to measure (prefillprof) | **reuse the kernel (done); keep our fast path** |
| 14 | Embedding | `deepseek_v3/tt/embedding/embedding1d.py` (table DRAM-sharded 1D over all devices, `ttnn.embedding` + `all_gather_async`) | DSv3 Galaxy | orchestration | `tt/embedding.py`: table replicated on every chip (1.27 GB bf16 per chip), `ttnn.embedding` -> fp32 x 4 streams | memory only (1.2 GB per chip) | **reuse with adapter, low priority** |
| 15 | LM head + greedy | `deepseek_v3/tt/lm_head1d.py` (vocab split 1D, `ttnn.linear`, slices), `llama3_70b_galaxy/tt/lm_head.py`, `models/common/modules/sampling` (device sampling under trace) | Galaxy | orchestration | `tt/model_head.py`: vocab split over the 4 columns (5 x 30976 chunks), all-gathered logits, `ttnn.argmax`; `test_embedding_head.py` 2.36 ms per B = 8 step | split over 32 chips saves 0.6 GB per chip at the price of a wider gather; sampling beyond greedy is missing | **keep ours now; sampling module when serving starts** |
| 16 | Norms / residual | `deepseek_v3/tt/rms_norm/distributed_rms_norm.py` (`rms_norm_pre/post_all_gather`, TP-sharded residual) | DSv3 Galaxy | orchestration | replicated 4-stream fp32 residual, `ttnn.rms_norm` (single-core input norms folded into `mhc_pre_block`) | needs the sharded residual (phase 2c of orchestration_v2) | **later** |
| 17 | CCL plumbing | `deepseek_v3/tt/ccl.py` (`all_gather_async`, `reduce_scatter_minimal_async`, `all_to_all_dispatch_metadata`, `selective_reduce_combine`, global semaphores, links per axis) | Galaxy | orchestration | `tt/mesh.py::GLMMesh` (TT_CCL async ops, `all_to_all_cols`, `GLM53_CCL=generic` fallback) | equivalent | **reuse as-is (done)** |
| 18 | Generator: traced per-length prefill | `models/demos/llama3_70b_galaxy/tt/generator.py::_easy_trace_prefill` / `_prepare_trace_prefill` / `_record_trace_prefill`: trace key `(prefill_seq_len, batch_size, sp0|sp1)`, page tables padded to a fixed max so one trace serves any prefix length, compile every bucket first then record, `warmup_model_prefill`; `models/tt_transformers/tt/generator.py` (same mechanics + chunked prefill `max_prefill_chunk_size`, power-of-2 batched prefill buckets, prefill sampling traces) | Llama-3 70B / tt_transformers on the Galaxy and TG | orchestration | `demo/text_demo.py` + `tt/mesh_model.py`: traced decode only; prefill eager and chunked (`FunctionalDecoder.prefill_forward`, `start`) | the whole prefill trace machinery | **reuse with adapter (traceprefill agent)** |
| 19 | Generator: decode trace, row-batched users | `deepseek_v3/tt/model/row_batched_model.py`, `generator.py::_decode_step_tt` (rows = independent user batches, one trace) | Galaxy | orchestration | `tt/mesh_model.py::capture_decode_trace` (rows = user groups) | none | **reuse as-is (done)** |
| 20 | Caches (paged latent, index) | `ttnn.experimental.paged_update_cache / paged_fill_cache / paged_fused_update_cache` (DSv3 mla1d) | Galaxy | kernel | same ops (`tt/dsa.py`, `tt/indexer.py`; one row per `paged_update_cache` call: multi-row writes race) | none | **reuse as-is (done)** |
| 21 | Weight loading / dequant | `deepseek_v3/utils/hf_model_utils.py::dequantize_weight_tensor`, `LazyWeight` cache | any | host | `reference/checkpoint.py`, `GLMMesh.upload_cached` tensorbins | none | **done** |
| 22 | Decode MoE alternative | `models/common/modules/moe/tt_moe_decode.py::TTMoEDecode` (`all_to_all_dispatch_metadata` -> `moe_compute` -> `deepseek_moe_fast_reduce_nc_fused` -> RS; YAML `configs/glm5.yaml` is `mesh_shape [16, 8]`, bfp4 only, no clamp) | 16x8 quad Galaxy (and TG) | orchestration | not used | mesh 16x8, bfp4-only, no swiglu clamp | **reject** (wrong mesh / dtype policy; row 10 instead) |

Rejected by the filter (not Galaxy orchestration): `qwen38_27b_qb2/tt/{generator,generator_vllm,cache_initialization,
decoder_tp,precision}.py` (QB2 1D TP, batch buckets 1/8/16, bfp4 LoFi baseline) and `blackhole/qwen36` (single chip);
their only kernel-level content for us is rows 2-4.

## 3. Ranking by expected gain

Prefill (today 168 ms per MoE layer at S = 1024 per row; the compute roofline is ~1-2 ms):
1. row 11, d_p dispatch-based MoE prefill (`unified_routed_expert_moe` + `dispatch/combine_fabric2d`): the MoE is
   ~90 % of a prefill layer; this is the only item that moves prefill from 0.3 % of roofline towards 30-50 %.
2. row 18, traced per-length prefill (llama3_70b_galaxy / tt_transformers mechanics): removes host dispatch from the
   128-token TTFT (919 ms eager today; the traced step would be the device time).
3. row 6, slab-local SP attention with the `_sparse_mla` heads -> sequence a2a: removes the 2x head padding of
   `sparse_sdpa` at TP4 and the 8x redundant per-row DSA of v1 (slabattn).
4. row 13 / 16 for prefill rows: measure `TtMHCWrap` and the norms at T >= 1024 before changing anything (prefillprof).
5. row 5, `flash_mla_prefill` for the dense rows: small, last.
KDA prefill: nothing to gain from siblings (row 1 is already the per-channel reference; the GDN op is rejected).

Decode (today 76.45 ms / step at 32 users; the design floor is ~30 ms):
1. row 10, EP32 experts after `deepseek_v3/tt/moe.py` with our `expert_stream` kernels: the MoE sublayer is
   3.2-6.9 ms of a ~8-12 ms layer at 32 users / row against a ~1.2 ms budget.
2. row 7, user-partitioned latent cache + a2a (adopted): the 1 M-context memory enabler (cache / 4 per chip).
3. row 16, sharded residual + distributed norms (phase 2c): after the mHC custom ops, after EP32.
KDA decode: keep ours (row 4: the adapter is 10-14x slower than the block).

## 4. Integration order, effort, gating tests

| # | swap | owner / status | effort (h) | gate (test, bar) |
|---|---|---|---|---|
| 1 | EP32 decode experts (row 10): `tt/moe.py::TtMoE(ep=32)` = all_gather_cols -> exact router -> RM dispatch -> `expert_stream` (pairs_per_bank 8 at I = 2048) over the dispatched rows -> combine -> sum over K -> reduce_scatter_cols | OPT_EXPERTS (in progress, `arg/glm53_optexperts`) | 16-24 | `tests/test_moe_ep.py` (exact vs hybrid PCC >= 0.99999, 36 experts on the (2,4) block), `tests/test_mesh_moe.py`, `tests/test_a2a_perf.py` (dispatch + combine <= ~150 us at 256 tokens), `tests/test_perf_mesh.py` layer 4 B = 4 / 32 (<= 1.2 ms per MoE layer at 32 users / row), `tests/test_mesh_model.py` 45 layers (greedy id == host head) |
| 2 | prefill MoE on the d_p dispatch engine (row 11): adapter class in `tt/moe.py` wrapping `deepseek_v3_d_p/tt/moe/tt_moe.py` with our router output (288 experts, top-8, scale 2.5), `ClampedSiluGlu` limit 10, shared expert added outside, weights from `GLMMesh.upload_cached` | prefillprof -> OPT_EXPERTS | 24-40 | `tests/test_moe_block.py` (PCC >= 0.9999 vs the sparse block on real layer-4 routing, T = 32 / 256), `tests/test_mesh_sp_prefill.py` (vs HF R4097 rows >= 0.999), `tests/test_mesh_prefill_perf.py` (169 ms -> <= 10 ms per MoE layer at S = 1024 per row), `tests/test_prefill_chunked.py` (chunked == single-shot) |
| 3 | traced per-length prefill (row 18): `_prepare / _record` split, trace key (seq bucket, rows, sp0/sp1), persistent inputs (`copy_host_to_device_tensor`), page tables padded to the max | traceprefill (Phase A host side committed 0e9f733b69b) | 16-24 | `tests/test_prefill_chunked.py`, `tests/test_decode_trace.py` (unchanged replay PCC 0.9999), `tests/test_mesh_long_context.py` (TTFT per chunk; needle retrieved), `test_needle_smoke.py` |
| 4 | slab-local SP DSA prefill with the `_sparse_mla` a2a (row 6) | slabattn | 24 | `tests/test_mesh_sp_prefill.py`, `tests/test_prefill_chunked.py` (index blocks identical through `TtDSA.on_indices`), `tests/test_cache_1m.py` (selected-id overlap, PCC >= 0.9996 at 1 M) |
| 5 | user-partitioned cache everywhere (row 7): promote `test_mesh_sparse_partitioned.py`'s layout to the default decode path on the 8x4, k_chunk 32 vs 64 sweep | prefixcopy / coordinator | 16-24 | `tests/test_mesh_sparse_partitioned.py` (every (row, user, step) > 0.995), `tests/test_mesh_dsa.py`, `tests/test_decode_trace.py`, `tests/test_cache_1m.py` |
| 6 | mHC / norms for prefill rows (row 13 / 16): measure `TtMHCWrap` vs `TtMHCFast` at T = 1024 / 4096 | prefillprof | 8 (measure) + 8 | `tests/test_mhc.py` (PCC >= 0.99999 vs the wrap on real streams, T = 32 / 256 / 1024), `tests/test_mhc_perf.py` |
| 7 | embedding / head 1D over 32 chips (rows 14-15, memory only) | later | 8 | `tests/test_embedding_head.py` (embedding exact, head PCC >= 0.9999, replay bit-identical) |
| 8 | `flash_mla_prefill` for the dense rows (row 5) | later | 8 | `tests/test_dsa.py`, `tests/test_chunked_prefill.py` (dense rows PCC >= 0.9999 vs today's SDPA) |

Not planned: KDA prefill / decode swaps (section 1), the DSv3 gate (row 9), `TTMoEDecode` (row 22), any QB2 or
Wormhole orchestration.

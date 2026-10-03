# DeepSeek-V4.1-Flash decoder layer on Blackhole Galaxy — design

Status: draft for review. Decisions below were made in conversation; items marked **OPEN** are not decided.

## Goal

Bring up one full decode-time decoder layer of DeepSeek-V4.1-Flash on a 4x8 Blackhole Galaxy
(host 10.82.97.45), built on the GPT-OSS Blackhole work, with PCC and decode latency measured.
Real weights for layers 0, 2 and 20. Target for the full model: 50 tok/s/user (20 ms/token,
about 500 us/layer over 40 layers). The layer milestone reports measured per-layer latency;
it does not claim the end-to-end target.

Out of scope for this milestone: Engram, encoder-decoder KV sourcing, MTP/DSpark, vision tower,
FP4 KV cache, full-model bring-up, prefill.

## Reference

`/mnt/tt-data/ssinghal/deepseek-v41-flash/inference/model.py` (HF `DeepSeek-V4.1-Flash`).
Facts verified from that file and `config.json`:

- dim 5120, 40 backbone layers (+3 MTP), all layers MoE (384 routed, top-6, 1 shared, inter 2304).
- Router: `sqrtsoftplus(x @ W^T)`; a correction bias is used for selection only; weights come
  from the raw scores, normalised over top-k, times `route_scale` 1.5. No expert groups.
- Expert: `silu(clamp(w1 x, max=10)) * clamp(w3 x, +-10)`, routing weight multiplied BEFORE `w2`.
- Attention: MQA, 64 q heads, 1 KV head (K = V), head_dim 512, rope dim 64, q_lora 1280,
  grouped o-proj (8 groups, o_lora 1024), per-head sink, sliding window 128 (ring cache).
- `compress_ratios`: layers 0-1 = 0 (window only), 2-19 = 2, 20-39 = 1, 40-42 = 0 (MTP).
  Only `kv_source_layers` [2, 8, 14, 20] own a compressor and cache; other layers read the last
  source's cache. Only `index_source_layers` [2, 8, 14, 20, 24, 28, 32, 36] run an indexer; the
  others reuse the published top-k. `candidate_source_layer` 20 computes the level-1 block top-k
  (blocks of 8, 2048 candidates).
- mHC: 4 residual streams; each sub-block's coefficients are used by the NEXT sub-block, so a
  layer takes and returns `pre_mix` explicitly. Sinkhorn: 20 iterations, eps 1e-6.
- Checkpoint formats: attention / shared expert FP8 E4M3 with E8M0 scales (32x32 blocks); routed
  experts FP4 packed in int8 with E8M0 scales per 32; gate and compressor BF16; mHC FP32.

## Test layers

| Layer | Ratio | Exercises |
|---|---|---|
| 0 | 0 | window attention, MoE, mHC |
| 2 | 2 | compressor (overlap-free pooling with gate), indexer, KV + index source |
| 20 | 1 | ratio-1 compressor (plain projection), indexer, candidate-block source |

Layers that reuse another layer's cache (e.g. 3-7) are not tested in this milestone.
Real weights are dequantised on host (FP8/FP4 -> bf16) and cached to NFS.

## Parallelism

Mesh 4x8 (rows x cols). MoE is EP=32 on both axes (12 experts/device), dispatch along rows
(`cluster_axis=0`), partial-sum reduction over columns, `num_links=2`, as in GPT-OSS.

A config switch `attn_parallelism` selects the attention layout:

| | `tp_heads` | `dp_users` |
|---|---|---|
| Users per row | 4 (batch 16) | 8 (batch 32) |
| Heads | 8 q heads per column, single KV head replicated | all 64 heads, one user per column |
| KV cache | replicated 8x | once per user |
| Residual layout | hidden/8 per column | whole hidden per user |
| Around MoE | all_reduce over columns | all-gather before, reduce-scatter after |
| Attention weights | sharded 1/8 | replicated |

Both modes share the router, experts, shared expert, mHC and norms.
Budget estimate (not measured): expert weight reads ~0.17 ms/layer; `dp_users` attention weight
reads ~0.17 ms/layer at bfp8. 50 tok/s/u looks tight in `dp_users`; measure both.

## Verified on Blackhole (2026-10-01)

- GPT-OSS demos on the rebuilt tree: prefill_128 / 128k pass; batch128 passes at 36 tok/s/user.
- Paged SDPA decode at 64 q heads / 1 KV head / d=512 / sink / window 128: PCC 0.9999 (test_sdpa_decode_d512.py).
- MoE block (`tt/moe_block.py`) on 4x8, EP=32: PCC 0.994 / 0.988 / 0.977 for layers 0 / 2 / 20.
  Required: (a) a one-line fix to the `generalized_moe_gate` kernel (removed `transpose_wh_*` API), (b) generic
  reduce_scatter with a DRAM fast-reduce output (fused path needs 8 shards <= 2 x num_links, BH has 2 links),
  (c) the router precision fix: the gate op sums `score + bias` in bf16 and the V4.1 correction bias is ~10-14,
  so the bias is shifted by a calibrated per-layer cutoff (`tt/router.py`). Without it only 12-39% of tokens
  get the reference expert set.

## Reuse

MoE block base is the generic, config-driven decode MoE, not the GPT-OSS wrapper:

- `models/common/modules/moe/tt_moe_decode.py` (`TTMoEDecode`): `all_to_all_dispatch_metadata ->
  moe_compute -> deepseek_moe_fast_reduce_nc_fused -> reduce-scatter`, with a replicated shared expert,
  driven by a YAML (`configs/*.yaml`). Existing test: `models/common/tests/modules/moe/test_tt_moe_decode.py`.
- `models/common/modules/moe/tt_moe_gate.py` (`TTMoEGate`): `n_group=1`, k=6, `sqrtsoftplus`,
  selection-only `score_correction_bias`, `routed_scaling_factor`; 257-512 experts use a 2-block kernel path.
  384 experts is supported. This matches the V4.1 gate (`model.py: Gate`).
- NOTE the repo's `configs/deepseek_v4_flash.yaml` is a DIFFERENT, smaller model (hidden 4096, inter 2048,
  256 experts, 43 layers, hash routing in layers 0-2). V4.1-Flash needs its own YAML:
  mesh [4, 8], cluster_axis 0, hidden 5120, inter 2304, 384 routed, 1 shared, k 6, scale 1.5, SILU.
- `models/demos/gpt_oss/tt/experts_throughput/moe_compute.py` (the cherry-picked Blackhole wrapper) stays
  as reference for CCL/mux-core placement and the batch-128 SDPA grid fix.
- `MeshConfig`, `CCLManager` from `models/demos/gpt_oss`.
- `models/demos/deepseek_v3_d_p/tt/`: `mhc/tt_mhc.py`, `mla/compressor.py`, grouped `_o_proj`
  (prefill-shaped; adapt to decode).
- ops: `ttnn.transformer.sparse_sdpa` (Blackhole), `indexer_score_dsa`, `topk_large_indices`,
  `paged_scaled_dot_product_attention_decode`, `deepseek_prefill.mhc_split_sinkhorn`.

## Gaps to build

1. Decode CSA path: indexer score -> block-max (8) -> level-1 candidate blocks -> top-512 ->
   `sparse_sdpa` with the 128-slot window appended as indices. `indexer_score_dsa` pools in
   multiples of 32, not 8.
2. Stateful decode-time compressor (partial group carried between steps).
3. MQA decode SDPA at head_dim 512, 64 q heads, 1 KV head, sink — untested.
4. `TTMoEDecode` at hidden 5120 / inter 2304 / 384 experts on a 4x8 Blackhole mesh — untested.
   Its default `fast_reduce_output_memory_config` and post-combine tilize grid were sized for hidden 7168
   (7 cores x 1024 wide) and must be re-derived for 5120. The config table lists the swiglu clamp as
   "routed only", but V4.1 `model.py` clamps the shared expert too (`Expert(..., swiglu_limit=...)`) —
   check whether the shared-expert path of `TTMoEDecode` can clamp, else it is a PCC loss.
5. mHC at decode shape (T <= 32).
6. FP4/FP8 weights run dequantised (bf16 -> bfp8/bfp4) for now; no device block-scaled matmul.

## Phases

0. De-risk op tests on Blackhole (items 3, 4, 5 above). Item 4 is `test_tt_moe_decode` with a new
   `deepseek_v41_flash.yaml`.
1. Torch reference of one layer per type, from `model.py`, pure torch (no custom kernels),
   plus an fp4/fp8-fake-quant variant matching the reference's rounding.
2. MoE block (router, routed experts, shared expert) — PCC vs reference.
3. Attention, `tp_heads` — window, q-LoRA, grouped o-proj, compressor, CSA decode.
4. mHC + full layer for ratio 2 and ratio 1 layers.
5. `dp_users` mode behind the switch.
6. Latency: per-layer decode time for both modes, extrapolated x40 as a rough guide only.

## Testing

- PCC per block and per layer against the torch reference with real layer 0 / 2 / 20 weights and
  real activations (layer 0: embedding of a short prompt; layers 2 and 20: chained reference output).
  Random weights are not used for PCC.
- Report PCC against both the unquantised reference and the fake-quant reference.
- All device tests run on 10.82.97.45. `tt-smi -r` / `tt-smi -glx_reset` are allowed there.

## Risks

- SDPA decode at head_dim 512 may not fit L1.
- `moe_compute` CB/L1 fit at the new shapes; mux/combine cores overlapping the attention grid
  (this was a silent-corruption bug in GPT-OSS).
- Decode collectives at hidden 5120 (all_reduce over 8 columns).
- `dp_users` weight-read latency.
- Reference layers 3-7 etc. depend on another layer's state.

## OPEN

- Whether `moe_compute` can take the routing-weight multiply before `w2`, or whether a different
  activation fusion is needed.
- Sequence-sharded KV (flash-decode style) as a later alternative to replication.


## C.sliding_moe.attn_norm (layer 0)

- `tt/norm.py:TtRmsNorm`: one local `ttnn.rms_norm` per chip (the full 2880 hidden is on every chip, so no
  collective), gamma bf16 ROW_MAJOR [1, 1, 90, 32] replicated, DRAM, bf16 out. Compute config from settings
  NORM_FIDELITY (HiFi4) and NORM_FP32_ACC (True), both in MAX_PRECISION. The same class serves ffn_norm (hooks map
  `ffn_norm` to post_attention_layernorm.weight) and can serve final_norm.
- `tt/layout.py`: harness-boundary helpers only. `rows_to_device` splits [S, H] by sequence over axis 0 (S/8 rows
  per chip row) and replicates it over axis 1; `rows_to_host` reads column 0 of each row back.
- `bringup/hooks.py:device_component` sets `ttnn.CONFIG.enable_fast_runtime_mode = False`: in fast mode ttnn's
  FastOperation skips PRE_OPERATION_HOOKS, and the device proof counted 0 ops (device_ops_attn_norm = 0).
- Gate: pcc 0.999998, auto_margin_min 3.21 (closest: vs_cpu rel 0.0019 of limit 0.0061), device_ops 1, host
  FLOPs 0, deterministic.
- Re-run: `bfw job check --task C.sliding_moe.attn_norm`.

## C.sliding_moe.attention (layer 0)

- `tt/attention.py:TtAttention` (plan.md step 2): fused QKV linear + bias (column c: 16 q | 2 k | 2 v heads, q/k rows
  permuted to meta order at load) -> `nlp_create_qkv_heads` -> `deepseek_prefill.rotary_embedding_indexed` (YaRN
  table `TtRope`, block-cyclic, built once) -> `update_padded_kv_cache` (bfp8, served ND-sharded cache from
  `tt/kv_cache.py`) -> `ring_joint_scaled_dot_product_attention` (SP ring over axis 0, sinks / 2^-3, window 128 or
  None, q128/k128, compact halo gather buffers from `tt/ccl.py:TtCcl`) -> `nlp_concat_heads` -> o_proj -> fp32
  `all_reduce` over TP -> + o bias, bf16.
- Two precision fixes that made the auto checks pass (first run: bias +0.0083 > 0.004, column 777 rel 0.153 > 0.105):
  - QKV_FP32 (default True): QKV, heads and RoPE in fp32, then fp32 -> bfp8 into the cache. The device's
    `ttnn.typecast` bf16 -> bfp8 scales values by about +0.28 % (fp32 -> bfp8 does not; a host from_torch bfp8 is
    unbiased). That K/V scale alone gave a +0.5 % output-norm bias. Q is cast back to bf16 for the ring SDPA.
  - o_proj weight stored [out, in] and applied with `transpose_b=True`: bfp8 blocks then run along the in-features.
    Stored as wo.T, the blocks ran along the out-features, and small output columns (777: norm 12 vs mean 28) lost
    precision (col rel 0.145 in a CPU model, 0.153 on device).
- Ring SDPA with sinks runs only with fp32 dest off (TT_FATAL), so SDPA_FP32_ACC is False. Its own error vs fp32 on
  the same cache is rel 0.009. q64 vs q128 changed nothing.
- Gate: pcc 0.99971, vs_golden rel 0.0242 (limit 0.0280), bias +0.0025 (0.004), auto_margin_min 1.006: the `big`
  input (x2) at rel 0.0451 against 0.0454 is the closest check. 14 device ops, host FLOPs 0, deterministic.
- Component boundary (`hooks._AttentionBoundary`) loads the golden KV prefix into a fresh served cache each call.
- Re-run: `bfw job check --task C.sliding_moe.attention`.

## C.sliding_moe.attn_residual (layer 0)

- `tt/residual.py:TtResidualAdd`: one local `ttnn.add` (bf16 out, DRAM). Both operands are the residual stream's
  layout ([1, 1, S/8, 2880] per chip, rows split over axis 0, replicated over axis 1), so no collective. The same
  class serves mlp_residual (hooks map both steps to it).
- `hooks._HostBoundary` now takes any number of [S, H] inputs (`*xs`), each sent with `rows_to_device`.
- Gate: pcc 0.9999985, auto_margin_min 1.97, device_ops 1, host FLOPs 0; attn_norm and attention regressions PASS.
- Re-run: `bfw job check --task C.sliding_moe.attn_residual`.

## C.sliding_moe.router (layer 0)

- `tt/router.py:TtRouter` (plan.md step 5): `ttnn.linear` (weight [2880, 128] + bias, bf16, replicated; fp32 out,
  HiFi4 fp32 dest) -> `ttnn.topk` k 4 sorted -> `ttnn.softmax` over the 4 chosen logits (fp32) -> `ttnn.scatter`
  (fp32 ROW_MAJOR: scatter has no fp32 TILE path) onto a zeros table built at load (`prepare(rows)`, rows = chunk /
  8). Local per chip, no collective. `route(x)` returns the sparse pair (indices uint16 ROW_MAJOR [.., S/8, 4],
  weights fp32 TILE [.., S/8, 4]) for the experts step; `__call__` returns the dense [1, 1, S/8, 128] fp32 graph
  output.
- Settings ROUTER_FIDELITY (HiFi4), ROUTER_FP32_ACC (True), ROUTER_WEIGHT_DTYPE (bfloat16), all in MAX_PRECISION.
- Gate: pcc 0.999866; selection overlap vs golden 0.99917 (matched rows 0.99668, rel 0.0019, rowsum 0.0038),
  auto_margin_min 2.60, deterministic. Worked first time.
- Re-run: `bfw job check --task C.sliding_moe.router`.

## C.sliding_moe.experts (layer 0)

- `tt/experts.py:TtExperts` (plan.md step 6): EP over all 32 chips, 4 experts per chip, column-major mapping (column c
  = dispatch group of 8 chips, experts 32c..32c+31). `ttnn.topk` k 4 on the dense fp32 routing input -> uint16 indices
  -> `masked_bincount` + `offset_cumsum` (axis 0, global [1, 128] counts) -> `dispatch` (axis 0, bf16 ROW_MAJOR, Linear
  topology) -> tilize bf16 -> per local expert `extract` -> `ttnn.linear` gate_up ([out, in] bfp8 + transpose_b, fp32
  bias, fp32 out) -> clamp / sigmoid(1.702 g) / (up + 1) * glu in fp32 -> `ttnn.linear` down (bf16 out) -> `insert`
  -> `combine` (axis 0, init_zeros) -> `post_combine_reduce` (bf16 weights, dispatch table masks other groups) ->
  fp32 `all_reduce` over TP (axis 1) -> bf16. Stock TTNN only, no fused unified_routed_expert_moe (LoFi, bfp8 acts).
- Capacity factor 4 (= top-k): dispatch buffer 4 x 5120 + 3 x 32 rows per chip, no token can be dropped; extract cap
  = 8 x 640 = 5120 rows (a whole column's tokens to one expert).
- Weights from `reference.gpt_oss_ref.LayerWeights` (mxfp4 dequantized exactly, gate/up de-interleaved), uploaded per
  local expert with ShardTensor2dMesh dims (0, 1).
- Settings: EXPERTS_FIDELITY HiFi4, EXPERTS_FP32_ACC True, EXPERTS_MID_DTYPE float32, EXPERTS_REDUCE_FP32 True (all in
  MAX_PRECISION), EXPERTS_CAPACITY_FACTOR 4.
- `hooks._ExpertsBoundary` sends routing [S, 128] as fp32 TILE (the router's output dtype), x as bf16 rows.
- Gate (first try): pcc 0.999992, vs_cpu rel 0.0032 (limit 0.0119), vs_golden rel 0.0040, ratio [0.9945, 1.0049],
  auto_margin_min 2.71, 79 device ops, host FLOPs 0, deterministic. Regressions of the 5 earlier C tasks PASS.
- Re-run: `bfw job check --task C.sliding_moe.experts`.

## C.attn_moe.attention (layer 1, full causal)

- Same `tt/attention.py:TtAttention` with `window=None`. As-is it failed the auto checks: pcc 0.9978, vs_golden rel
  0.067 (limit 0.046), bias +0.0068 (0.004), `big` rel 0.165 (0.069), margin 0.42.
- Cause (CPU simulation of the step, fp32 SDPA): bfp8 K alone gives rel 0.063; V bfp8 0.005, bf16 Q 0.003. bfp8 on
  the prefix keys is 0.017, on the chunk's own keys 0.061. The bf16 SDPA intermediates add nothing measurable. The
  test's precision model rounds the golden prefix state to bfp8 but sees the chunk's fresh K / V in fp32. Reordering
  the head dim before bfp8 (meta, by magnitude) did not help (0.067 to 0.075).
- Fix, behind `ATTN_CHUNK_KV_BF16` (True, in MAX_PRECISION): on full layers SDPA reads a per-call bf16 copy of the
  slot. `ttnn.slice` (DRAM interleaved out: typecast rejects ND-sharded to interleaved) -> `typecast` bf16 ->
  `update_padded_kv_cache` with the chunk's K / V in bf16 -> ring SDPA on it (batch 0), bf16 gather buffers. The served
  cache still gets the bfp8 K / V as before. Sliding layers are unchanged. Cost per chip at the 56320 target:
  ~1.8 MB per copy, 14 MB per bf16 gather buffer (7 MB in bfp8).
- Result: rel 0.0306, bias +0.0042 (limit 0.004), margin 0.95. Then `SDPA_FULL_K_CHUNK` 512 (was 128, a full-layer
  K block of its own; sliding still needs 128): bias +0.0036, rel 0.0296, margin 1.03.
- Gate: PASS. pcc 0.99957, auto_margin_min 1.03 (bias on golden is the closest check), 22 device ops, host FLOPs 0,
  deterministic. All 8 earlier C regressions PASS (sliding attention too).
- Re-run: `bfw job check --task C.attn_moe.attention`.

## S.sliding_moe.02 (layer 0: attn_norm + attention on device)

- First run failed the swap step checks: swap_attention vs the CPU step on the same inputs had rel 0.0246 (limit
  0.02 = min(0.02, 2 x component error)), worst row 0.075 (0.05) and ratio [0.965, 1.038] (1 +- 0.02). The block out
  vs CPU attention had rel 0.0154 (0.01), and the margin was 0.53. The component gate passed because its limits are
  set against the golden.
- **Error sources, from a CPU simulation and an on-device stage probe** (rel of the attention output):
  - bfp8 on the chunk's own K: 0.020. The prefix K / V in bfp8 contribute only 0.003; the test's CPU step sees the
    chunk's K / V in fp32.
  - `rotary_embedding_indexed`: 0.0044 on q / k, with norms 0.2 % low. It runs with a 16-bit dest, and
    fp32_dest_acc_en TT_FATALs in program setup (`dfbs_with_entry.contains`). fp32 cos / sin tables changed nothing.
  - bfp8 QKV weights: V rel 0.005, q / k 0.0004 more.
  - The ring SDPA itself (bf16 intermediates, no fp32 dest with sinks): 0.007 against an fp32 SDPA on the same
    inputs. This one is left as is.
- **Fixes** (all Settings, in MAX_PRECISION):
  - `ATTN_SLIDING_KV_SPLIT`. Ring SDPA's sliding path accepts only bfp8 K / V (TT_FATAL "Chunked sliding attention
    requires BF16 Q and BFP8_B K/V"), so the bf16 slot copy of the full layers is not possible here. Instead SDPA
    reads a per-call copy of the slot that is 128 wide:
    - every row is [cache | 0];
    - the chunk's rows are [hi | lo], with hi = bfp8(x) and lo = bfp8(x - hi);
    - Q is [q | q].

    With that, q.k and P.v see hi + lo, which is within 1e-4 of fp32. The output halves are sliced and added. The
    served cache still gets plain bfp8 hi, written after the copy is built. The gather buffers are 128 wide.
  - `ROPE_FP32`. `rotary_embedding_indexed` only fetches this chunk's table rows: it is called with input ones, a
    zero rotation matrix and hi / lo bf16 tables, which is exact. The rotation itself is composed: matmuls of x's bf16
    hi / lo halves with the 64x64 meta rotation (fp32 dest), then x * cos + rot * sin in fp32. q / k rel drops from
    0.0044 to 0.0004.
  - `ATTN_QKV_WEIGHT_DTYPE` bfloat16. This costs 3.5 MB more per chip per layer.
- Probe after the fixes: module vs the CPU step rel 0.0088, row 0.029, ratio [0.989, 1.017], bias +0.0029.
- Re-run: `bfw job check --task S.sliding_moe.02`.

## S.sliding_moe.03 (layer 0: attn_norm + attention + attn_residual on device)

- No code change: the C.sliding_moe.attn_residual module (one local `ttnn.add` on the residual-stream layout) and the
  S.sliding_moe.02 attention passed together on the first run.
- Gate: PASS. pcc_swap_out 0.99998, swap_margin_min 1.25 (attention's step check is still the closest).
  attn_residual vs CPU step: rel 0.0017 (limit 0.005), row 0.0024, ratio [0.9995, 1.0015], bias +0.0005;
  block out vs CPU attn_residual rel 0.0027 (0.01). Device ops: attn_norm 1, attention 58, attn_residual 1; host
  FLOPs 0.
- Re-run: `bfw job check --task S.sliding_moe.03`.

## S.attn_moe.02 (layer 1: attn_norm + attention on device)

- First run failed the swap step check: swap_attention vs the CPU step had rel 0.0249 (limit 0.02), worst row 0.111
  (limit 0.05) and ratio [0.971, 1.050] (limit 1 +- 0.02); the margin was 0.40. The golden prefix state is bf16, and the
  CPU step attends it at that precision.
- CPU simulation (scratch sim.py, fp32 SDPA, every 5th query row of chunk 1): bfp8 on the prefix K alone gives rel
  0.0170, row 0.068 and ratio [0.975, 1.030]. Prefix V in bfp8 adds 0.001, the chunk's K / V in bf16 add 0.0045, q in
  bf16 0.0025. With the prefix K as bfp8 hi + bfp8 lo: 0.0001.
- Fix: `ATTN_FULL_K_LO` (True, in MAX_PRECISION). On full layers the module also keeps a K residual cache
  `cache_k_lo`, with the same layout as the served cache and lo = bfp8(k - bfp8(k)), written beside the served K
  through `update_padded_kv_cache`. SDPA's per-call bf16 slot copy then holds the prefix K as hi + lo (each typecast to
  fp32 and summed, then bf16). The served cache is unchanged: plain bfp8, per the contract. `TtAttention.__call__` takes
  `cache_k_lo=` and asserts it is given exactly when `module.k_lo` is set. `_residual` is shared with the sliding
  split. It costs one more [n_batch, 2, 7040, 64] bfp8 tensor per full layer per chip, which is ~1 MB per slot-layer.
- `hooks._AttentionBoundary` loads the K residual of the golden prefix too, as host - `kv_cache.bfp8_round(host)`
  (ttnn's host bfp8 conversion, the same as cache_from_host).
- **The assemble task must allocate `cache_k_lo` for each full-attention layer and slot** (same `cache_from_host` /
  allocator layout) and pass it on every call. It is model state outside the served cache: when the decode side or a
  migration pulls the served cache, it does not get lo, and prefill only needs it for its own later chunks.
- Gate: swap_attention vs CPU rel 0.0170 (0.02), row 0.043, ratio [0.992, 1.015], bias +0.0034. Block out vs CPU
  attention rel 0.0068 (0.01). swap_margin_min 1.158 (the closest check is rel against the CPU step). pcc_swap_out
  0.99997. Device ops: attention 53, attn_norm 1; host FLOPs 0.
- Re-run: `bfw job check --task S.attn_moe.02`.

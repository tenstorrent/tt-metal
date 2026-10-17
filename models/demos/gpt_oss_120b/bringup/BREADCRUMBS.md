
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

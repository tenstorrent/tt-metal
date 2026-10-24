# qwen38_27b bring-up breadcrumbs

## C.linear.attn_norm (layer 0)

- `tt/norm.py:TtRmsNorm`: one fused `ttnn.rms_norm` on the local [1, 1, 640, 5120] bf16 rows (SP 8 on mesh axis 0,
  replicated on axis 1), no collective. The `1 + w` gamma is added in fp32 at load and replicated as a
  [1, 1, 1, 5120] fp32 TILE tensor. HiFi4 + fp32 dest (`NORM_FIDELITY`, `NORM_FP32_ACC`, `NORM_WEIGHT_DTYPE` in
  `tt/settings.py`).
- A ROW_MAJOR [1, 1, 160, 32] gamma overflowed L1 (1.85 MB of CBs): a row-major affine turns off the op's streaming
  kernel for a TILE input. A TILE gamma fixes it.
- `bringup/hooks.py`: `device_component` builds the module from the checkpoint's `input_layernorm.weight`. The host
  boundary (`_rows_to_device` / `_rows_to_host`) splits rows over mesh axis 0 and reads column 0 of each mesh row.
  `_ops_visible` runs the call with `ttnn.CONFIG.enable_fast_runtime_mode = False` for that call only, because
  FastOperation never calls ttnn's pre-operation hooks. Without it the gate's device proof counts 0 ops.
- Gate: PCC 0.999998, rel 0.0020 vs CPU (limit 0.0064), auto_margin_min 3.21, 1 device op, determinism
  bit-identical. Re-run: `bfw job check --task C.linear.attn_norm --gate-only`.

## C.linear.attention (layer 0)

- `tt/linear_attention.py:TtGatedDeltaNet`: the qwen36 SP fold (`qwen36/tt/sp.py`) plus deepseek's conv exchange
  (`deepseek_v3_d_p/tt/kda/kda.py`), copied and adapted to 8x4 (SP 8 on axis 0, TP 4 on axis 1; per column 4 k / 12 v
  heads, conv channels 2560). The ops: qkv|z linear (bf16) and a|b linear (fp32; a and b each padded to 32 columns),
  `kda.qkv_causal_conv1d_silu_with_carry` (fabric exchange on axis 0, carried conv history in, new carry out),
  g/beta in fp32, `chunk_gated_delta_rule` on v' = [0 | v] from [I | 0], `all_gather` of [A | B] (bf16) over axis 0,
  `kda.chain_affine_transforms` from the slot's carried recurrent state, [U | O0] @ [[S_entry], [I]], gated
  `rms_norm` (eps 1e-6) * silu(z), out_proj row-parallel, fp32 `all_reduce` over axis 1, typecast to bf16.
- Weight layout matters for bfp8 (one exponent per 16 values along a tile row): out_proj stays [N, K] and is read with
  `transpose_b=True`. As [K, N], output column 3993 was off by rel 0.50, because its block neighbour 3994 is about 20x
  larger at every linear layer. The in-projections stay [K, N]: with transpose_b the layer-3 input (one huge channel)
  failed rel 0.038 > 0.033.
- `GDN_DELTA_CHUNK` 64 (default) vs 32: the op decays its fp32 state once per chunk with `(dl*I) @ S` on the FPU,
  which reads fp32 operands with a short mantissa (measured: an fp32 identity matmul returns S x 0.99928, about half
  the values truncated to TF32). That compounds over the 20 chunks of one 640-row SP segment and the 8 segments of
  the chain on slow-decay heads. Post-chunk state vs golden: rel 0.016 / worst head 0.029 at 64, 0.029 / 0.057 at 32.
  At 64 the op takes rank-4 q/k/v, so q/k are l2-normalised in the module.
- The hook builds the state from `ctx.extra["state_prefix"]` (none at start 0) and writes the post-chunk state to
  `ctx.extra["state_out"]` (conv [conv_dim, 3], recurrent [48, 128, 128]), in the reference's layout.
- Gate: PCC 0.999994; vs CPU rel 0.0034 (limit 0.015); layer-3 input rel 0.0175 (limit 0.033); auto_margin_min 1.56;
  49 device ops; bit-identical determinism. Re-run: `bfw job check --task C.linear.attention --gate-only`.

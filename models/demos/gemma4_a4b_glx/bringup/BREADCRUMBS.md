# Bring-up breadcrumbs: gemma4_a4b_glx

## Reference (R.1, R.2, R.3)

Wrote `reference/gemma4_ref.py` from the prior `gemma4_a4b_d_p` package. The model is a
30-layer text decoder with sliding-window (1024) and global full attention alternating,
128 experts per layer with top-8 routing, and dense MLP + MoE in parallel per layer.

Block graph: 14 steps (attn_norm, attention, post_attn_norm, attn_residual, ffn_norm,
mlp, post_mlp_norm, router, moe_norm, experts, post_moe_norm, ffn_combine, post_ffn_norm,
ffn_residual).

### Decisions

- **Experts accumulation matches HF grouped_mm.** HF defaults to `grouped_mm` experts
  implementation, which accumulates with reshape+sum, not index_add_. Using index_add_
  caused compounding FP error (~6e-8/layer) that flipped MoE routing at L23. Rewrote
  experts_forward to sort by expert, per-expert F.linear, inverse-permute, reshape+sum.

- **route() returns dense [S, 128] tensor.** Returns a dense routing-weight tensor with
  zeros except top-8 weights scattered at their expert indices. The previous packed
  `[S, K, 2]` format carried expert indices as floats, which the component harness
  corrupted during perturbation sweeps (mixed permutes, small rescales, big x2).
  experts_forward() now derives selection via `torch.topk(dense, k=top_k)`.

- **MKL_CBWR=COMPATIBLE removed.** Previously set at import time in gemma4_ref.py for
  deterministic sgemm. Lint forbids environment reads/writes in the package. Removed; R.2
  parity unchanged (the expert padding fix from the earlier attempt is what matters).

- **reference_hooks.py uses spec.hf_dir.** Model package must not import bringup_fw.
  Replaced `from bringup_fw.reference.golden import hf_path` with `str(spec.hf_dir)`.

### Re-run

```bash
source /data/dgolubovic/repos/tt-agentic-bringup-sc-gemma-glx/.bringup/runs/prefill-stepwise/20261013-011638-gemma4-glx-setup3/artifacts/setup/prepare/env.sh
bfw tasks run R.1 R.2 R.3
```

## C.sliding.attn_norm

Implemented `TtRMSNorm` in `tt/rms_norm.py` for the Gemma-4 RMSNorm (`x * w * rsqrt(mean(x^2) + eps)`,
weight is plain `w`, not `1 + w`). Weight from `model.language_model.layers.{layer}.input_layernorm.weight`,
replicated across all 32 chips. TILE layout weight, HiFi4 + fp32 accumulation.

`hooks.py` dispatches all norm steps through `_NORM_WEIGHTS` dict and shared `_norm_module`/`_host_fn`
helpers. The same module handles all 7 norm step types.

### Decisions

- **Disabled fast_runtime_mode in device_component.** `ttnn.CONFIG.enable_fast_runtime_mode = True` (default)
  makes `FastOperation.__call__` skip `PRE_OPERATION_HOOKS`. `DeviceProof` uses `register_pre_operation_hook`,
  which only fires in the slow `Operation` path. Without this fix, `device_ops_attn_norm = 0` despite the op
  running correctly on device (PCC 0.999996).

### Re-run

```bash
source /data/dgolubovic/repos/tt-agentic-bringup-sc-gemma-glx/.bringup/runs/prefill-stepwise/20261013-011638-gemma4-glx-setup3/artifacts/setup/prepare/env.sh
bfw job check --task C.sliding.attn_norm
```

## C.sliding.attention implement (attempt 1)

New `tt/attention.py`:
- `TtSlidingAttention`: the chunk is replicated on all 8 mesh rows and the heads are split over the 4 columns (TP=4).
  Column c holds Q heads 4c..4c+3 and KV heads 2c, 2c+1.
  - Fused per-chip QKV weight [2816, 2048], bfp8, sharded with `ShardTensor2dMesh(dims=(None, -1))`. Then
    `nlp_create_qkv_heads(4, 2)`.
  - q/k/v are cast to fp32. The per-head `rms_norm` (q_norm, k_norm, unscaled for V) and the rotate-half
    `experimental.rotary_embedding` run in fp32 at HiFi4 + fp32 acc. The fp32 cos/sin tables are built once for
    max_seq and sliced on the device.
  - Q goes to bf16. K and V go fp32 -> bfp8 and are written with `fill_cache(update_idx=start)`.
  - SDPA runs over the square [cache rows start-hist .. start+S) with hist = min(start, 1024). Q is front-padded with
    hist dummy rows, whose outputs are sliced off. Settings: `scaled_dot_product_attention(is_causal,
    sliding_window_size=1024, scale 1.0)`, q256/k128, HiFi4 + fp32 acc.
  - Then `nlp_concat_heads`, row-parallel o_proj ([1024, 2816] bfp8 per chip) and `all_reduce(cluster_axis=1)`.
- `TtKVCache`: [1, 8, max_seq, 256] bfp8 per layer, heads sharded over the columns and replicated over the rows. Built
  from the host prefix.
- `tt/settings.py`: ATTN_MATMUL_FIDELITY, ATTN_MATMUL_FP32_ACC, SDPA_FIDELITY, SDPA_FP32_ACC (all four in
  MAX_PRECISION), SDPA_Q_CHUNK and SDPA_K_CHUNK.
- `hooks.py`: `device_component("attention")` builds the module from the layer's six self_attn tensors only. On each
  call it builds a fresh device cache from `ctx.extra["state_prefix"][:, :prefix_len]`. Global layers raise
  NotImplementedError.

### Decisions
- **Departed from the plan's SP=8 ring_joint SDPA.** I used the prior's proven TP-by-head layout with the rows
  replicated. It is correct but redundant: all 8 rows compute the same chunk. ring_joint SDPA on a chunk-major
  CP-sharded cache, with its logical_n / kv_actual constraints, is the performance follow-up. The module interface
  `(x, cache, start)` stays the same.
- **K/V reach bfp8 from fp32, never from bf16.** A first run scored PCC 0.999965 but failed the median row-norm bias
  check: +0.0043 against a 0.004 limit, with rel 0.0093.
  - A stage-by-stage probe showed SDPA itself unbiased: +0.0001 against the same inputs.
  - `rotary_embedding` at its default compute config grew norms by +0.2%. At HiFi4 + fp32 the change is -0.02%.
  - The device `typecast` bf16 -> bfp8 grew K norms by +0.29%, and every row went up. Typecast, clone, mul and
    to_layout all behave the same, even through an intermediate fp32 typecast.
  - fp32-valued data -> bfp8 is unbiased (-0.03%), as is host rounding.
- After both fixes: PCC 0.999968, rel 0.0079, bias +0.0007, margin 1.52.

### Re-run
```bash
source /data/dgolubovic/repos/tt-agentic-bringup-sc-gemma-glx/.bringup/runs/prefill-stepwise/20261013-011638-gemma4-glx-setup3/artifacts/setup/prepare/env.sh
scripts/run_safe_pytest.sh --run-all generated/bringup/gemma4_a4b_glx/tests/test_c_sliding_attention.py
bfw job check --task C.sliding.attention
```

### Restore (run 20261013-175448-gemma4-glx-14)
I re-applied the attempt above unchanged; it had passed under an older framework.
- I re-added its 6 Settings in `tt/settings.py`, which had not been saved: ATTN_MATMUL_FIDELITY=HiFi4,
  ATTN_MATMUL_FP32_ACC=True, SDPA_FIDELITY=HiFi4 and SDPA_FP32_ACC=True (all four in MAX_PRECISION), plus
  SDPA_Q_CHUNK=256 and SDPA_K_CHUNK=128.
- `bfw job check --task C.sliding.attention`: PASS. pcc 0.999968, margin 1.52, 32 device ops, 0 host FLOPs.
  The C.sliding.attn_norm regression also passed.

## C.sliding.attn_residual (run 20261013-175448-gemma4-glx-14)
- `tt/residual.py`: `TtResidualAdd(a, b)` = one `ttnn.add` (bf16, DRAM). Inputs are replicated, so no CCL is needed.
- `hooks.py`: `_host_fn2` wraps a two-input device module for the component harness. `device_component` returns it
  for `attn_residual`. The global block's attn_residual is the same add, so it takes the same path.
- Check: PASS. pcc 0.999996, auto_margin_min 1.95, 1 device op, 0 host FLOPs. The attn_norm, attention and
  post_attn_norm regressions also passed.
- Re-run: `bfw job check --task C.sliding.attn_residual`

## C.sliding.mlp (run 20261013-175448-gemma4-glx-14)
- `tt/mlp.py`: `TtMLP` is the dense GeGLU MLP, `down(gelu_tanh(gate(x)) * up(x))`, with TP=4 over the mesh columns
  (the plan's layout). The rows are replicated, as in attention.
  - gate and up are column-parallel and down is row-parallel, all bfp8 weights. Each chip's 528-wide intermediate
    slice is zero-padded to 544. The padded gate/up columns give gelu(0)*0 = 0, and the padded down rows are zero.
  - The ops: two `ttnn.linear` (bf16 out), `ttnn.gelu(variant=Tanh)`, `ttnn.mul`, `ttnn.linear` down, then
    `all_reduce(cluster_axis=1)`.
- `tt/settings.py`: MLP_MATMUL_FIDELITY=HiFi4 and MLP_MATMUL_FP32_ACC=True (both in MAX_PRECISION), and
  MLP_GELU_VARIANT=Tanh (choices Tanh/Accurate; the reference uses tanh).
- `hooks.py`: `device_component("mlp")` loads `mlp.{gate,up,down}_proj.weight` of the layer and wraps it with
  `_host_fn`. The global block's mlp has the same shape, so it takes the same path.
- Frozen test: pcc 0.999982, rel 0.0059 against the calibrated 0.0110, auto_margin_min 1.74, 6 device ops, 0 host FLOPs.
- Re-run: `bfw job check --task C.sliding.mlp`

## C.sliding.router (run 20261013-175448-gemma4-glx-14)
- `tt/router.py`: `TtRouter` is replicated, so every chip of a row routes its own tokens and no CCL is needed.
  - The ops run all in fp32 after a typecast of the bf16 input: `rms_norm` (no weight), then `linear` with
    W' = (proj * router.scale * 2816^-0.5)^T (the scale is folded in at load, W' is fp32), then `softmax(numeric_stable)`.
  - Then `topk(8)` gives the ids, and a bf16 `scatter` of ones at those ids into zeros gives an exact 0/1 mask
    (scatter has no fp32 TILE path).
  - Last, dense = probs * per_expert_scale * mask / sum(probs * mask). This equals renorm then per_expert_scale[id].
    Every op is 128 wide, so there is no reduction over a padded 8-wide tile.
  - It returns `(dense [1,1,S,128] fp32, ids [1,1,S,8] uint32)`. The experts step can use the ids.
  - The zeros/ones scatter operands are built once per S on the device (`zeros_like`/`ones_like`) and cached.
- Why fp32 W', from a CPU study on the layer-0 golden:
  - bfp8 W fails the selection checks (overlap 0.9947, matched rel 0.0057).
  - bf16 logits fail too (matched rel 0.0065 > 0.005).
  - bf16 h input costs only rel 0.0006, so h_mid goes up as bf16. That is exact on the golden.
- A first version built the mask as `probs >= topk_values[..., 7]`. It failed: 30-46 rows selected 7 or 9
  experts. The topk values do not compare equal to the fp32 probs on this Blackhole build, so build the mask from the ids.
- `tt/settings.py`: ROUTER_FIDELITY=HiFi4 and ROUTER_FP32_ACC=True, plus ROUTER_WEIGHT_DTYPE=float32
  (choices float32/bfloat16/bfloat8_b). All three are in MAX_PRECISION.
- `hooks.py`: `_router_module` loads `router.{proj.weight, scale, per_expert_scale}`. `_router_fn` uploads h_mid as
  bf16 and returns the dense matrix. `device_component("router")` uses them. The global router is the same op.
- Frozen test results:
  - pcc 0.999816, overlap 0.99841 vs golden (0.99854 vs CPU), matched rel 0.0024 vs golden, row sums 0.0023.
  - auto_margin_min 2.12, 13 device ops, 0 host FLOPs, deterministic.
- Re-run: `bfw job check --task C.sliding.router`

## C.sliding.experts (run 20261013-175448-gemma4-glx-14)
- `tt/experts.py`: `TtExperts(x, routing)` is composed from stock TTNN ops. The plan's `ttnn.bringup` dispatch/combine
  and `unified_routed_expert_moe` GeluTanh fork are not on this tt-metal.
  - Each mesh row takes its S/8 tokens with `mesh_partition(dim=2, cluster_axis=0)`, a local slice.
  - Column c holds experts 32c..32c+31 (EP=4) and takes its 32 columns of the dense routing with
    `mesh_partition(dim=3, cluster_axis=1)`.
  - Per chip, dense over its 32 experts:
    - gate = x @ Wg [2816, 32*704] and up = x @ Wu, both bfp8 with expert-major 704-wide blocks.
    - h = gelu_tanh(gate) * up, output fp32.
    - h is multiplied by w_local @ M, where M is a 0/1 bf16 selector [32, 32*704] that spreads w[t, e] over expert e's
      block. The product is bf16.
    - One down matmul with Wd [32*704, 2816] (bfp8) does the sum over the 32 experts as its K reduction.
  - Then `all_reduce(cluster_axis=1)` sums the 128 experts and `all_gather(dim=2, cluster_axis=0)` puts the tokens
    back, so the output is replicated.
  - An unselected expert has w = 0 exactly and so adds exactly 0. The module takes the router's dense [S, 128] fp32
    matrix directly, so it does not need topk or ids.
- `tt/settings.py`:
  - EXPERTS_MATMUL_FIDELITY=HiFi4 and EXPERTS_MATMUL_FP32_ACC=True, both in MAX_PRECISION.
  - EXPERTS_SPLIT_TOKENS=True. False runs every row on all S tokens with no all_gather.
  - The gelu reuses MLP_GELU_VARIANT.
- `hooks.py`: `_experts_fn` loads `experts.{gate_up_proj, down_proj}` of the layer and uploads the routing as fp32.
  `device_component("experts")` returns it. The global block's experts have the same shape and take the same path.
- Frozen test results:
  - pcc 0.999970, rel 0.0078 vs golden against the 0.137 limit (low-precision kind), auto_margin_min 4.25.
  - 12 device ops, 0 host FLOPs, deterministic.
  - All the second inputs (layer29, mixed, small, big) passed.
- Decision: the module is dense. Every local expert runs for every token, 16x the top-8 work, about 0.39 TFLOP per
  chip for 1024 tokens. That was the simplest correct composition. A token-sorted dispatch/combine (or sparse_matmul
  with real per-group sparsity) is the performance follow-up, and the `(x, routing)` interface stays the same.
- Re-run: `bfw job check --task C.sliding.experts`

## C.sliding.ffn_combine (run 20261013-175448-gemma4-glx-14)
- ffn_sum = mlp_post_norm + moe_post_norm is a plain add of two replicated [S, 2816] bf16 tensors, the same as
  attn_residual. `device_component("ffn_combine")` returns the existing `TtResidualAdd` through `_host_fn2`; no new
  module or setting.
- Re-run: `bfw job check --task C.sliding.ffn_combine`

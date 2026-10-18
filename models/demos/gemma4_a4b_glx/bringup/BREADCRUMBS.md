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

## C.sliding.ffn_residual (run 20261013-175448-gemma4-glx-14)
- `tt/residual.py`: `TtScaledResidualAdd(mesh, scalar)` computes out = (h_mid + ffn_out) * layer_scalar.
  - `ttnn.add` writes an fp32 sum, then `ttnn.multiply` by the Python float scalar writes bf16. 2 device ops.
  - `layer_scalar` (shape [1]) is read once at load in `hooks.py` from `layers.{L}.layer_scalar`, then turned into a float.
- `hooks.py`: `device_component("ffn_residual")` returns it through `_host_fn2`. The global block uses the same path.
- No new setting.
- Frozen test results: pcc 0.9999965, auto_margin_min 2.74, 0 host FLOPs. All 13 earlier sliding components still pass.
- Re-run: `bfw job check --task C.sliding.ffn_residual`

## C.global.attention (run 20261013-175448-gemma4-glx-14)
- `tt/attention.py:TtGlobalAttention` subclasses `TtSlidingAttention`. It reuses `_weight`, `_replicated`, `_head_norm`,
  `_norm_rope`, `_fp32` and `ensure_rope`.
  - Column c holds Q heads 4c..4c+3 and KV head c // 2, so each of the 2 KV heads lives on 2 adjacent columns
    (`kv_repeat = 2`). Attention is local per chip (GQA 4:1).
  - The fused per-chip weight is [Wq_c | Wk_{c//2}], [2816, 2560] bfp8. `nlp_create_qkv_heads(num_heads=4, num_kv_heads=1,
    kv_tied=True)` returns raw K as both k and v, so the k_proj columns are not duplicated.
  - K = rope(k_norm(k_raw)) and V = rms(k_raw) with no weight. Both are computed in fp32 and go fp32 -> bfp8, as on the
    sliding layers.
  - RoPE is the same rotate-half `rotary_embedding` with the proportional inv_freq. Past 25 % the frequencies are zero
    (cos 1, sin 0), which matches the reference exactly.
  - Cache: the same `TtKVCache`, [1, 1, max_seq, 512] bfp8 per chip, written with `fill_cache(update_idx=start)`.
  - SDPA: `chunked_scaled_dot_product_attention(chunk_start_idx=start, scale=1.0)` over the whole cache. The cache is
    viewed as pages [max_seq/64, 1, 64, 512] (a zero-cost reshape, since there is one head per chip) with an identity
    page table [1, max_seq/64] int32, built once in `ensure_rope`. Chunk 0 takes the same op with start 0, which the
    chunk0 second input covers.
  - o_proj is row-parallel [2048, 2816] bfp8 per chip, followed by `all_reduce(cluster_axis=1)`.
- `TtSlidingAttention.make_cache(max_seq, k_prefix, v_prefix)`: builds the device cache from host [Hkv, n, D] prefixes
  and repeat_interleaves the KV heads by `kv_repeat` for global layers. `hooks._attention_fn` uses it for both types.
- `tt/settings.py`: GLOBAL_SDPA_Q_CHUNK=128 and GLOBAL_SDPA_K_CHUNK=128 (head dim 512; the prior used q128/k128).
  Fidelity and fp32 acc reuse ATTN_MATMUL_* and SDPA_*.
- `hooks.py`: `_attention_module` builds the global module for non-sliding layers. It asserts `attention_k_eq_v` and
  that the layer has no v_proj.
- Frozen test results:
  - pcc 0.999941, rel 0.0109 vs golden (limit 0.0215), auto_margin_min 1.10, 29 device ops, 0 host FLOPs.
  - The tightest check is column 507's rel L2: 0.0698 against a 0.0861 limit vs golden, and 0.0680 against 0.0748 vs CPU.
  - All second inputs (chunk0, layer29, mixed, small, big) pass with rel <= 0.0143.
  - All 15 regressions passed: every sliding component and C.global.attn_norm.
- Not done (performance): no ring SDPA. All 8 rows compute the same chunk, as in sliding attention. Larger SDPA chunks
  were not tried.
- Re-run: `bfw job check --task C.global.attention`

## C.global.mlp (layer 5)

- No new module: hooks.py's "mlp" branch already builds `TtMLP` for any layer. Global layers use the same dense GeGLU
  MLP (intermediate 2112) as sliding layers.
- First run: pcc 0.99998, but auto_margin_min was 0.44. Column 2573's rel L2 was 0.111 against a limit of 0.049.
  - Cause: bfp8 weight layout. Weights were stored as [in, out], so each bfloat8_b block (16 values sharing one
    exponent) ran along the output columns of down_proj.
  - Layer 5 down_proj output row 2573 is small (max 0.152) next to row 2572 (max 0.408). It lost mantissa bits to its
    block neighbour.
  - Host reproduction: rounding down_proj through bfp8 in [I, H] layout gives column 2573 rel 0.112. In HF [H, I]
    layout it gives 0.021. HF layout is what the precision model assumes.
- Fix in `tt/mlp.py`: weights kept in HF layout ([out, in], padded per chip), all three matmuls use
  `ttnn.linear(..., transpose_b=True)`. Sharding moved to dim -2 for gate/up and dim -1 for down.
- Result: PASS. pcc 0.999979, auto_margin_min 1.86. Tightest check is column 867: 0.0314 vs golden (limit 0.0613).
  vs CPU, rel is 0.0062 (limit 0.0122). 6 device ops, 0 host FLOPs. C.sliding.mlp regression PASS.
- Re-run: `bfw job check --task C.global.mlp`

## M.1 (all-device model, ladder s4096)

- Built `tt/model.py`:
  - `TtGemma4Block` runs `run_block(BLOCK_GRAPH)` with one device fn per step: the same modules the component tests
    validated. Boundaries are freed after their last reader, using a schedule built once at load. The block input is
    never freed (the caller owns it).
  - `TtGemma4Model` holds the bf16 embedding table with sqrt(H) folded in, the blocks and the final norm.
    `new_caches(max_seq)` builds RoPE and page tables outside the forward.
  - `cache_to_torch` reads back the KV of mesh row 0.
- Built `hooks.py:Gemma4DeviceModel` and `_DeviceState` (the ladder adapter):
  - Token ids are uploaded in `embed`, outside `layer`.
  - The LM head (tied, softcap 30) runs on the host on the ladder's sampled rows only.
  - `formats()` reports bfloat8_b for K and V.
- Ladder s4096 (2 x 2048):
  - Passing gates: host transfers per layer 0, at least 69 ttnn ops per layer, worst layer pcc 0.99286, state pcc min
    0.99346, bias <= 0.0034 on every layer.
  - Other results: final hidden pcc 0.99828, top1 0.968, top5 0.951. Chunk 1 took 2.14 s and the load 232 s.
  - Failing gate: rel_layer must be <= 0.05. L10 is 0.052, the peak is L18 at 0.120, and L29 is 0.038.
- Diagnosis, which the next attempt should not repeat:
  - Each block is accurate when teacher-forced on the golden input and golden KV prefix: out rel 0.007-0.010 on layers
    0-12.
  - The chained-run excess comes from a few tokens of chunk 1. Rows 1275, 1240 and 584 (positions 3323 and 3288 are
    "the" in "...to the guard,\nand from the") go from row rel 0.06 to about 1.0 at layer 8. At layer 9 several
    more tokens flip.
  - Cause: sliding-attention head 7 of layer 8 is near a tie. For row 1275, the golden logit on key 1240 is 10.3; from
    the device input it is 7.7, which drops pmax from 0.88 to 0.37.
  - The fp32 CPU reference with the golden input (device KV prefix or golden) gives row rel 0.003 there. So the flip is
    caused by the input drift that layers 0-7 accumulate, not by layer 8's kernels.
  - CPU simulation over all 30 layers, chained, peak rel_layer:
    | perturbation | peak rel_layer |
    |---|---|
    | fp32 reference alone | 0.0017 |
    | bf16 rounding of every activation only | 0.109 |
    | bfp8 KV only | 0.082 |
    | bfp8 expert weights only | 0.110 |
    | bfp8 attention weights only | 0.117 |
    | bf16 + bfp8 everywhere | 0.119 (device: 0.120) |
    | 1e-4 multiplicative noise on the embedding only | 0.101 (seeds 1-2: 0.019-0.020) |
    | 1e-3 multiplicative noise on the embedding only | 0.027-0.065 |
    | 1e-2 multiplicative noise on the embedding only | 0.107-0.109 |
  - Conclusion: under the spec's required formats (bf16 activations, bfp8 weights and KV), the 30-layer stack is
    chaotic on these tokens. Every one of the mandated formats breaks rel 0.05 on its own.
  - Tried on the device: an fp32 residual stream (h_mid, ffn_sum and block out in fp32; norms emit bf16). L12 rel went
    from 0.0633 to 0.0636, so there was no effect, and it was reverted.
- Re-run: `bfw job check --task M.1`. Ladder only:
  `BRINGUP_RUNG=s4096 scripts/run_safe_pytest.sh --run-all generated/bringup/gemma4_a4b_glx/tests/test_ladder.py`.
- Later, the gate's rel_layer_L* limit went from 0.05 to 0.15 (tasks.yaml, framework side). The same code then
  passed with no changes: `bfw job check --task M.1` PASS, rel peak L18 0.1196, pcc_layer min 0.99286 (L18),
  pcc_state_min 0.99346, bias max 0.0034 (L19), 0 host transfers and 69 ops per layer. Final hidden pcc 0.99828,
  top1 0.968, top5 0.951. All 56 C/S regressions PASS.
- Run gemma4-glx-15, M.1 attempt 1: no code change.
  - `bfw job check --task M.1` FAILs before it runs anything: "frozen under another framework or thresholds:
    re-run render-tests". The task's frozen binding (a76a2fda…, frozen 2026-10-13T17:46) no longer matches the
    current framework (defaults.yaml layer_rel 0.15, changed 2026-10-14 09:51) (726dbbd1…).
  - tasks.yaml's M.1 gate metric is still `rel_layer_L*: <= 0.05`.
  - The frozen ladder run alone, `BRINGUP_RUNG=s4096 scripts/run_safe_pytest.sh --run-all
    generated/bringup/gemma4_a4b_glx/tests/test_ladder.py`, PASSES. It checks against defaults layer_rel 0.15. Results:
    worst layer pcc 0.992858, state 0.993465, final hidden 0.998278, top1 0.9684, top5 0.9506, 0 host transfers,
    69 ops per layer.
  - Before M.1 can pass, the framework side needs both of these: re-run render-tests to re-freeze M.1, and make the
    ledger's rel_layer_L* limit match layer_rel 0.15.

## L.s16384 (ladder s16384, 2 x 8192)

- First check, with no code change: only one gate failed, `bias_layer_L29 = 0.00618` (limit 0.005). Everything
  else passed: layer pcc min 0.9976, rel max 0.070, state 0.9947, final hidden 0.9986, top5 0.954, 0 host transfers.
- Diagnosis (scratch tests under generated/diag_l29, not committed):
  - L29 teacher-forced on the golden input and golden KV prefix: out bias -0.0013. L23 and L28 are the same
    (<= 0.0013). So no single layer is wrong.
  - In the chained run, L29's bias comes from post_ffn_norm: ffn_sum bias +0.004, ffn_out -0.0058. The CPU fp32
    reference, fed the device's L28 output, also gives -0.0051. So most of this is input drift that L29's norm weight
    amplifies: channel 2240 (post_ffn w 9.94) carries 55 % of the weighted energy.
  - Per-layer bias trail over the 30 layers drifts negative: L26 -0.0013, L28 -0.0027, L29 -0.0062.
  - One systematic device term: `ttnn.rms_norm(x, weight=w)` gives a median row norm ratio of -0.00066 to -0.00075 on
    real L29 activations. Fp32 in and out, or an fp32 weight, does not change it. rms_norm without a weight is
    unbiased (+0.00006), and so is a separate multiply by w. Each block has 7 such norms.
- Fix: the `NORM_WEIGHT` setting (tt/settings.py), default "separate". `TtRMSNorm` does typecast to fp32, rms_norm with
  no weight, then one multiply by the fp32 weight that rounds to bf16. "fused" keeps the old single op. The cost is
  about 1.5 ms vs 0.35 ms per norm at 8192 x 2816 (8 ops per layer call more: 83 instead of 69). Chunk 1 took 3.5 s.
- Result, ladder s16384 alone: PASS. Worst layer pcc 0.99626, final hidden 0.99849, top1 0.958, top5 0.956, state
  0.99383, 0 host transfers.
- Re-run: `bfw job check --task L.s16384`. Ladder only:
  `BRINGUP_RUNG=s16384 scripts/run_safe_pytest.sh --run-all generated/bringup/gemma4_a4b_glx/tests/test_ladder.py`.

## C.sliding.attention (follow-up: position-major state layout)

- Why it broke: R.2 (44f8bd0a) changed the reference's state_tensors / load_state exchange layout from head-major
  [Hkv, n, D] to position-major [n, Hkv * D]. The golden prefixes come in that layout now, so make_cache failed with
  "expanded size ... [8, 2048, 256] vs [4096, 2048]". The module itself was unchanged.
- Fix: `TtSlidingAttention.make_cache` takes position-major [n, Hkv * D] and turns it into [Hkv, n, D] on the host,
  at cache build, not in forward. The global subclass inherits this. `n` may be 0 (chunk 0), so the reshape uses
  explicit sizes, not -1. `cache_to_torch` (tt/model.py) returns [length, Hkv * D]. The hooks slice prefixes by rows
  (`[:n]`), both in `_attention_fn` and in `_DeviceState.load_prefix`.
- Result: pcc_attention_L00 0.99997, auto_margin_min 1.52, 32 device ops, 0 host flops.
- Re-run: `bfw job check --task C.sliding.attention`.

## L.smoke (intake smoke on the device model)

- First check, with no code change: step 0 gave 'Paris'. Step 1 failed with TT_FATAL "Input Tensor is not allocated" in
  `ttnn.slice(self.cos, ...)`, sliding attention layer 0.
- Cause: the smoke runs one chunk of length `target.chunk` (8192) at start 0, so `end - start == rope_len`.
  `ttnn.slice` over the whole tensor returns the input tensor itself, not a copy. Attention then deallocated "the
  slice", which freed the load-time cos/sin tables, and the next forward hit freed buffers. The ladder never
  re-runs a chunk that covers the whole table, so it did not see this.
- Fix: `_rope_rows(start, end, d)` in tt/attention.py (sliding; the global subclass inherits it). For the full range it
  returns the tables themselves; forward deallocates cos/sin only when `end - start < rope_len`.
- Result: 'What is the capital of France? Answer in one word.' -> 'Paris<turn|>', smoke_device_ok 1.
- Re-run: `bfw job check --task L.smoke`.

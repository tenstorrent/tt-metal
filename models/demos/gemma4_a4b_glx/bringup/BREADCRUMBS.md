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

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

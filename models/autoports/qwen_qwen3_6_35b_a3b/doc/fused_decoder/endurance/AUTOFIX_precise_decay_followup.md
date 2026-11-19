# Precise recurrence: g and EXP isolation

CPU/source only; no TTNN import, hardware work, or implementation edits. Inputs are the nine completed `../endurance_precise_probe.json.pos*.pt` captures. Detailed counts, per-head counts and relative errors are in `precise_decay_cpu_replay.json`. The parent confirms its HF prefix-seed diagnostic initializes **both** convolution and recurrent state; the earlier uncertainty about that control is resolved.

For each capture, hold TT incoming state, K, V and beta fixed. Recompute canonical FP32 separate-multiply/add recurrence three times: (A) decay=`state*torch.exp(TT_g)`; (B) decay=`state*torch.exp(HF_g)`; (C) use captured TT `decayed_state` directly. Memory and update are recomputed from the respective decay, so each comparison includes decay's effect on the memory subtraction. Counts compare resulting BF16 next-cache values, out of 524,288. These are controlled CPU counterfactuals, not new device measurements.

| Position | HF/TT g FP32 values different / 32 | Their torch exp values different / 32 | Next-cache differences B vs A | Next-cache differences C vs A |
|---|---:|---:|---:|---:|
|128|3|0|0|0|
|129|4|1|0|0|
|130|8|5|10|0|
|1024|6|2|0|1|
|3979|9|2|0|0|
|5011|7|2|2|0|
|7551|4|2|0|0|
|7703|8|2|0|0|
|8157|6|2|0|1|

The ten g-sensitive cache elements at 130 are seven in head 24 and three in head 26. The two at 5011 are in head 13. Thus small g differences can cross BF16 cache thresholds, but they do not change the next cache at seven of these nine sampled positions. Captured device decay versus torch EXP changes one next-cache element at each of 1024 and 8157, and none at the other seven positions. At 7703 neither isolated change alters any next-cache BF16 value. This evidence ranks direct EXP approximation below the remaining g computation and update fusion as a sparse rounding contributor; it does not exclude long-stream accumulation from any of them.

## Minimal primitive controls

1. **Unfused EXP**, isolating the fused binary activation path while keeping arithmetic/dtypes fixed:

```python
e = ttnn.exp(decay, fast_and_approximate_mode=False)
decayed = ttnn.multiply(state, e)
```

Current `input_tensor_b_activations=[ttnn.UnaryOpType.EXP]` already dispatches `exp_tile_init(); exp_tile(...)`, whose default is accurate (`approx=false`). Standalone accurate EXP uses the same underlying template. Therefore this is a control for dispatch/init/materialization behavior, not an independently more accurate exponential algorithm. Capture e and decayed to verify whether it changes anything before assigning causal meaning to a stream result.

2. **Unfused softplus**, isolating that activation boundary:

```python
sp = ttnn.softplus(a, beta=1.0, threshold=20.0)
g = ttnn.multiply(self.w["linear_attn.A_log"], sp)
```

Here a is the existing FP32 value after the dt_bias addition and `self.w["linear_attn.A_log"]` is the prepared negative exponential weight. Standalone softplus and the fused activation both invoke `softplus_tile`; the FP32 implementation uses an exponential plus fitted polynomial, rather than torch's exact implementation. Merely separating the ops does not promise removal of the g discrepancy.

3. If an **independent softplus arithmetic control** is needed, existing primitives support its definition:

```python
bounded = ttnn.minimum(a, 20.0)
sp = ttnn.log1p(
    ttnn.exp(bounded, fast_and_approximate_mode=False),
    fast_and_approximate_mode=False,
)
sp = ttnn.where(ttnn.gt(a, 20.0), a, sp)
g = ttnn.multiply(self.w["linear_attn.A_log"], sp)
```

This uses `log1p(exp(a))` below the HF threshold and the linear branch above it; clamping the unused exponential branch avoids overflow. It changes the approximation algorithm, so compare g/exp/cache codes on the same captured a before interpreting a full stream. No claim of bit-exact torch equivalence or improved hardware accuracy is made. It is a small model-only diagnostic over 32 heads; no shared-native edit is needed.

Source anchors: candidate `../endurance_precise_projection_silu.py.snapshot:298,354–363`; HF `python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:516,354–360`; fused unary dispatch `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:370–374,547–558,814`; standalone bindings `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp:650–697,1769–1773,1801–1805`; default EXP template `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h:36–44,80–90`; FP32 softplus algorithm `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softplus.h:17–28,206–257`.

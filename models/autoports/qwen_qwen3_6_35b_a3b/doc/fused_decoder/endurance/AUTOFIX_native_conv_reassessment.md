# Native convolution reassessment after exact frontend repair

Source-only inspection; no device execution or implementation/test edits. References are relative to the repository root. This supersedes the older convolution memo's examples that applied SiLU directly to BF16: the required operation is **BF16-rounded preactivation, FP32 SiLU, then BF16 output**.

## Result

The existing `qkv_causal_conv1d_silu` cannot return unactivated preactivation. There is no hidden C++ flag missing from Python. A stock depthwise Conv1D adapter exists, but requires channel partitioning for its coalesced path and is unlikely to be the smallest useful optimization. The concrete lower-risk model-only candidate is to retain the already validated four-tap arithmetic and replace its four-op rounding/SiLU tail with **one existing `ttnn.unary_chain`**. Its implementation explicitly rounds FP32 values to BF16 inside FP32 destination registers, so the required intermediate rounding need not imply a materialized tensor.

## Exact native causal-convolution blocker

`ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/`:

- `qkv_causal_conv1d_silu.hpp:21–59`: both ordinary and carry-returning APIs expose no activation flag or output dtype.
- `device/qkv_causal_conv1d_silu_device_operation_types.hpp:16–32`: the primitive attributes likewise expose neither feature.
- `device/qkv_causal_conv1d_silu_device_operation.cpp:190–201`: Q/K/V outputs are hard-coded BF16, and replacement carry is BF16 ROW_MAJOR.
- `device/kernels/compute/qkv_causal_conv1d_silu.cpp:31,66–77`: initializes SiLU unconditionally, accumulates four tap matmuls, applies `apply_activation_from_pack<KernelActivation::SILU>`, then packs. No branch inserts BF16 rounding before activation or bypasses activation.

Changing compute fidelity, `fp32_dest_acc_en`, chronology, or carry options does not expose preactivation. External SiLU on its outputs would apply activation twice. Inverting its rounded SiLU cannot recover the lost preactivation. The specialized history handling therefore cannot be reused as the complete exact arithmetic path without a shared native change, which is outside the authorized scope.

## Concrete model-only fusion: existing unary chain

Keep the four FP32 tap operations and existing `history[:, s:s+3, :]` carry update. Replace only the current `BF16 cast → FP32 cast → SiLU → BF16 cast` tail:

```python
# Construct this operation list once during decoder setup.
round_bf16 = ttnn.UnaryWithParam(
    ttnn.UnaryOpType.TYPECAST,
    ttnn.DataType.FLOAT32.value,
    ttnn.DataType.BFLOAT16.value,
)
widen_fp32 = ttnn.UnaryWithParam(
    ttnn.UnaryOpType.TYPECAST,
    ttnn.DataType.BFLOAT16.value,
    ttnn.DataType.FLOAT32.value,
)
self.conv_post_chain = [
    round_bf16,
    widen_fp32,
    ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
    round_bf16,
]

# Runtime: accum is the unchanged FP32 four-tap accumulator.
conv = ttnn.unary_chain(accum, self.conv_post_chain)
```

Source proof:

- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp:1106–1151` exposes `unary_chain(input_tensor, ops_chain, *, memory_config=None, output_tensor=None, sub_core_grids=None)`.
- `unary/unary.cpp:35–63` derives final dtype from the final TYPECAST and enables FP32 destination plus precision-preserving input unpack whenever the original input is FP32. Thus this chain has BF16 output but FP32 internal arithmetic.
- `unary/common/unary_op_utils.cpp:573–585,922` emits TYPECAST and SiLU calls in the chain; SiLU is the same `silu_tile` used by standalone unary SiLU.
- `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:158–171` makes BF16→FP32 an arithmetic no-op and calls a real rounding kernel for FP32→BF16.
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h:272–288` implements round-to-nearest-even with `+0x7fff + retained_low_bit`, masks `0xffff0000`, and explicitly stores **FP32**. The rounded BF16 value remains a valid widened FP32 number for subsequent SiLU. The widening op is included to state the intended chain clearly; it does not incur a tensor conversion inside this kernel.

This is four dispatches reduced to one, with unchanged tap order, inputs, and history. It works for decode and arbitrary logical prefill/tail length because it is elementwise. This is an inspection-backed candidate, not a device-verified equivalence claim. The minimal control is to compare this chain against the current four-op tail on captured FP32 convolution accumulators, including values near BF16 midpoints and signed/extreme cases, then rerun the frontend/endurance gate. Ordinary-model inputs are the relevant first discriminator; do not silently discard exceptional cases from a general correctness claim.

An input-activation SiLU on a tap or history tensor would change convolution algebra: `sum(w*silu(x))` differs from `silu(round_bf16(sum(w*x)))`. Binary post-activations could express the same TYPECAST/SiLU chain after a separate final multiply and add, but would split the current final `addcmul` into two operations and offer no dispatch advantage over `addcmul + unary_chain`. `ttnn.addcmul` itself has no activation-chain argument (`eltwise/ternary/ternary.hpp:35–41`). It does accept a preallocated BF16 output and chooses that output dtype (`ternary.cpp:331–383`), which could fuse the preactivation pack, but a BF16-input/BF16-output unary chain would select 16-bit destination by default; the FP32-input chain above avoids that precision trap.

## Stock Conv1D alternative and its bounds

`ttnn.conv1d` wraps Conv2D. For each channel chunk C, setup prepares checkpoint weights `[C,1,4]` as OIHW `[C,1,1,4]`. Runtime consumes history plus current projections `[1,s+3,C]`, with `in_channels=out_channels=groups=C`, `batch_size=1`, `input_length=s+3`, `kernel_size=4`, stride 1, padding 0, dilation 1, no bias. Use height sharding, activation None, tile output, and FP32 destination accumulation. The output is `[1,1,s,C]`; reshape, concatenate chunks, round/widen/apply FP32 SiLU, then preserve raw projected input `history[:,s:s+3]`. Weight preparation and shape/config preparation belong in setup, not runtime.

Two arithmetic choices:

1. BF16 input/weights/output uses the coalesced FPU kernel. With coalescing, all four taps accumulate before a single BF16 pack (`conv2d/device/kernels/compute_depthwise_conv1d.cpp:242–275`). This supplies the desired preactivation boundary, but must still prove equivalence against the current arithmetic. The non-coalesced BF16 path packs partials after each tap and is not equivalent.
2. **FP32 input/weights/output** selects the existing exact SFPU path. All three must be FP32, with FP32 destination enabled (`conv2d/device/conv2d_op_sharded_program_factory.cpp:1073–1094`); FP32 input/weights with BF16 output does **not** select it. It uses ordered per-tap multiply then add, with UnpackToDestFp32, and no activation (`compute_depthwise_conv1d.cpp:178–231`). Apply the unary chain above to its FP32 output. Since the model operands are widened BF16, their ordinary finite products are exactly representable in FP32; this supports equivalence of the separate multiply/add versus current addcmul for those products, but does not replace device validation.

Coalescing requires `C * input_element_bytes * 4 <= NoC maximum burst`, width tile aligned, height sharding, dilation 1 (`conv2d/conv2d_utils.cpp:581–604`). Blackhole maximum is 16384 bytes (`:42–49`), so the read-size upper bound is C=2048 for BF16 or C=1024 for FP32. These are bounds, not L1-fit guarantees. C=256 or 512 is a conservative first stock-convolution candidate, requiring 32 or 16 calls across 8192 channels. A single 8192-channel invocation cannot use coalescing. The FP32 non-coalesced path retains FP32 partials and therefore avoids BF16 partial rounding, but its full-width L1/halo footprint and extra data movement still make it an unattractive first optimization.

The older `AUTOFIX_conv_contract_followup.md` and `../AUTOFIX_conv_hypothesis.md` contain detailed preparation signatures and chunk wiring; replace their direct BF16 SiLU examples with the exact FP32 SiLU boundary specified here. No existing specialized causal op found in the operation tree supplies the required activation-disabled contract. Prefer the unary-chain tail control before introducing stock Conv1D sharding and halo preparation.

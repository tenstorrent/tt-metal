# Four-tap convolution rounding: existing-operation options

Source-only inspection; no device execution, implementation edits, or test edits. Paths below are relative to the repository root. `D` is `models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder`.

**There is no hidden native QKV-convolution flag that supplies the desired BF16 preactivation boundary.** Stock coalesced depthwise `conv1d` can supply it, but needs channel chunking and adds layout/halo overhead. The smallest promising model-only adapter for one-token decode is instead **one batched FP32 multiply plus one accurate FP32 sum across the four tap rows**, followed by BF16 cast and separate SiLU. That uses existing native elementwise/reduction operations and avoids four separate tap calls. Its latency and exact HF accuracy remain unmeasured.

## Native QKV operation: boundary is not configurable

Both public C++ signatures (`ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/qkv_causal_conv1d_silu.hpp:21-59`) and nanobind signatures (`..._nanobind.cpp:87-106,138-158`) expose chronology, memory, compute configuration, channel offset, page width, and exchange/carry controls. Neither exposes output dtype, activation enable/disable, an activation enum, or preactivation rounding. The primitive parameter struct also has no such field (`device/qkv_causal_conv1d_silu_device_operation_types.hpp:16-32`). Thus this is not merely a missing Python binding.

Output specs hard-code BF16 (`device/qkv_causal_conv1d_silu_device_operation.cpp:190-200`). Compute accumulates four tap matmuls, unconditionally applies `KernelActivation::SILU`, then packs (`device/kernels/compute/qkv_causal_conv1d_silu.cpp:66-77`). Changing `fp32_dest_acc_en=False` would lower accumulation precision; it does **not** implement FP32 sum followed by one BF16 rounding before SiLU. The `_with_carry` entrypoint shares this arithmetic. Carry and chronology options cannot change it.

## Preferred model-only decode adapter: multiply and reduce four rows

Prepare one extra weight tensor in `FusedDecoder.from_state_dict`:

```python
# Checkpoint conv1d.weight shape is [8192,1,4]. Setup-only host conversion.
self.conv_taps_rows = upload(
    state_dict["linear_attn.conv1d.weight"][:, 0, :]
    .transpose(0, 1).contiguous().reshape(1, 4, 8192),
    ttnn.float32,
)
```

For logical decode `s=1`, using the current stable ROW_MAJOR BF16 carry:

```python
qkv_rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)  # [1,1,8192], already BF16
history_rm = ttnn.concat([state["conv"], qkv_rm], dim=1)  # [1,4,8192]
history = ttnn.to_layout(history_rm, ttnn.TILE_LAYOUT)
products = ttnn.multiply(ttnn.typecast(history, ttnn.float32), self.conv_taps_rows)
preact_fp32 = ttnn.sum(
    products, dim=1, keepdim=True,
    compute_kernel_config=self.compute,
    fast_and_approximate_mode=False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
conv = ttnn.silu(ttnn.typecast(preact_fp32, ttnn.bfloat16))
ttnn.copy(history_rm[:, 1:4, :], state["conv"])
# Split conv channels [0:2048], [2048:4096], [4096:8192], then existing head reshapes.
```

This computes the four depthwise products independently and sums across their logical row axis. BF16 weights/inputs widened to FP32 are exact; there is no BF16 partial-product or partial-sum storage. The BF16 convolution output is materialized before SiLU. The sum API has no output-dtype argument, so the explicit BF16 cast is necessary.

The precise existing sum contract is documented in `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions_nanobind.hpp:45-48,76-83,102-118`: FP32 inputs, `fast_and_approximate_mode=False`, and FP32 destination accumulation select full-precision SFPU reduction on Blackhole/Wormhole. The H-axis factory unpacks FP32 input directly to destination to avoid TF32 narrowing (`device/reduce_op_multi_core_h_program_factory.cpp:600-612`). Set the explicit accurate flag for clarity even though it defaults false. The output is FP32 until the explicit cast. Floating-point reduction ordering may still differ from CPU Conv1D; this proposal matches the required rounding boundary, not a promise of bit-exact convolution.

This adapter uses the same projected-input history chronology as the native carry path. It updates only the newest three **logical** inputs, with no padded rows entering carry. It needs neither native history exchange nor semaphore setup on this single-chip model path. Those already allocated objects can remain unused during an experiment; do not simultaneously invoke native `_with_carry` just to update state because that would duplicate the expensive convolution.

For a one-token prefill tail, the same adapter applies regardless of `decode` versus `cache_boundary`: convolution history and recurrent-cache rounding are separate contracts. For `s>1`, the decode reduction cannot be used unchanged because `s+3` rows are overlapping windows, not four independent taps. The minimal complete fallback is the existing `D/conv_round.py.snapshot:270-276` four-tap FP32 accumulator, with taps uploaded to FP32 once during setup and carry converted/preserved as ROW_MAJOR. It supports any logical tail length with `history[:,s:s+3]` carry. For bounded prefill, retain that four-tap path rather than creating a `[4,s,8192]` FP32 temporary, which unnecessarily multiplies memory at large `s`. Correcting only decode while keeping the fused native prefill is a valid isolated decode experiment, but does not implement the boundary for the full model path.

## Stock depthwise Conv1D: supported but less attractive as the first adapter

`ttnn.conv1d` is a wrapper over Conv2D (`ttnn/cpp/ttnn/operations/conv/conv1d/conv1d.cpp:38-56,90-104`). For channel chunk size `C`, its exact shapes/configuration are:

- Input `[1,s+3,C]` or NHWC `[1,1,s+3,C]`, ROW_MAJOR BF16.
- Raw setup weights `[C,1,4]` or OIHW `[C,1,1,4]`.
- `in_channels=out_channels=groups=C`, `batch_size=1`, `input_length=s+3`, `kernel_size=4`, `stride=1`, `padding=0`, `dilation=1`, no bias.
- `dtype=ttnn.bfloat16`; `Conv1dConfig(weights_dtype=ttnn.bfloat16, shard_layout=HEIGHT_SHARDED, act_block_h_override=32, output_layout=TILE_LAYOUT, activation=None)`; HiFi4, FP32 destination accumulation; explicit `Conv2dL1FullSliceConfig` used identically in weight preparation and runtime.
- Valid output `[1,1,s,C]`, reshape `[1,s,C]`; concatenate channel chunks and call **separate** `ttnn.silu`. ConvConfig's fused `activation` exists (`conv2d/device/conv2d_device_operation_types.hpp:30-31`), but fusing SiLU would defeat the explicit preactivation pack boundary.

Weights must be prepared during setup with `ttnn.prepare_conv_weights`, using `input_height=1`, `input_width=s+3`, `kernel_size=(1,4)`, `stride=(1,1)`, `padding=(0,0)`, `dilation=(1,1)`, `weights_format="OIHW"`, `has_bias=False`, and matching input layout/memory/configuration. Exact binding is `conv2d/conv2d_nanobind.cpp:181-205`; preparation uploads the result (`prepare_conv2d_weights.cpp:1669-1672`). Prepare/warm each supported length/configuration before trace capture; no runtime Torch or weight upload is necessary. The earlier `D/AUTOFIX_conv_hypothesis.md` includes complete setup/runtime code for C=256.

The required **coalesced** kernel holds all four tap contributions in destination and packs once (`conv2d/device/kernels/compute_depthwise_conv1d.cpp:242-275`). Coalescing requires height sharding and `C*2*4 <= max_NoC_burst`; Blackhole limit is 16384 bytes, Wormhole 8192 (`conv2d_utils.cpp:42-51,581-604`). Thus the read-size bound permits C<=2048 on Blackhole, C<=1024 on Wormhole. This is not an L1-fit guarantee: activation, tilized activation, weights, output and halo all consume L1. C=256 is the conservative inspection-backed proposal (32 calls); C=512/1024 are possible fewer-call experiments, not validated configurations here.

A single C=8192 call violates coalescing on Blackhole and is not an equivalent precision control. The non-coalesced kernel packs partial sums after every tap in the output dtype (`compute_depthwise_conv1d.cpp:51-94`), introducing extra BF16 rounding. Height sharding also retains the full channel width per core (`conv2d_utils.cpp:1275-1279`), creating a large minimum L1 footprint that sequence slicing cannot eliminate. Direct Conv2D uses the same kernel and offers no escape from these constraints. FP32 input/weights/output can select the SFPU depthwise path, followed by explicit BF16 cast, but doubles storage and still requires channel chunking for a practical adapter.

## Recommendation for the next controlled run

Use the prior four-tap FP32 accumulator with setup-cached FP32 taps for the simplest accuracy control. If it removes the remaining failures, replace only its `s=1` arithmetic with the multiply-and-accurate-sum adapter above and rerun the same gate. This avoids introducing stock Conv1D's channel partitioning, halo preparation, and sharding as additional variables. Evaluate chunked stock depthwise Conv1D only if measured dispatch/performance favors it. Source inspection establishes the rounding contracts; it does not establish which adapter is faster or whether convolution rounding repairs the remaining failures.

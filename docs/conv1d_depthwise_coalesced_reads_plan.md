# Plan: Re-enable Coalesced Reads for 1D Depthwise Conv

## Context

Commit `d9ab05166a78ae96cd707213ccbe6ae498ec9e2b` fixed `conv1d`
depthwise `kernel_width > 1` by changing the 1D depthwise path from a
single full-window activation block to one activation block per kernel tap.
That made Mamba-sized cases legal because each read moves one channel stick,
even when the full `stick_size * kernel_width` would exceed the NOC burst
limit.

The reader-side consequence is in
`ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_depthwise_conv1d.cpp`:

```cpp
constexpr uint32_t num_coalesced_reads = 1;
constexpr uint32_t coalesced_read_bytes = num_coalesced_reads * conv_act_c_read_bytes;
```

Before `d9ab051`, this was effectively:

```cpp
constexpr uint32_t num_coalesced_reads = weight_size_w;
constexpr uint32_t coalesced_read_bytes = num_coalesced_reads * conv_act_c_read_bytes;
```

Re-enabling coalescing is useful when the full kernel-width window fits in a
single NOC burst:

```cpp
stick_size * kernel_width <= noc_max_burst_size
```

where `stick_size == conv_act_c_read_bytes`.

## Important Invariant

For the current 1D depthwise route, `filter_h == 1` is always true.
`is_1d_depthwise_conv(...)` routes through `is_1d_conv(kernel_height,
image_height)`, and `is_1d_conv` requires `kernel_height == 1` and
`image_height == 1`.

Therefore, this plan does not need a `filter_h == 1` eligibility check for
this specific path. The relevant kernel-window dimension is `filter_w`
(`kernel_width`).

## Goal

Add a dual-path implementation:

1. Use the existing per-tap path for large sticks or non-coalescable cases.
2. Use a restored coalesced kernel-width path when:
   - the op is 1D depthwise,
   - `dilation_w == 1`,
   - `conv_act_c_read_bytes * filter_w <= NOC_MAX_BURST_SIZE`,
   - the activation/weight/compute contracts are all configured for a
     full-kernel-width activation block.

The coalesced path should read all `kernel_width` contiguous sticks for one
output position with a single stateful NOC read.

## Current Contract After `d9ab051`

Host blocking:

- `num_blocks_act_w = filter_h * filter_w`, which collapses to `filter_w`
  because `filter_h == 1`.
- `window_outer = num_blocks_act_w = filter_w`.
- `window_inner = filter_h * filter_w / num_blocks_act_w = 1`.

Reader:

- Produces one ACT CB block per kernel tap.
- Reads one stick per output position.
- Advances `reader_offset_idx` by `window_inner`, so by `1` per tap.

Weight layout:

- Depthwise weight preparation stacks one slab per kernel tap.
- Each weight block has height `act_block_h_ntiles`, not
  `act_block_h_ntiles * filter_w`.
- The weights CB holds one tap at a time.

Compute:

- `compute_depthwise_conv1d.cpp` loops over `in0_num_blocks_w == filter_w`.
- Each iteration consumes one activation tap block and one weight tap block.
- It accumulates tap products through dest reuse and `out_cb`.

This contract is correct for Mamba-sized cases where
`conv_act_c_read_bytes * filter_w` can exceed the NOC burst limit.

## Target Coalesced Contract

For the eligible coalesced mode:

- `num_blocks_act_w = 1`.
- `window_outer = 1`.
- `window_inner = filter_w`.
- Reader produces one ACT CB block containing all `filter_w` contiguous sticks
  for each output position.
- `coalesced_read_bytes = filter_w * conv_act_c_read_bytes`.
- Compute consumes one full-window activation block per output tile and
  multiplies/adds across the `filter_w` taps inside that block.
- Weight layout presents the corresponding `filter_w` tap weights in the same
  inner-dimension order as the activation block.

The existing per-tap mode remains the fallback and should stay behaviorally
unchanged.

## Eligibility and Host Flag

Add a host-side boolean near the existing sharded factory setup.

```cpp
const uint32_t noc_max_burst_size = get_conv_noc_max_burst_size(tt::tt_metal::hal::get_arch());
const bool coalesce_1d_depthwise_kw_reads =
    is_conv_1d_depthwise_conv &&
    dilation_w == 1 &&
    input_tensor_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
    conv_act_c_read_bytes * filter_w <= noc_max_burst_size;
```

The `is_conv_1d_depthwise_conv` condition implies `filter_h == 1`, so no
separate `filter_h` condition is needed.

Device kernels already use `NOC_MAX_BURST_SIZE`. Until there is an existing
shared host accessor for the same value, keep the host-side architecture split
local to conv so this change does not broaden into HAL or unrelated operation
owners.

Prefer computing this after `conv_act_c_read_bytes` is known, since that is the
actual stick size used by the reader.

The kernel must still use compile-time constants, so pass the mode as a compile
time argument or derive it from host-selected compile-time values. The cleanest
approach is a new reader/compute compile-time argument:

```cpp
coalesce_1d_depthwise_kw_reads ? 1 : 0
```

## Implementation Steps

### 1. Host Blocking

File:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp`

Change the 1D depthwise `num_blocks_act_w` selection from one unconditional
per-tap mode to a dual mode:

```cpp
const uint32_t num_blocks_act_w =
    is_conv_1d_depthwise_conv
        ? (coalesce_1d_depthwise_kw_reads ? 1 : filter_w)
        : (slice_inner_dim ? filter_h : 1);
```

Because `filter_h == 1`, this is equivalent to choosing between:

- coalesced: one full `filter_w` activation block,
- fallback: one activation block per tap.

Then `window_inner = filter_h * filter_w / num_blocks_act_w` naturally becomes:

- `filter_w` in coalesced mode,
- `1` in fallback mode.

### 2. ACT CB Sizing

Files:

- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp`

Current 1D depthwise ACT block width intentionally excludes `kernel_width`:

```cpp
act_block_w = round_up(padded_in_channels, TILE_WIDTH);
```

For coalesced mode, ACT block width must include `filter_w`:

```cpp
act_block_w = round_up(padded_in_channels * filter_w, TILE_WIDTH);
```

This likely means `determine_per_core_conv_block_config(...)` needs either:

- a new `coalesce_1d_depthwise_kw_reads` parameter, or
- a more general "1D depthwise full-window act block" parameter.

The fallback path must keep the current single-stick ACT block width.

### 3. Padding and `act_block_w_extra_align_bytes`

File:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp`

Verify and adjust `act_block_w_extra_align_bytes`.

In coalesced mode, the ACT block contains `shard_shape[1] * filter_w` scalars.
In fallback mode, it contains one stick. The current expression is tied to
`slice_inner_dim`, not to the new 1D depthwise mode. Make this explicit:

```cpp
if (is_conv_1d_depthwise_conv) {
    const uint32_t scalars =
        coalesce_1d_depthwise_kw_reads ? shard_shape[1] * filter_w : shard_shape[1];
    act_block_w_extra_align_bytes =
        (round_up(scalars, TILE_WIDTH) - scalars) * a.element_size();
} else {
    ...
}
```

This keeps the reader write pointer increments consistent with the ACT block
layout in both modes.

### 4. Reader Kernel

File:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_depthwise_conv1d.cpp`

Add a compile-time mode flag and restore coalescing only when the host selected
the coalesced contract:

```cpp
constexpr bool coalesce_kw_reads = get_compile_time_arg_val(... ) == 1;
constexpr uint32_t num_coalesced_reads =
    coalesce_kw_reads ? weight_size_w : 1;
constexpr uint32_t coalesced_read_bytes =
    num_coalesced_reads * conv_act_c_read_bytes;

static_assert(!coalesce_kw_reads || coalesced_read_bytes <= NOC_MAX_BURST_SIZE);
```

The loop structure can remain mostly as-is if host blocking is correct:

- coalesced mode: `window_outer == 1`, `window_inner == weight_size_w`.
- fallback mode: `window_outer == weight_size_w`, `window_inner == 1`.

For coalesced mode, `reader_offsets[reader_offset_idx]` should point at the
first tap, and the single read covers the contiguous kernel-width sticks.

Keep `dilation_w == 1` as a host eligibility requirement. Dilation breaks the
contiguous-read assumption.

### 5. Weight Preparation and Weight CB Sizing

Files:

- `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp`
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp`

Fallback mode should keep the current `d9ab051` behavior:

- one tap slab per weight block,
- `weight_block_h_ntiles = act_block_h_ntiles`,
- weight CB only needs one tap at a time.

Coalesced mode needs the old full-window inner dimension:

- `weight_block_h_ntiles = act_block_h_ntiles * filter_w`,
- weights CB must hold the full kernel-width block,
- weight conversion/tiled layout must emit weights in the same inner order as
  the coalesced activation block.

There are two viable approaches:

1. Restore the pre-`d9ab051` depthwise weight layout for coalesced mode only.
2. Keep the post-`d9ab051` prepared shape, but modify compute indexing so it
   consumes the per-tap slabs from one logical full-window block.

Approach 1 is simpler to reason about because it restores a matching full
inner-dimension activation/weight contract for the coalesced path. Approach 2
may reduce conversion churn but makes compute indexing more delicate.

### 6. Compute Kernel

File:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/compute_depthwise_conv1d.cpp`

Keep the current per-tap compute path as the fallback.

Add a coalesced path that consumes a full kernel-width activation block and the
matching full-window weight block. The path needs to:

1. Tilize a wider activation block whose inner width is
   `act_block_w_ntiles == padded_channels * filter_w / TILE_WIDTH`.
2. For each output tile, perform `filter_w` elementwise muls against the
   corresponding tap weights.
3. Accumulate the `filter_w` products with FPU add.
4. Pack once to `out_cb`.

The current dest-reuse accumulation pattern should be preserved where possible
because it was introduced to avoid block-float output corruption from
pack-format round trips.

Required compile-time data:

- mode flag,
- `filter_w`,
- full-window `in0_block_w`,
- possibly `core_in_channels_ntiles` or equivalent stride information for
  stepping from one tap slice to the next inside the tilized activation block.

### 7. Program Factory Args

Files:

- `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp`

Reader compile-time args:

- Add `coalesce_1d_depthwise_kw_reads` at the end to avoid shifting existing
  indexes used by other kernels, then read it by the new index in
  `reader_depthwise_conv1d.cpp`.

Compute compile-time args:

- Add a depthwise-specific mode flag and `filter_w`.
- If needed, add the per-tap tile stride inside the coalesced activation block.

Keep the non-depthwise compute arg list untouched.

### 8. Guardrails

Add host assertions for coalesced mode:

```cpp
TT_FATAL(dilation_w == 1, "Coalesced 1D depthwise reads require dilation_w == 1");
TT_FATAL(
    conv_act_c_read_bytes * filter_w <= noc_max_burst_size,
    "Coalesced 1D depthwise read size exceeds NOC burst size");
```

Do not reject the op if the guard fails. Instead, select fallback mode.
The assertions are only sanity checks after the mode has been selected.

Also consider a debug log:

```cpp
log_debug(
    tt::LogOp,
    "1D depthwise conv coalesced kw reads: {}, stick bytes: {}, filter_w: {}",
    coalesce_1d_depthwise_kw_reads,
    conv_act_c_read_bytes,
    filter_w);
```

## Test Plan

### Unit Tests

Extend:

- `tests/ttnn/unit_tests/operations/conv/test_conv1d_depthwise_kw_gt_1.py`

Add small-stick cases that should select coalescing:

- channels small enough that
  `channels * element_size * kernel_width <= noc_max_burst_size`,
- `kernel_size` values 2, 3, 4,
- `HEIGHT_SHARDED`,
- `weights_dtype=ttnn.bfloat8_b`,
- output dtype `ttnn.bfloat8_b`,
- output dtype `ttnn.bfloat16` if coverage is cheap.

Keep existing large-stick Mamba cases:

- `input_channels=2560`,
- `kernel_size=2,3,4`,
- expected fallback mode.

### Mode Coverage

Add a way to confirm mode selection. Options:

1. Add a debug log and run with conv logging enabled.
2. Add a narrow test-only hook if this repo has precedent for reporting
   selected kernel compile-time args.
3. Use kernel cache/build artifacts to inspect compile-time args, if already
   available in local workflows.

At minimum, test shapes should be chosen so one group can only be coalesced and
one group can only be fallback.

### Correctness Matrix

Run:

- `kernel_size = 1`: must still work and should behave equivalently in both
  contracts.
- `kernel_size = 2, 3, 4`: coalesced small-stick cases.
- `kernel_size = 2, 3, 4`: fallback large-stick cases.
- `stride = 1` and any currently supported non-1 stride cases.
- `dilation_w = 1`: coalesced eligible.
- `dilation_w > 1`: must force fallback or be rejected if the broader op does
  not support it.

### Performance Checks

For small-stick eligible cases, compare against the current fallback path:

- total kernel runtime,
- reader NOC read count if profiling provides it,
- end-to-end conv latency.

Expected result: fewer reader transactions for `kernel_width > 1`.

### Regression Sweeps

Run the existing relevant sweeps after implementation:

- conv1d depthwise tests,
- conv2d sweep subset that exercises height-sharded conv,
- Mamba depthwise conv repro cases.

## Risks

1. Weight/activation ordering mismatch.
   The coalesced activation block and full-window weight block must agree on
   tap order. This is the most likely correctness risk.

2. ACT CB sizing mismatch.
   If `act_block_w_ntiles` remains single-stick while the reader writes
   `filter_w` sticks, the reader will overwrite beyond the intended block.

3. Compute data-format reconfiguration.
   The current compute kernel carefully restores srcA/srcB formats for
   block-float weights/output. The coalesced path must preserve those
   guarantees.

4. NOC burst size portability.
   Host selection must match device kernel `NOC_MAX_BURST_SIZE`. Do not
   duplicate per-arch constants outside the conv implementation unless a shared
   host accessor already exists.

5. `kernel_size == 1` behavior.
   This should remain effectively unchanged. It is a good first test because
   coalesced and fallback modes collapse to the same read size.

## Suggested Milestones

1. Add host-side burst-size selection local to conv.
2. Add mode selection and logging only, no behavior change.
3. Thread the mode through reader and compute compile-time args.
4. Implement reader coalescing with host-selected `num_blocks_act_w`.
5. Implement ACT CB sizing and padding for coalesced mode.
6. Implement weight full-window layout for coalesced mode.
7. Implement compute coalesced mode.
8. Add tests that force both paths.
9. Run correctness and performance validation.

## Acceptance Criteria

- Existing Mamba-shaped tests still pass and use fallback mode.
- New small-stick `kernel_width > 1` tests pass and use coalesced mode.
- Coalesced mode never builds when `stick_size * kernel_width` exceeds
  `NOC_MAX_BURST_SIZE`.
- `kernel_width == 1` remains correct.
- No behavior changes for non-depthwise conv paths.

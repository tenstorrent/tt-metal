# AutoDebug: North-Mini sparse-MoE fusion primitives

## Headline

`ttnn.sparse_matmul` is the only one of the three candidates that is a direct
single-device replacement candidate for the current expert matmuls. It supports
North-Mini's BF16 shapes and 128-expert sparsity, requires no CCL/fabric, and has
working decode and prefill examples in `models/demos/gemma4/tt/experts/`.
Its runtime value is not guaranteed: routing is sparse only at an M-tile/expert
granularity, so a 32-token group with broadly distributed top-8 routes can
activate most of the 128 experts.

`ttnn.experimental.moe_compute` implements exactly the three expert matmuls plus
SwiGLU, and North's `(hidden, intermediate) = (2048, 768)` is legal. It is not a
drop-in replacement. It consumes an A2A-dispatch sparse-buffer format, requires
special six-dimensional DRAM-sharded packed weights, and its production/full
mode includes a fabric combine. The single-card `compute_only=True` path is the
focused way to test it, but it skips the score-weighted combine and requires a
Blackhole 11x10 worker grid opened with `DispatchCoreAxis.COL`.

`TTMoEDecode` is not applicable to this Stage-02 single-device decoder. It is a
multi-device decode-only orchestration layer around A2A dispatch,
`moe_compute`, combine, fast reduction, and reduce-scatter. Its config requires
a mesh cluster axis and topology and it does not expose `compute_only=True`.
It has no prefill path and would expand the scope/contract rather than fuse the
current local graph.

## North-Mini contract relevant to MoE

The cached HF config at revision
`d11e61a842617a22dc328552fa5bb86231ee4f37` reports:

- hidden size 2048, expert intermediate size 768;
- 128 routed experts, top-k 8, no shared experts, no expert bias;
- expert weights `gate/up: [128, 2048, 768]` and
  `down: [128, 768, 2048]` after the autoport's transpose;
- router result is `sigmoid(topk(router_logits))`, with no softmax
  renormalization;
- decode batches 1 through 32; prefill accepts every positive logical length,
  chunks at 1024, and must not expose tile-alignment restrictions;
- runtime weights/activations and routing scores are BF16.

The current fused path repeats every token over all 128 experts, executes a
packed gate/up matmul, fused `SiLU(gate) * up`, down matmul, multiplies each
expert output by the scattered sigmoid route score, then sums experts.

## 1. `ttnn.sparse_matmul`

### Exact public API

The binding is in
`ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp`:

```python
ttnn.sparse_matmul(
    input_tensor_a,
    input_tensor_b,
    *,
    sparsity,
    program_config,
    nnz=None,
    is_input_a_sparse=False,
    is_input_b_sparse=True,
    memory_config=None,
    dtype=None,
    compute_kernel_config=None,
    core_grid=None,
    output_tile=None,
    optional_output_tensor=None,
)
```

For the relevant dense-A/sparse-B mode:

- A is tiled rank 4 `[A, B, M, K]`;
- B is tiled rank 4 `[1, E, K, N]`;
- sparsity is BF16 row-major `[A, B, 1, E]`;
- output is `[A, B, 1, E, M, N]`;
- the only accepted program config is
  `MatmulMultiCoreReuseMultiCast1DProgramConfig` with `mcast_in0=True`;
- interleaved L1 or DRAM inputs are supported;
- BF16 A/B/output is supported.

If `nnz` is supplied, it must equal the exact number of nonzero sparsity
entries. A mismatch can deadlock. Omitting it enables runtime inference.

The down projection uses `is_input_a_sparse=True`, giving A
`[A, E, M, K]`, B `[1, E, K, N]`, sparsity `[1, 1, A, E]`, and output
`[A, E, M, N]`.

### Fit to North

All North dimensions are tile multiples: 2048/32=64 and 768/32=24.
The op already has unit coverage with 128 experts
(`tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py`) and the Gemma4
implementation demonstrates both sparse-B gate/up and sparse-A down.

For decode, reshape `[1, batch, 1, 2048]` to `[1, 1, batch, 2048]`;
use scattered routing scores `[1, 1, batch, 128]` directly as the sparsity
tensor after row-major conversion. Sigmoid scores are positive, and top-k
indices are distinct, so the exact count is `batch * 8`. The values only select
work; they do not weight the matmul result. Preserve North semantics by applying
the original sigmoid routing scores after the down projection.

The packed gate/up candidate can be retained with B
`[1, 128, 2048, 1536]`, followed by a split/slice at 768. If that geometry is
unsupported or slower, use two sparse matmuls with
`[1, 128, 2048, 768]`. The down input becomes
`[1, 128, batch, 768]`.

For prefill, sparsity is per M-tile group, not per token. Group tokens in
internal 32-token chunks. The correct group mask activates an expert when any
token in that group routes to it; individual sigmoid weights are still applied
after down projection. A random 32-token top-8 group has an expected active
union of approximately
`128 * (1 - (120/128)^32) = 112` experts, so arithmetic sparsity may be modest.
The main potential wins are eliminating `repeat` and avoiding outputs for the
remaining experts.

Non-aligned public lengths must be internally padded to 32, with zero routing
scores for padded lanes, and sliced back afterward. Do not copy Gemma4's public
`seq_len % 32 == 0` assertion. For the dynamic union mask, omit `nnz` unless an
on-device exact count is demonstrably available; a guessed count is unsafe.

### Verify/refute experiments

Add a temporary test-only helper or candidate branch, without changing the
public decoder contract:

1. Decode batch 1 with nonzero synthetic weights:
   packed sparse gate/up, sparse down, `nnz=8`; compare the MoE chunk to the
   current fused chunk at PCC >= 0.99 and inspect all inactive-expert outputs as
   zero.
2. Decode batch 32: same with `nnz=256`; this specifically catches the common
   error of passing only `top_k` (the Gemma4 decode example does that and is
   only safe for a single M group).
3. Prefill lengths 1, 31, 32, 33, 65, 1023, 1024, 1025. Internally pad/chunk;
   construct a per-32-token expert-union mask; omit `nnz`; compare selected
   logical rows and final decoder PCC.
4. Run under watcher once, especially batch 32, because an `nnz` mismatch is a
   device hang/assert class of failure.
5. Benchmark warmed prefill and traced warmed decode at batches 1 and 32 against
   the current packed-dense-expert implementation. Accept only if end-to-end
   latency wins. Record active-expert union counts per prefill group to explain
   the result.

Minimal decode call shape:

```python
# x: [1, 1, B, 2048], gate_up_w: [1, 128, 2048, 1536]
# routes_rm: [1, 1, B, 128], exactly B*8 nonzeros
gate_up = ttnn.sparse_matmul(
    x,
    gate_up_w,
    sparsity=routes_rm,
    nnz=B * 8,
    program_config=gate_up_program_config,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    dtype=ttnn.bfloat16,
    output_tile=ttnn.Tile([32, 32]),
)
```

Use `_build_sparse_matmul_config` in
`models/demos/gemma4/tt/experts/decode.py` as a starting point, but derive its
grid from the actual Blackhole worker grid rather than retaining its hardcoded
8x8 search bound.

## 2. `ttnn.experimental.moe_compute`

### Exact public API

The binding is in
`ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/moe_compute_nanobind.cpp`:

```python
ttnn.experimental.moe_compute(
    tilize_input_tensor,
    tilize_expert_indices_tensor,
    tilize_expert_scores_tensor,
    tilize_expert_mapping_tensor,
    matmul_w0_w1_tensor,
    matmul_w2_tensor,
    *,
    layer_id,
    output_height_shard_dim,
    intermediate_size,
    has_bias=False,
    cluster_axis=None,
    topology=None,
    num_links=None,
    mux_core_range_set=None,
    output_memory_config=None,
    optional_output_tensor=None,
    optional_cross_device_semaphore=None,
    activation_type=None,
    compute_only=False,
    num_shared_experts_per_device=None,
)
```

`activation_type=ttnn.experimental.MoEActivationFunction.SWIGLU` matches
North's `SiLU(gate) * up`; the default `SILU` does not express the same
two-projection activation.

Inputs are not the current dense activation/routing tensors. They are dispatch
outputs: a BF16 row-major sparse activation buffer, UINT16 selected expert
indices, selected expert scores, and a rank-2 expert-to-device mapping. Weights
must be packed with:

```python
get_weight_core_shard_maps(device, hidden_size, intermediate_size)
get_weight_mem_configs(...)
prepare_w0_w1_tensor_for_moe_compute(...)
prepare_w2_tensor_for_moe_compute(...)
```

from `ttnn.experimental.moe_compute_utils`. The device weights are rank 6,
DRAM height-sharded, and ring-layout-specific; ordinary `[E,K,N]` tiled weights
are invalid.

### Fit and blockers for North

The arithmetic dimensions are legal:

- hidden 2048 and intermediate 768 are positive multiples of 32;
- 768 gives 24 intermediate tiles, at least one per Blackhole ring core;
- hidden tiles 64 select four data-parallel cores, which divides the public
  Blackhole ring size 8;
- BF16, top-k 8, no bias, no shared experts, and SWIGLU are supported.

For single-card evaluation, use `compute_only=True`, `cluster_axis=None`, and
do not pass an optional output tensor. It returns five tensors; output slot 4
is the expert matmul result. This mode skips the score-weighted combine, so a
separate local score multiply/reduction is required before comparing with
North. Full mode requires fabric combine parameters and is not justified for a
1x1 decoder.

Hardware/runtime blockers:

- Blackhole requires the production 11x10 worker grid;
- the device must be opened with `DispatchCoreAxis.COL`;
- indices and scores must be L1-sharded on the exact drain core `(10, 9)`;
- the current decoder accepts an already-open 1x1 `MeshDevice`, so changing the
  dispatch axis is an integration/fixture requirement, not a local op option;
- the op's shared-buffer token limit is
  `32 * num_data_parallel_cores * output_height_shard_dim`; at North's four
  data-parallel cores and default height 4 this is 512 tokens. Decode fits;
  the current 1024-token prefill chunk does not without a smaller internal
  chunk or a proven larger height configuration.

### Verify/refute experiments

Start from
`tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_compute_single_card.py`
and change only:

```text
experts_per_device=128
tokens_per_device=1, then 32
selected_experts_k=8
hidden_size=2048
N=768
has_bias=False
activation_type=MoEActivationFunction.SWIGLU
compute_only=True
```

Run with the same COL-dispatch fixture and watcher. First validate the op's
internal matmul output with the existing golden helper. Then apply the returned
route scores locally and reduce the K/expert dimension to compare to North's
current `_sparse_moe_chunk`. Measure packed-weight memory and setup time
separately from warmed runtime.

Refute this candidate for Stage 02 if any of these holds:

- the existing 1x1 decoder fixture cannot provide COL dispatch without changing
  the functional public/device contract;
- 128-expert packed weights or scratch buffers do not fit;
- score-weighted local combine requires enough extra data movement that warmed
  batch-1 and batch-32 decode do not beat `sparse_matmul`;
- trace capture/replay cannot safely include the op.

Do not attempt prefill until decode wins. If decode wins, use internal chunks no
larger than the validated token limit and test the same non-aligned logical
lengths as the sparse-matmul candidate.

## 3. `TTMoEDecode`

Exact constructor and forward APIs:

```python
TTMoEDecode(
    mesh_device,
    config: TTMoEDecodeConfig,
    torch_w0, torch_w1, torch_w2,
    ...,
)

decode.forward(tt_x, tt_scores, tt_indices, layer_id=0)
```

Weights are host tensors shaped `[layers, routed_experts, K, N]` and are packed
inside construction. Runtime inputs are decode-only:

- `tt_x [1, batch_per_device, 1, hidden]`;
- UINT16 indices `[batch_per_device, 1, 1, top_k]`;
- scores `[batch_per_device, 1, 1, top_k]`.

`TTMoEDecodeConfig` requires `mesh_shape`, integer `cluster_axis`, topology,
batch/device, hidden, top-k, routed/shared counts, bias policy, expert mapping,
and `compute.intermediate_size`/activation. The forward always performs A2A
dispatch, full `moe_compute`, post-combine tilize, fused score reduction, and
reduce-scatter/no-op selection. `ComputeConfig` has no `compute_only` field.

This conflicts with the present scope:

- the decoder contract is one already-open 1x1 device, not a distributed expert
  mesh;
- the module's value proposition is CCL dispatch/combine/reduce-scatter, which
  are unnecessary locally;
- it implements decode only, while Stage 02 must preserve arbitrary-length
  prefill;
- adopting it would replace routing/orchestration and device-opening contracts,
  not merely fuse the measured local op graph.

Therefore no hardware experiment is warranted for `TTMoEDecode` in Stage 02.
Its underlying `moe_compute` should be evaluated directly through the
single-card `compute_only` test above.

## Source anchors

- `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp`
- `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py`
- `models/demos/gemma4/tt/experts/decode.py`
- `models/demos/gemma4/tt/experts/prefill.py`
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/moe_compute_nanobind.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/moe_compute_device_operation.cpp`
- `ttnn/ttnn/_experimental/moe_compute_utils.py`
- `tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_compute_single_card.py`
- `models/common/modules/moe/tt_moe_decode.py`
- `models/common/modules/moe/tt_moe_decode_config.py`

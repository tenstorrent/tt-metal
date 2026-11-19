# Indexed decode storage and fused CCL source audit

CPU-only source audit, 2026-11-13. No devices were opened, no latency was
measured, and no implementation changes were made. Paths below are relative
to the repository root. The target remains four p300c Blackhole devices,
logical 1x4 ring, two links, H=2560, I=640, E=512, top-k=10, 48 layers.

## Indexed EP is not a direct replacement for the existing tensor lists

`tt/multichip_decoder.py:MeshMoe.__init__` stores three Python lists of 128
separately allocated mesh tensors. Each rank owns 128 complete experts.
Gate/up matrices are [2560,640], down matrices [640,2560], BFP4 TILE in
interleaved DRAM. `moe_fused_swiglu` accepts these lists, and its program
factory binds every individual weight buffer into its reader/writer runtime
arguments:

- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_program_factory.cpp:620-652`.
- `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:1103-1242` exposes
  `sparse_matmul` with one Tensor B, not a list or address table.
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp:305-323`
  computes the selected expert's tile address as
  `in1_tensor_start_tile_id + group_id * KtNt` in that single allocation.

The matching packed EP tensors would be gate/up [1,128,2560,640] and down
[1,128,640,2560], with the same BFP4 TILE encoding. A combined gate/up tensor
[1,128,2560,1280] is also possible if its output is split before SiLU/multiply;
it has the same aggregate tile payload. Native indexed sparse matmul supports
BFP4 (see `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py`,
especially `test_gather_gate_up` and `test_gather_down`). This is source and
test coverage, not a target-model numerical acceptance claim.

The required indices are same-device UINT16 ROW_MAJOR, one stick
[1,1,1,A], all IDs in [0,127], no `nnz`; A is the static logical index count.
The required ROW_MAJOR sparsity operand remains volume128, though its values
are not read in indexed mode. Gate/up use dense broadcast A
[1,1,T,2560], `is_input_a_sparse=False`, `is_input_b_sparse=True`, producing
[1,A,T,640]. Down uses compact A [1,A,T,640], both sparse flags true,
producing [1,A,T,2560]. Input and weight must be TILE; physical M is padded
to32. Compact input slot i addresses weight expert `indices[i]`, rather than
input expert `indices[i]`. These contracts are validated in
`.../matmul/device/sparse/sparse_matmul_device_operation.cpp:220-425`.

## Zero-copy API search and exact duplicate-storage cost

No safe Python zero-copy bridge from these existing 128 independent
allocations to the single packed Tensor was found. This conclusion is about
the present allocation/API contract, not the impossibility of a new kernel
or a different initial allocation strategy.

* `ttnn.view` (`.../operations/data_movement/view/view.cpp`) changes metadata
  on one buffer, requires equal logical volume and preserves its address.
  `ttnn.experimental.view` delegates to the same core view machinery.
  `ttnn/core/tensor/tensor_ops.cpp:450-529` explicitly creates the view at the
  input buffer's address. Neither accepts a list or an expert byte offset.
* `ttnn.OverlappedTensor` is an actual exposed API, but is a metadata carrier
  containing `fused_tensor`, shape, dtype, and byte_offset. It is not a
  `ttnn.Tensor` accepted by these matmuls. See
  `ttnn/cpp/ttnn-nanobind/tensor.cpp:746-800` and
  `ttnn/api/ttnn/tensor/overlapped_tensor.hpp`. It does not stitch independent
  allocations together or teach the sparse reader to use multiple buffers.
* C++ `MeshBuffer::create(..., optional<DeviceAddr> address)` can create a
  non-owning view at a supplied address (`tt_metal/api/tt-metalium/mesh_buffer.hpp`).
  That is a low-level building block for a **new packed-first alias design**,
  not a supported Python conversion of the current list. Such a design must
  prove interleaved bank/page mapping for every expert origin, lifetime
  ownership, mesh topology, and cached runtime buffer rebinding. Merely adding
  a byte count to an interleaved DRAM address is not sufficient. It would also
  require a new Tensor-facing binding/helper; it is outside this source-only
  audit and is not an already-working bridge.

A 32x32 BFP4 tile occupies576 bytes (512 mantissa plus64 exponent bytes),
per `tt_metal/impl/data_format/tile.cpp:135-145`. Payloads below exclude
allocator alignment/metadata, so duplication can only cost more.

| Storage per device | Bytes | GiB |
|---|---:|---:|
| One expert, one projection (1600 tiles) | 921600 | 0.0008583 |
| 128 experts, one projection | 117964800 | 0.1098633 |
| All three projections, one layer | 353894400 | 0.3295898 |
| One representation, all48 layers | 16986931200 | 15.8203125 |
| Lists plus packed tensors, all48 layers | 33973862400 | 31.640625 |
| Both representations plus current packed PLE | 41653903360 | 38.7932205 |

The authoritative `doc/context_contract.json` multichip plan reports PLE
7680040960 bytes/device, resident subtotal27704012800, and request state
3499430720 at context262144. Adding the duplicate experts gives
48190374720 bytes/device before runtime reserve. Even both expert copies
plus PLE alone exceed32GiB, independent of context. Therefore **persistent
dual representation for every layer is physically infeasible**, without
using a context reduction as a workaround.

This does not reject every packed candidate. Concrete options are:

1. A one-layer packed experiment needs an additional353894400 bytes/device
   and can discriminate kernel speed, but cannot be extrapolated to a
   stack that retains both formats.
2. One reusable packed staging arena has that same bounded extra cost, but
   copying lists into it for each layer requires at least353894400 bytes
   read plus353894400 bytes written per layer per step. Include that work in
   timing; across48 layers it is at least33973862400 DRAM bytes/device/step.
   Setup-only repacking is not a valid steady-state substitute.
3. Packed-only weights can use native sparse matmul for both decode and
   batched prefill. A mask-mode batched path can scan128 groups and execute
   only those selected by any token; it still needs per-token routing masks
   and weighted output assembly. This replaces the grouped list consumer,
   avoids duplication, and needs its own batched correctness/performance
   comparison. The arithmetic may include padded rows within active experts,
   as the grouped implementation does; it must not become dense all-expert
   execution. Do not reject this design merely because dual storage fails.

## EP dynamic active-count contract and TP4 comparison

For one token EP has0..10 active experts per rank, with average2.5. The
current indexed kernel compiles `num_active` from the indices Tensor's
logical volume (`.../sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:98-115,330`).
It has no device-resident count or invalid-index skip. A traced mesh program
cannot change that shape to the routing-dependent count on each rank.
Passing128 IDs executes128 experts; padding with arbitrary real experts and
masking afterwards is extra inactive-expert execution, not compact EP.

For an honest first indexed EP experiment, a host-prepared routing case can
measure equal, nonzero local active counts (e.g. distribution3/3/3/3 is not
possible for global k10; use2/2/2/2 with k8 only as a separate diagnostic,
or individual-device exact-count diagnostics). Such a diagnostic does not
validate production k10 mesh trace. The production candidates are:

- Native mask mode with128 groups, no exact `nnz`, which dynamically skips
  inactive groups and retains a static output contract. This is sparse
  selected execution but is **not** the indexed candidate.
- A fixed10-slot indexed path with each non-owned global selected ID mapped
  to a dedicated zero expert128, not a real inactive expert. Gate/up/down
  tables become129 groups and masks likewise129. This is trace-shaped and
  preserves real selected experts, but executes40 slots mesh-wide, including
  30 zero-expert slots. The zero expert costs2764800 bytes/layer/rank,
  132710400 for48 layers. Report the dummy work and compare it, rather than
  pretending the native kernel skips it. Build IDs on device and validate
  all-selected-on-one-rank and zero-local-expert cases.
- A device-count or sentinel-skip extension would remove dummy execution,
  but is new kernel work, not an available `indices` argument contract.

TP4 indexed decode avoids the variable-count problem: all ranks receive the
same10 selected global IDs; each owns all512 experts with I/4=160. Per-rank
weights are [1,512,2560,160] gate/up and [1,512,160,2560] down. This has the
same353894400-byte layer payload as EP4, and10 real expert fragments per
rank. Selected down outputs require an across-rank reduction, preferably
after the local weighted expert sum, retaining the routing weights exactly
once. The five-tile intermediate is a performance concern, not an API
invalidity: the indexed unit test's 1D config uses `per_core_N=1`; a separate
gate/up projection of160 columns exposes only5 column workers, versus20
for EP640. Combining gate/up exposes10 column tiles; down still has80
output-column tiles but only5 K tiles. Benchmark the actual gate/up/down,
reduction, and routing/assembly scope before declaring TP slower.

Padding local160 to192 adds20% expert storage:424673280 bytes/layer and
20384317440 bytes for48 layers, an extra3397386240. The present context
budget has only2975287488 bytes before runtime reserve at262144, so that
padding exceeds its headroom by422098752 bytes even before reserve. A
padded192 full stack therefore does not fit the existing advertised-context
budget without another demonstrated storage saving. The unpadded160
candidate remains feasible in the arithmetic budget; a single-layer packed
TP experiment cannot prove whole-stack dual-format feasibility.

## Fused matmul-RS and AG-matmul contracts for real decoder shapes

Both fused families exist in this checkout. Their public Python APIs have
no `cluster_axis` argument; the current1x4 mesh uses the complete device ring.
Use explicit persistent buffers and semaphores, and separate matmul worker
cores from CCL worker cores. Avoid assuming synchronous CCL arguments can
be copied unchanged.

`ttnn.experimental.matmul_reduce_scatter_async` requires dim3 and an explicit
`MatmulMultiCoreReuseMultiCastProgramConfig` (2D multicast). The factory
dereferences the config, so omitting it is not a useful fallback. Relevant
source is `.../experimental/ccl/matmul_reduce_scatter_async/`:
`matmul_reduce_scatter_async_nanobind.cpp:30-140`,
`device/matmul_reduce_scatter_async_device_operation.cpp:25-50`, and
`device/matmul_reduce_scatter_async_program_factory.cpp:35-130`.

For GDN WO, use real weight column shards: local input [1,1,T,1536], local
weight [1,1,1536,2560], partial [1,1,T,2560], RS result [1,1,T,640]. All
widths are tile aligned. A starting config on an8x8 MM grid is K block4,
per-core N10, N output block5, subblock1x1, per-core M at least1, with RS
workers offset(0,8), **after checking available target worker coordinates**.
This mirrors the existing four-chip Blackhole regression
`tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/watcher/test_matmul_reduce_scatter_fp32_bh.py`
while reducing N5120 to2560. Allocate three RS global semaphores plus a
barrier over all participating workers, an FP32 persistent intermediate
[1,1,Tphysical,2560] and FP32 output [1,1,Tphysical,640]. Use FP32 matmul
output/accumulation and BF16 real operands first; compare with an **unfused
FP32-output matmul then FP32 RS** control using the same program config.
This changes the baseline's BF16-partial boundary, so also compare final
residual/next-GR output against the existing path, not just that control.
Exercise logical1,2,31,32 and a chunked33; physical padding must remain
private. Feed RS directly to the hidden-residual consumer in
`tests/probe_multichip_residual_contract.py`, not an immediate all-gather.

`ttnn.experimental.all_gather_matmul_async` requires input and weight rank4,
input batch dimensions[1,1], gather dim3, distinct input/weight/bias buffers,
and explicit 1D or2D multicast matmul config. If the gathered buffer is
sharded, its number of core shards must equal ring size; use interleaved
DRAM first to avoid adding this constraint. See
`.../experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation.cpp:27-90`
and its `all_gather_matmul_async_nanobind.cpp`. The existing lifecycle and
cache/semaphore tests are in
`tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py`.

For GDN use local [1,1,T,1536], gathered [1,1,T,6144], real output-column
weight shard [1,1,6144,640], local output [1,1,T,640]. Start with two AG
semaphores plus barrier, persistent gathered DRAM storage, and a1D mcast-in0
config covering20 output tiles, K block4, per-core N1, subblock1x1. Reserve
separate CCL cores, e.g. a5x4 MM grid and CCL offset(0,6), verifying physical
worker bounds. This uses native FPU matmul instead of the shipping exact
GDN tree; numerical equivalence is an experiment, not an assumption.
Compare to ordinary AG plus native matmul first, then to the exact producer
and complete residual/next-GR outputs. Its640-column result naturally feeds
the same hidden-residual consumer (no extra output gather required).

Minimum viable device probe order for the owner: (1) unfused references and
one eager call, (2) repeated calls with changed inputs and stable buffers,
(3) whole producer-plus-next-consumer PCC, (4) warmed trace replay with
persistent semaphores/buffers, (5) profiler outside watcher and watcher in a
separate run. Report CCL, DRAM/data movement, matmul time, and complete
scope latency. Source validity and the existing generic regressions do not
establish target-shape correctness, stability, or a speedup.

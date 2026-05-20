# Phase 0 audit: `ttnn/cpp/ttnn/operations/reduction/generic/device/` (reduction op family)

Two `DeviceOperation` types share this directory and audit kernels in common (notably `reader_unary_reduce_universal_start_id.cpp` and `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`). Per the multi-device-op guidance, they are audited together.

- **`ReduceDeviceOperation`** (generic W/H/HW reduce)
  - `ReduceSingleCoreHwProgramFactory` (`reduce_op_single_core_hw_program_factory.cpp`)
  - `ReduceMultiCoreWProgramFactory` (`reduce_op_multi_core_w_program_factory.cpp`)
  - `ReduceMultiCoreHProgramFactory` (`reduce_op_multi_core_h_program_factory.cpp`)
- **`WelfordReduceDeviceOperation`** (variance/std via Welford)
  - `WelfordReduceProgramFactory` (`welford_reduce_program_factory.cpp`) — single factory, three internal variants: W, H, HW

**Kernels referenced** (audited):

- Compute: `reduce.cpp`, `reduce_w_neg.cpp`, `reduce_h_neg.cpp`, `reduce_hw_neg.cpp`, `welford_reduce_w.cpp`, `welford_reduce_h.cpp`, `welford_reduce_hw.cpp` (in `kernels/compute/`)
- Dataflow: `reader_unary_reduce_universal_start_id.cpp`, `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`, `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp`, `writer_welford_hw.cpp` (in `kernels/dataflow/`)
- Out-of-tree kernels referenced: `eltwise/unary/.../writer_unary_interleaved_start_id.cpp`, `data_movement/sharded/.../writer_unary_sharded.cpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**YELLOW (with user-confirmed override)** — port is feasible across all four factories. The factories are clean of UNSUPPORTED features except for one signal that the audit doc would normally flag RED, which has since been resolved in main but not yet reflected in the audit Appendix A:

- The H factory's width-sharded code path uses a `CBDescriptor` with `.buffer` set (dynamic CB / borrowed memory). The audit Appendix A's *Dynamic CircularBuffer* entry classifies this RED, but **borrowed-memory DFB support landed in main at commit `f06cb279620` (PR #44662)** — the audit appendix is stale on this point. The user has confirmed this is now supported and authorized proceeding.

Two Yellow side-issues for the user:

1. Device 2.0 DM has a handful of isolated CB-index-keyed helper holdovers (`get_tile_size(cb_id)`, `get_local_cb_interface(cb_id).fifo_page_size`) in kernels that otherwise consistently use the `CircularBuffer` wrapper. Recommend port-time cleanup.
2. The H-sharded reader (`reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp`) lacks `TensorAccessor` — but this is *causally downstream* of the dynamic CB (the kernel reads from a borrowed-memory CB, not via tensor address). Per the causal-link gate, this is not an independent gap; re-evaluate after the borrowed-memory DFB port translates the dynamic CB.

Handoff to the recipe doc is appropriate.

## Porting prerequisites

### ProgramDescriptor API: **GREEN**

All four factories populate a `tt::tt_metal::ProgramDescriptor` and use `KernelDescriptor`, `CBDescriptor`, plus `TensorAccessorArgs`. None use the imperative `host_api.hpp` builder calls (`CreateKernel`, `CreateCircularBuffer`, `SetRuntimeArgs`, etc.).

### Device 2.0 DM: **YELLOW (isolated holdovers)**

The kernels are substantively Device-2.0 compliant — they consistently use `Noc`, `CircularBuffer` wrapper objects, `TensorAccessor`, `dataflow_kernel_lib::*` helpers. A small number of free-function-style CB helpers remain, taking a `cb_id` while the wrapper is already in scope:

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | 26 | `get_tile_size(cb_id_in0)` | `cb_in0` (defined L31) |
| `kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | 34 | `get_tile_size(cb_id_in0)` | `cb_in0` (defined L44) |
| `kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | 37 | `get_tile_size(cb_id_in0)` | `cb_in0` (defined L40) |
| `kernels/dataflow/writer_welford_hw.cpp` | 55 | `get_tile_size(cb_partial)` | `cb_partial_obj` (defined L59) |
| `kernels/dataflow/writer_welford_hw.cpp` | 56 | `get_tile_size(cb_out)` | `cb_out_obj` (defined L61) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | 19 | `get_local_cb_interface(cb_id_out).fifo_page_size` | `cb` (defined L22) |

**Recommendation: port-time cleanup**, per the audit's default for the yellow tier. The replacements are 1-line mechanical (`get_tile_size(cb)` → `cb_obj.get_tile_size()`; the `get_local_cb_interface(...).fifo_page_size` case is slightly less canonical but Audrey can confirm the wrapper accessor).

Note: the `eltwise/unary` writer is shared by many ops; care is appropriate when modifying it. Leaving it alone is the conservative choice if the user prefers to keep the reduction-port diff minimal.

### TensorAccessor usage: **GREEN (with one causally-linked exception)**

Three of the four dataflow kernels use `TensorAccessor` with `TensorAccessorArgs<N>()` CTA plumbing:

- `reader_unary_reduce_universal_start_id.cpp` — `TensorAccessor(tensor_args, src_addr)` at L28
- `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` — `TensorAccessor(tensor_args, src_addr)` at L41
- `writer_welford_hw.cpp` — `TensorAccessor(dst_args, dst_addr)` at L63
- `eltwise/.../writer_unary_interleaved_start_id.cpp` — `TensorAccessor(dst_args, dst_addr)` at L31

The fourth dataflow kernel — `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` — does **not** use `TensorAccessor`. Per the causal-link gate (Check 3 preamble), this lack-of-TensorAccessor is *symptomatic* of the borrowed-memory CB approach: the kernel reads from `cb_in1` (sized at the shard, `.buffer = a.buffer()`) via a `UnicastEndpoint` walk over the shard's L1 memory; there is no tensor-memory read to convert to `TensorAccessor`. This is not an independent gap, and converting to `TensorAccessor` while the borrowed-memory CB exists would be incoherent — the kernel will need a redesign as part of the borrowed-memory-DFB port, not before.

The compute kernels read only from CBs; they are out of scope for Check 3.

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | GREEN | No references to `experimental::GlobalCircularBuffer`, `CreateGlobalCircularBuffer`, or `CBDescriptor::global_circular_buffer`. |
| Dynamic CircularBuffer (CB on borrowed memory) | YELLOW (override) | Two `.buffer` settings in H factory's width-sharded path; see detail below. |
| CBDescriptor `address_offset` (non-zero) | GREEN | No `.address_offset` field set anywhere. |
| Aliased Circular Buffers | GREEN | All `format_descriptors` initializers are single-element `{{CBFormatDescriptor{...}}}`. |
| GlobalSemaphore | N/A | No semaphores of any kind in this op family. |
| Non-zero semaphore initial value | N/A | No semaphores of any kind. |
| `ArgConfig::Runtime*` tensor-accessor flavors | GREEN | All `TensorAccessorArgs(*buffer)` calls are single-argument form. |
| `UpdateCircularBuffer*` | GREEN | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` calls in this op. |

### Dynamic CircularBuffer (CB on borrowed memory): **YELLOW (override)**

**Signal:** `CBDescriptor` literal with `.buffer` set to a non-null `Buffer*`.

**Sites:**

- `reduce_op_multi_core_h_program_factory.cpp:111` — `.buffer = a.buffer(),` (src1_cb / `c_1`, sized at the input shard)
- `reduce_op_multi_core_h_program_factory.cpp:148` — `.buffer = output.buffer(),` (output_cb / `c_3`, sized at the output shard)

Both occur inside `if (use_width_sharding)` branches; the interleaved code paths are clean.

**Expected resolution:** the audit appendix is stale here. Borrowed-memory DFB support landed at `f06cb279620` (PR #44662). The Metal 2.0 spec for this is in `tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp` — see `DataflowBufferSpec::borrowed_from` (line 95). User has confirmed and authorized the port to proceed.

## Path forward

Port all four factories. The H factory's width-sharded sub-path will use the new borrowed-memory DFB construct (`borrowed_from = <input tensor parameter name>`) rather than RTAs threading the shard address. The sharded reader kernel may or may not need updating depending on whether the borrowed-memory DFB attach mechanism changes the L1 address resolution model — the recipe doc will tell us.

The CB-index-keyed helper holdovers should be folded into the port (per the audit default) — six call sites, all one-line replacements.

## Questions for the user

1. **Stale audit appendix on dynamic CB.** Confirmed: borrowed-memory DFB now supported via `DataflowBufferSpec::borrowed_from`; proceed with port. *(Resolved per task setup.)*
2. **Device 2.0 DM holdovers.** Recommend folding the six CB-index-keyed call sites into the Metal 2.0 port as port-time cleanup. The shared `eltwise/unary` writer site (one occurrence) could optionally be deferred to a separate cleanup PR to keep this port's blast radius narrow. *(Default: port-time cleanup. Will confirm during Phase A.)*

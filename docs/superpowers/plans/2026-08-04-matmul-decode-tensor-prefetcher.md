# Tensor Prefetcher for `matmul_decode` (Full Width-Sharded) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `ttnn.experimental.matmul_decode` (full width-sharded factory only) take its weights from a DRAM-sender GlobalCircularBuffer filled by the DRISC tensor prefetcher, so the weight move happens off the command queue instead of via a blocking `to_memory_config` DRAM→L1 copy.

**Architecture:** Add an optional `global_cb` to the op. When present, the weight lives in DRAM as an ND-sharded tensor with one contiguous `[K, N/num_receivers]` slab per receiver core (this is exactly the "receiver-contiguous" layout the prefetcher is fastest at, and exactly the per-core shard `matmul_decode` already needs). The prefetcher delivers each receiver's whole slab as **one** GCB page (`block_count = 1`), so the compute kernel's existing random access over `[K_tiles, N_tiles_per_core]` keeps working unchanged in structure — it just reads from a GCB-backed circular buffer instead of a globally-allocated one. The reader gains a remote-CB wait/pop around the compute, and a small sync CB carries the "compute is done with in1" signal back to the reader.

**Tech Stack:** C++20, tt-metal `ProgramDescriptor` device-op path, TT-Metalium GlobalCircularBuffer / remote circular buffer API, DRISC tensor prefetcher (`ttnn.experimental.start_tensor_prefetcher`), nanobind, pytest.

## Global Constraints

- **Blackhole only.** The tensor prefetcher requires programmable DRAM cores: Blackhole, firmware >= 19.12.0.0, and either no harvested DRAM channels or a single device. Every new test must guard with `ttnn.experimental.is_tensor_prefetcher_supported(device)` and `pytest.skip` otherwise.
- **Check the hardware prerequisite before starting.** Because every test is skip-guarded, an unsupported machine makes the whole suite report green while validating nothing. Confirm support first:
  ```bash
  python_env/bin/python -c "
  import ttnn
  d = ttnn.open_device(device_id=0)
  print('supported:', ttnn.experimental.is_tensor_prefetcher_supported(d))
  ttnn.close_device(d)
  "
  ```
  This must print `supported: True`. If it prints `False`, look for the UMD line `Established firmware bundle version: X` in the device-open log — as of 2026-08-04 this machine reports 19.8.0, below the 19.12.0.0 floor in `tt_metal/llrt/firmware_capability.cpp`, so the plan cannot be validated here until the firmware is updated. Do **not** work around it with `TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1`: on firmware below the floor the syseng firmware occupies a DRAM core the prefetcher needs (issue #45751), so forcing it produces hangs that look like bugs in this feature.
- **Scope is the `FullWidthSharded` factory only.** `PartialWidthSharded` and `BatchedWidthSharded` must `TT_FATAL` if handed a `global_cb`. Do not attempt them in this plan.
- **`block_count = 1`.** One GCB page per receiver per invocation, equal to that receiver's entire `[K, N/num_receivers]` weight slab. Do not introduce K-blocking or streaming rotation — that is deliberately deferred.
- **No device-level fusion.** The prefetch request and the matmul stay two separate calls, paired by a host-side Python helper, mirroring the existing `ttnn/ttnn/_experimental/tensor_prefetcher_matmul.py`.
- **Existing behavior must not regress.** With `global_cb=None`, every code path must behave byte-identically to today. `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode.py` must keep passing unchanged.
- **Copyright header** on every new file:
  ```
  // SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
  //
  // SPDX-License-Identifier: Apache-2.0
  ```
  (`#` comments for Python.)
- **Build command** (run from repo root, takes ~20-40 min cold, a few minutes warm):
  ```bash
  ./build_metal.sh
  ```

---

## Background: why the design is shaped this way

Read this once before starting. It will save you from "improving" the design into a deadlock.

**A GCB is a FIFO ring with credit-based backpressure.** The sender (a DRISC core on a DRAM bank) writes pages into receiver L1 and atomically bumps a `pages_sent` counter on each receiver. The receiver spins in `remote_cb_wait_front(cb, n)` until `pages_sent - pages_acked >= n`, and calls `remote_cb_pop_front(cb, n)` to advance its read pointer and atomically bump `pages_acked` back on the sender. If the receiver pops a page the sender has not sent, or waits for a page the sender will never send, you get a silent hang, not an error.

**Why `block_count = 1`.** The `compute_full_width_sharded` kernel loops N on the outside and K on the inside:

```45:53:ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/compute_full_width_sharded.cpp
    for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
        tile_regs_acquire();
        for (uint32_t sender = 0; sender < num_senders; ++sender) {
            const uint32_t in0_base = sender * sender_slice_tiles;
            for (uint32_t kc = 0; kc < inA_K_tiles_per_core; ++kc) {
                const uint32_t in0_tile = in0_base + kc;
                const uint32_t k_global = sender * inA_K_tiles_per_core + kc;
                const uint32_t in1_tile = k_global * N_tiles_per_core + bw;
                matmul_block(in0_cb_id, in1_cb_id, in0_tile, in1_tile, 0, false, out_block_w, out_block_h, in0_block_w);
```

Every K tile is re-read once per output column `bw`. You therefore cannot consume K-blocks FIFO and pop them — the whole slab must be resident for the duration. Delivering it as a single page is the only arrangement that leaves this loop alone. (Streaming would require reordering to K-outer with dst accumulation; that is a separate, later piece of work.)

**Why the weight moves to DRAM ND-sharded.** The prefetcher reads from DRAM, not L1. "Receiver-contiguous" means the buffer holds `num_receivers` slabs, each the full `[K, N/num_receivers]` for exactly one receiver, so a receiver's page is one contiguous DRAM region. `make_recv_contig_weight` in `tests/ttnn/unit_tests/operations/prefetcher_common.py` builds exactly this via `ttnn.NdShardSpec`.

**Ring position == row-major core index.** The GCB's receiver ordering is defined by `bank_to_receivers`: bank `b` owns a set of receiver cores, and the *shard index* of the weight must equal the *ring position* of the receiver that gets it. `bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols)` maps ring position `p` to core `(p % ring_cols, p // ring_cols)`. With `ring_cols` set to the B-grid width and the B grid anchored at `(0, 0)`, ring position equals the row-major index of the core in `inputB_core_range_set` — which is the order `matmul_decode` already assigns N-columns to B cores. Pair `ROUND_ROBIN_1D` weights with `bank_receivers_strided`.

**An ND-sharded tensor has no legacy `shard_spec()`.** `input_tensor_b.memory_config().shard_spec()` returns `std::nullopt` for a `NdShardSpec` tensor. The full factory currently calls `.value()` on it in three places to get the B grid and shard shape. On the GCB path all of that must come from the GCB instead: `global_cb->receiver_cores()` for the grid, and `N_tiles / num_receivers` for the per-core N.

**Key reference implementation.** The gather_in0 1D matmul already does all of this against the legacy MeshWorkload factory. When in doubt about the wait/pop contract, read:
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` (lines 195-271, the batched `ENABLE_GLOBAL_CB` path)
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` lines 2128-2234 (remote CB creation)
- `tt_metal/impl/buffers/prefetcher_matmul_design.md` (the prefetcher↔receiver contract)

And for the `ProgramDescriptor`-flavored GCB CB (rather than the legacy `CircularBufferConfig`), the pattern is:

```180:189:ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp
    desc.cbs.push_back(CBDescriptor{
        .total_size = remote_cb_size,
        .core_ranges = reader_core_range,
        .remote_format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(remote_cb_index),
            .data_format = max_tile_size_df,
            .page_size = L1_ALIGNMENT,  // set to 16B so that the infra won't update write pointers to wrong location
        }}},
        .global_circular_buffer = std::addressof(global_cb),
    });
```

We need the *receiver* variant, which sets **both** `format_descriptors` (the local alias index, page = one tile) and `remote_format_descriptors` (the remote index, page = one whole slab), plus `global_circular_buffer`.

---

## File Structure

| Path | Responsibility | Change |
|---|---|---|
| `ttnn/cpp/ttnn/operations/experimental/matmul_decode/matmul_decode.hpp` | Public C++ signature | Modify — add `global_cb` |
| `ttnn/cpp/ttnn/operations/experimental/matmul_decode/matmul_decode.cpp` | Forwards to prim | Modify — add `global_cb` |
| `ttnn/cpp/ttnn/operations/experimental/matmul_decode/matmul_decode_nanobind.cpp` | Python binding | Modify — add `global_cb` kwarg |
| `.../device/matmul_decode_device_operation.hpp` | Attributes struct, prim decl | Modify — add `global_cb` field + param |
| `.../device/matmul_decode_device_operation.cpp` | Validation, output spec, prim impl | Modify — GCB validation + GCB-derived output grid |
| `.../device/full_width_sharded_program_factory.cpp` | Full-width program descriptor | Modify — GCB-backed in1 CB, sync CB, kernel wiring |
| `.../device/kernels/dataflow/reader_full_width_sharded.cpp` | A gather + in1 activation | Modify — remote CB wait/pop under `ENABLE_GLOBAL_CB` |
| `.../device/kernels/compute/compute_full_width_sharded.cpp` | The matmul | Modify — in1 wait/pop + sync push under `ENABLE_GLOBAL_CB` |
| `ttnn/ttnn/_experimental/tensor_prefetcher_matmul_decode.py` | Host-side prefetch+matmul pairing | **Create** |
| `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py` | End-to-end tests | **Create** |
| `tt_metal/impl/buffers/prefetcher_matmul_design.md` | Prefetcher↔receiver contract doc | Modify — document the second consumer |

Task order is dependency order: API surface (1) → validation and output spec (2) → program factory and kernels (3) → Python pairing helper (4) → end-to-end tests (5) → docs (6). Tasks 1-3 are not independently *useful*, but each is independently *compilable and testable*, which is what the gates check.

---

### Task 1: Thread `global_cb` through the op API

Adds the parameter end-to-end and rejects it everywhere it isn't implemented yet. No behavioral change when it is `std::nullopt`.

**Files:**
- Modify: `ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/matmul_decode_device_operation.hpp:25-36` (attributes) and `:80-87` (prim decl)
- Modify: `ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/matmul_decode_device_operation.cpp:14-23` (factory select) and `:315-421` (prim impl)
- Modify: `ttnn/cpp/ttnn/operations/experimental/matmul_decode/matmul_decode.hpp:15-20`
- Modify: `ttnn/cpp/ttnn/operations/experimental/matmul_decode/matmul_decode.cpp:11-18`
- Modify: `ttnn/cpp/ttnn/operations/experimental/matmul_decode/matmul_decode_nanobind.cpp:16-47`
- Test: `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py` (create)

**Interfaces:**
- Produces: `ttnn.experimental.matmul_decode(a, b, *, partial_width_sharded=False, dtype=None, output_mem_config=None, global_cb=None)` — the new keyword accepts a `ttnn.GlobalCircularBuffer` or `None`.
- Produces: `MatmulDecodeDeviceOperation::operation_attributes_t::global_cb` of type `std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>`, defaulted to `std::nullopt`.

- [ ] **Step 1: Write the failing test**

Create `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py`:

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for matmul_decode fed by the DRISC tensor prefetcher.

The weight lives in DRAM as an ND-sharded (receiver-contiguous) tensor -- one
[K, N/num_receivers] slab per B core -- and the prefetcher pushes each slab into
the matmul's in1 circular buffer through a DRAM-sender GlobalCircularBuffer.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    bank_receivers_strided,
    make_recv_contig_weight,
    tensor_prefetcher_session,
)


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    """Skip unless programmable DRAM cores are available on this device."""
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip(
            "programmable DRAM cores unavailable (need Blackhole, firmware >= 19.12.0.0, "
            "and either no harvested DRAM channels or a single device)"
        )


def test_matmul_decode_accepts_global_cb_kwarg(device):
    """The global_cb keyword exists and defaults to None (no behavior change)."""
    m, k, n = 32, 1024, 2048
    num_a_cores = 32
    num_b_cores = n // 64

    torch.manual_seed(0)
    pt_a = torch.randn((m, k), dtype=torch.bfloat16)
    pt_b = torch.randn((k, n), dtype=torch.bfloat16)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    a_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_a_cores - 1, 0))})
    b_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_b_cores - 1, 0))})
    a = ttnn.from_torch(
        pt_a,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (m, k // num_a_cores),
            core_grid=a_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    b = ttnn.from_torch(
        pt_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (k, n // num_b_cores),
            core_grid=b_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    out = ttnn.experimental.matmul_decode(a, b, global_cb=None)
    assert_with_pcc(ref, ttnn.to_torch(out).float(), 0.99)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py::test_matmul_decode_accepts_global_cb_kwarg -v`
Expected: FAIL with a `TypeError` about an unexpected keyword argument `global_cb`.

- [ ] **Step 3: Add the field to the attributes struct and the prim declaration**

In `.../device/matmul_decode_device_operation.hpp`, add the include near the other includes at the top:

```cpp
#include <tt-metalium/global_circular_buffer.hpp>
```

Extend `operation_attributes_t` (replace the existing struct body):

```cpp
    struct operation_attributes_t {
        int M;
        int N;
        int K;
        std::optional<MemoryConfig> output_mem_config;
        std::optional<DataType> output_dtype;
        bool partial_width_sharded = false;
        int batch = 1;
        int b_blocks = 1;
        int n_blocks = 1;
        // DRAM-sender GlobalCircularBuffer feeding in1 from the tensor prefetcher.
        // When set, the weight is a DRAM ND-sharded tensor (one slab per receiver)
        // and the receiver grid comes from the GCB, not from a legacy shard spec.
        std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb = std::nullopt;
    };
```

Extend the prim declaration at the bottom of the same file:

```cpp
namespace ttnn::prim {
ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation::tensor_return_value_t matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded = false,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt);
}  // namespace ttnn::prim
```

- [ ] **Step 4: Populate the field in the prim implementation and reject unsupported factories**

In `.../device/matmul_decode_device_operation.cpp`, change the prim signature and both `operation_attributes_t` constructions.

Signature (line 316):

```cpp
ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation::tensor_return_value_t matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded,
    std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb) {
```

Batched construction (currently lines 374-384) — append `global_cb` as the last initializer:

```cpp
            auto operation_attributes = OperationType::operation_attributes_t{
                M,
                N,
                K,
                output_mem_config,
                dtype.has_value() ? std::optional<DataType>(*dtype) : std::nullopt,
                /*partial_width_sharded=*/false,
                batch,
                b_blocks,
                n_blocks,
                global_cb,
            };
```

Non-batched construction (currently lines 410-417) — the intermediate fields must be spelled out because `global_cb` comes after them:

```cpp
    auto operation_attributes = OperationType::operation_attributes_t{
        M,
        N,
        K,
        output_mem_config,
        dtype.has_value() ? std::optional<DataType>(*dtype) : std::nullopt,
        partial_width_sharded,
        /*batch=*/1,
        /*b_blocks=*/1,
        /*n_blocks=*/1,
        global_cb,
    };
```

Add the rejection at the top of `validate_on_program_cache_miss` (insert immediately after the existing `const auto& input_tensor_b = ...` line):

```cpp
    if (operation_attributes.global_cb.has_value()) {
        TT_FATAL(
            !operation_attributes.partial_width_sharded,
            "matmul_decode: global_cb (tensor prefetcher weights) is only supported by the full width-sharded "
            "program factory, but partial_width_sharded was requested.");
        TT_FATAL(
            !(input_tensor_a.logical_shape().rank() == 4 && operation_attributes.batch > 1),
            "matmul_decode: global_cb (tensor prefetcher weights) is only supported by the full width-sharded "
            "program factory, but a batched (rank-4, batch={}) activation selects the batched factory.",
            operation_attributes.batch);
    }
```

- [ ] **Step 5: Forward the parameter from the public API**

In `.../matmul_decode.hpp`, add the include and extend the signature:

```cpp
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::experimental {

// Decode-optimized matmul C = A @ B for L1 width-sharded operands (full, partial, or batched B layout).
// `global_cb`: optional DRAM-sender GlobalCircularBuffer supplying in1 from the tensor prefetcher
// (full width-sharded factory only; the weight must then be a DRAM ND-sharded tensor).
Tensor matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded = false,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt);

}  // namespace ttnn::experimental
```

In `.../matmul_decode.cpp`:

```cpp
Tensor matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded,
    std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb) {
    return ttnn::prim::matmul_decode(
        input_tensor_a, input_tensor_b, partial_width_sharded, dtype, output_mem_config, global_cb);
}
```

- [ ] **Step 6: Add the nanobind keyword**

In `.../matmul_decode_nanobind.cpp`, add to the docstring's Keyword Args block (after `output_mem_config`):

```
            global_cb (ttnn.GlobalCircularBuffer, optional): DRAM-sender global circular buffer
                supplying the weights from the tensor prefetcher. Requires the full width-sharded
                factory and a DRAM ND-sharded (receiver-contiguous) weight. Defaults to None.
```

and add the argument after `nb::arg("output_mem_config") = nb::none()`:

```cpp
        nb::arg("global_cb") = nb::none());
```

- [ ] **Step 7: Build**

Run: `./build_metal.sh`
Expected: build succeeds with no errors.

- [ ] **Step 8: Run the new test and the existing regression suite**

Run:
```bash
pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py -v
pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode.py -v
```
Expected: both PASS.

- [ ] **Step 9: Commit**

```bash
git add ttnn/cpp/ttnn/operations/experimental/matmul_decode tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py
git commit -m "feat: add optional global_cb parameter to matmul_decode

Plumbs a DRAM-sender GlobalCircularBuffer through the op API so the
tensor prefetcher can supply in1. Rejected by the partial and batched
factories; no behavior change when unset."
```

---

### Task 2: Accept a DRAM ND-sharded weight and derive the receiver grid from the GCB

Makes validation and output-spec computation work when the weight has no legacy shard spec. After this task the op still fails at program creation (Task 3 fixes that), so the test asserts on the *validation* boundary only.

**Files:**
- Modify: `.../device/matmul_decode_device_operation.cpp:36-39` (the in1 WIDTH_SHARDED fatal) and `:257-305` (`compute_output_specs`)
- Test: `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py`

**Interfaces:**
- Consumes: `operation_attributes_t::global_cb` from Task 1.
- Produces: when `global_cb` is set, the output tensor is L1 WIDTH_SHARDED across `global_cb->receiver_cores()` with shard `[round_up(M, tile_h), N / num_receivers]`.
- Produces: a free helper in the anonymous namespace of `matmul_decode_device_operation.cpp`:
  `uint32_t gcb_num_receivers(const tt::tt_metal::experimental::GlobalCircularBuffer& gcb)` returning `gcb.receiver_cores().num_cores()`. Task 3 uses the same expression inline; keep the helper file-local.

- [ ] **Step 1: Write the failing test**

Append to `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py`:

```python
def _make_gcb_and_operands(device, m, k, n, num_a_cores, num_slabs=2):
    """Build the activation, the DRAM receiver-contiguous weight, and the GCB.

    The B/receiver grid is the rectangle `_num_cores_to_rectangle_core_range_set`
    picks for `num_b_cores`, anchored at (0, 0). `bank_receivers_strided` maps ring
    position p to core `(p % ring_cols, p // ring_cols)`, so passing that rectangle's
    WIDTH as `ring_cols` makes ring position equal the core's row-major index -- which
    is the order matmul_decode assigns N-columns to B cores, and the order the weight's
    ND shards are laid out in. Passing `num_b_cores` instead is only correct when the
    rectangle happens to be a single row, and silently produces wrong results otherwise.
    """
    torch.manual_seed(0)
    num_dram_banks = device.dram_grid_size().x
    num_b_cores = n // 64
    assert num_b_cores % num_dram_banks == 0, f"{num_b_cores} receivers must divide across {num_dram_banks} banks"
    recv_per_bank = num_b_cores // num_dram_banks

    pt_a = torch.randn((m, k), dtype=torch.bfloat16)
    pt_b = torch.randn((k, n), dtype=torch.bfloat16)

    a_grid = _num_cores_to_rectangle_core_range_set(num_a_cores, device)
    b_grid = _num_cores_to_rectangle_core_range_set(num_b_cores, device)
    # Rectangle width == the row-major stride, hence ring_cols.
    ring_cols = b_grid.bounding_box().grid_size().x
    a = ttnn.from_torch(
        pt_a,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (m, k // num_a_cores),
            core_grid=a_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    weight = make_recv_contig_weight(
        device,
        pt_b.reshape(1, 1, k, n),
        num_dram_banks=num_dram_banks,
        ring_size=num_b_cores,
        dtype=ttnn.bfloat16,
        distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )

    # One GCB page == one receiver's whole [K, N/num_b_cores] slab.
    tile_bytes = 2048  # bfloat16 32x32
    slab_bytes = (k // 32) * ((n // num_b_cores) // 32) * tile_bytes
    gcb_size = num_slabs * slab_bytes

    bank_to_receivers = [
        (b, bank_receivers_strided(b, recv_per_bank, num_dram_banks, ring_cols=ring_cols))
        for b in range(num_dram_banks)
    ]
    gcb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(device, bank_to_receivers, gcb_size)
    return pt_a, pt_b, a, weight, gcb, num_b_cores


def test_matmul_decode_gcb_output_spec(device):
    """With a GCB, the output grid/shard come from the GCB receivers, not from a
    legacy shard spec on the DRAM weight (which has none)."""
    m, k, n = 32, 1024, 2048
    _, _, a, weight, gcb, num_b_cores = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)

    out = ttnn.experimental.matmul_decode(a, weight, global_cb=gcb)

    assert tuple(out.shape) == (m, n)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert out.memory_config().shard_spec.shape == [m, n // num_b_cores]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py::test_matmul_decode_gcb_output_spec -v`
Expected: FAIL with `Input tensor B must be in width sharded memory layout, but got ND_SHARDED`.

- [ ] **Step 3: Relax the in1 memory-layout check**

In `.../device/matmul_decode_device_operation.cpp`, replace the in1 `TT_FATAL` at lines 36-39 with:

```cpp
    if (operation_attributes.global_cb.has_value()) {
        // Prefetcher-fed weights live in DRAM as an ND-sharded (receiver-contiguous) tensor:
        // one contiguous [K, N/num_receivers] slab per receiver core. There is no legacy
        // shard spec on such a tensor, so the receiver grid comes from the GCB.
        TT_FATAL(
            input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED,
            "matmul_decode with global_cb requires input tensor B to be ND_SHARDED, but got {}",
            input_tensor_b.memory_config().memory_layout());
        TT_FATAL(
            input_tensor_b.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "matmul_decode with global_cb requires input tensor B to live in DRAM (the prefetcher reads DRAM), "
            "but it is in L1");
        const uint32_t num_receivers = gcb_num_receivers(*operation_attributes.global_cb);
        TT_FATAL(
            num_receivers > 0 && operation_attributes.N % static_cast<int>(num_receivers) == 0,
            "matmul_decode with global_cb requires N ({}) to be divisible by the GCB receiver count ({})",
            operation_attributes.N,
            num_receivers);
        // Note: the NdShardSpec lives on the Tensor, not on the MemoryConfig, and the shard count
        // comes from the buffer's BufferDistributionSpec -- same accessors the recv-contig weight
        // validator in ttnn/core/global_circular_buffer.cpp uses.
        const auto& nd = input_tensor_b.nd_shard_spec();
        TT_FATAL(
            nd.has_value(),
            "matmul_decode with global_cb requires input tensor B to carry an NdShardSpec (receiver-contiguous "
            "layout)");
        const auto& bds = input_tensor_b.buffer()->buffer_distribution_spec();
        TT_FATAL(
            bds.has_value(),
            "matmul_decode with global_cb requires input tensor B to have a BufferDistributionSpec");
        const uint32_t num_shards = static_cast<uint32_t>(bds->num_shards());
        TT_FATAL(
            num_shards == num_receivers,
            "matmul_decode with global_cb requires one weight shard per GCB receiver, but the weight has {} shards "
            "and the GCB has {} receivers",
            num_shards,
            num_receivers);
        TT_FATAL(
            static_cast<int>(nd->shard_shape[-2]) == operation_attributes.K &&
                static_cast<int>(nd->shard_shape[-1]) == operation_attributes.N / static_cast<int>(num_receivers),
            "matmul_decode with global_cb requires each weight shard to be [K, N/num_receivers] = [{}, {}], but got "
            "[{}, {}]",
            operation_attributes.K,
            operation_attributes.N / static_cast<int>(num_receivers),
            nd->shard_shape[-2],
            nd->shard_shape[-1]);
    } else {
        TT_FATAL(
            input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
            "Input tensor B must be in width sharded memory layout, but got {}",
            input_tensor_b.memory_config().memory_layout());
    }
```

Add the file-local helper just below the `using namespace` / opening of `namespace ttnn::operations::experimental::matmul_decode {` in the same file:

```cpp
namespace {
uint32_t gcb_num_receivers(const tt::tt_metal::experimental::GlobalCircularBuffer& gcb) {
    return gcb.receiver_cores().num_cores();
}
}  // namespace
```

- [ ] **Step 4: Derive the output spec from the GCB**

In `compute_output_specs`, replace line 279 (`CoreRangeSet output_core_range_set = input_tensor_b.memory_config().shard_spec().value().grid;`) and line 280 with:

```cpp
    // A prefetcher-fed weight is ND-sharded in DRAM and has no legacy shard spec, so the
    // receiver (= output) grid comes from the GCB instead.
    CoreRangeSet output_core_range_set = operation_attributes.global_cb.has_value()
                                             ? operation_attributes.global_cb->receiver_cores()
                                             : input_tensor_b.memory_config().shard_spec().value().grid;
    int output_num_cores = output_core_range_set.num_cores();
```

The `partial_width_sharded` branch immediately below is unreachable with a GCB (Task 1 rejects that combination), so it needs no change.

- [ ] **Step 5: Build**

Run: `./build_metal.sh`
Expected: build succeeds.

- [ ] **Step 6: Run the test**

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py::test_matmul_decode_gcb_output_spec -v`

Expected at this point: the op gets **past** validation and output-spec computation and fails inside `FullWidthSharded::create_descriptor`, with a `std::bad_optional_access` or a `TT_FATAL` mentioning the B shard shape — because the factory still calls `input_tensor_b.memory_config().shard_spec().value()`. That is the correct intermediate state; Task 3 fixes it.

If instead it fails with an ND_SHARDED / receiver-count message, your validation is wrong — fix it before moving on.

- [ ] **Step 7: Verify no regression**

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/matmul_decode_device_operation.cpp
git commit -m "feat: validate DRAM ND-sharded weights and take the output grid from the GCB

A prefetcher-fed weight has no legacy shard spec, so matmul_decode must
source the receiver grid from the GlobalCircularBuffer and check the
one-slab-per-receiver geometry itself."
```

---

### Task 3: GCB-backed in1 in the full width-sharded factory and kernels

The core of the feature. After this task the op runs end to end when someone else fills the GCB.

**Files:**
- Modify: `.../device/full_width_sharded_program_factory.cpp` (in1 CB, sync CB, reader/compute wiring, B-grid derivation)
- Modify: `.../device/kernels/dataflow/reader_full_width_sharded.cpp`
- Modify: `.../device/kernels/compute/compute_full_width_sharded.cpp`
- Test: `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py`

**Interfaces:**
- Consumes: `operation_attributes_t::global_cb` (Task 1), validated geometry (Task 2).
- Produces: the on-device contract the prefetcher must satisfy — **exactly one GCB page per receiver per `matmul_decode` invocation**, each page `K_tiles * (N_tiles / num_receivers) * in1_tile_size` bytes, laid out K-row-major (tile `(kt, nt)` at offset `(kt * N_tiles_per_core + nt) * in1_tile_size`). Task 4's Python helper must queue `block_count = 1` to match.
- Produces: reader compile-time args extended to indices 19 (`remote_cb_index`) and 20 (`sync_cb_index`); reader runtime arg index 3 (`is_in1_receiver`); compute compile-time arg index 4 (`sync_cb_index`); define `ENABLE_GLOBAL_CB` on both kernels.

- [ ] **Step 0: Fix Task 2's output-spec test so it cannot wedge the device**

**Do this before anything else. A first attempt at Task 3 hung the machine here and had to be recovered with `tt-smi -r`.**

Task 2 left behind `test_matmul_decode_gcb_output_spec`, which calls `matmul_decode(a, weight, global_cb=gcb)` without ever starting the prefetcher, then asserts only on `out.shape` and `out.memory_config()`. That is harmless while the factory ignores the GCB. The moment this task makes the reader actually wait on a GCB page, that call enqueues a program whose reader blocks forever in `remote_cb_wait_front` on a page nobody will ever send.

The failure mode is nastier than a plain hang: **shape and memory-config assertions never force the device to finish**, so the test itself reports PASS and the wedged program is only discovered when the *next* test hangs during device sync. Expect the symptom to appear one test away from its cause.

Replace the body of `test_matmul_decode_gcb_output_spec` so it is actually fed, and so it forces completion:

```python
def test_matmul_decode_gcb_output_spec(device):
    """With a GCB, the output grid/shard come from the GCB receivers, not from a
    legacy shard spec on the DRAM weight (which has none).

    The prefetcher must run even though this test only checks the output spec: once
    the in1 CB is GCB-backed the reader blocks until its page arrives, and a matmul
    launched without a prefetch request would wedge the device for every later test.
    """
    m, k, n = 32, 1024, 2048
    _, _, a, weight, gcb, num_b_cores = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, 1)], global_cb=gcb)
        out = ttnn.experimental.matmul_decode(a, weight, global_cb=gcb)
        ttnn.synchronize_device(device)

    assert tuple(out.shape) == (m, n)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert out.memory_config().shard_spec.shape == [m, n // num_b_cores]
```

If `ttnn.synchronize_device` is not the correct spelling in this build, use whatever this repo's tests use to force device completion (reading the tensor back with `ttnn.to_torch(out)` also works and is acceptable here).

**General rule for every test in this file from now on:** a test that invokes `matmul_decode` with a `global_cb` must (a) run inside `tensor_prefetcher_session` with a matching prefetch request, and (b) force device completion before leaving the session. Never assert only on metadata.

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
def test_matmul_decode_prefetched_weights_pcc(device):
    """Full end-to-end: prefetcher pushes each receiver's weight slab into the GCB,
    matmul_decode consumes it, result matches torch."""
    m, k, n = 32, 1024, 2048
    pt_a, pt_b, a, weight, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, 1)], global_cb=gcb)
        out = ttnn.experimental.matmul_decode(a, weight, global_cb=gcb)
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result, 0.99)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py::test_matmul_decode_prefetched_weights_pcc -v`
Expected: FAIL inside `FullWidthSharded::create_descriptor` (bad optional access on the weight's missing shard spec), same as Task 2 Step 6.

**Run every test in this file with a timeout from here on** — e.g. `timeout 300 pytest ... -v`. A credit-protocol mistake hangs rather than fails, and an untimed run will sit on the device indefinitely and require a `tt-smi -r` to recover. If a run does hang: kill it, run `python_env/bin/tt-smi -r` to reset all eight boards, confirm recovery with a trivial `ttnn.add` on device 0, and only then reason about the sent/acked accounting. Do not respond to a hang by adding retries or by widening the GCB.

- [ ] **Step 3: Derive the B grid and per-core N from the GCB in the factory**

In `full_width_sharded_program_factory.cpp`, add near the top of the file with the other includes:

```cpp
#include <tt-metalium/global_circular_buffer.hpp>
```

Replace the B-grid line (currently line 90) and the B shard-shape block (currently lines 114-124) as follows.

Replace line 90:

```cpp
    const bool use_global_cb = operation_attributes.global_cb.has_value();
    // A prefetcher-fed weight is ND-sharded in DRAM (no legacy shard spec), so the receiver
    // grid is the GCB's receiver set. Otherwise it is the weight's own shard grid.
    auto inputB_core_range_set = use_global_cb ? operation_attributes.global_cb->receiver_cores()
                                               : input_tensor_b.memory_config().shard_spec().value().grid;
```

Replace lines 114-124:

```cpp
    uint32_t inB_N_tiles_per_core;
    if (use_global_cb) {
        const uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);
        const uint32_t num_receivers = inputB_core_range_set.num_cores();
        TT_FATAL(
            N_tiles % num_receivers == 0,
            "full_width_sharded matmul_decode with global_cb requires N in tiles ({}) to be divisible by the GCB "
            "receiver count ({})",
            N_tiles,
            num_receivers);
        inB_N_tiles_per_core = N_tiles / num_receivers;
    } else {
        std::array<uint32_t, 2> inputB_shard_shape = input_tensor_b.memory_config().shard_spec().value().shape;
        TT_FATAL(
            inputB_shard_shape[0] == (K_tiles * tt::constants::TILE_HEIGHT),
            "Input tensor B shard shape {} [0] must be equal to K_tiles {} * tile height {}",
            inputB_shard_shape[0],
            K_tiles,
            tt::constants::TILE_HEIGHT);
        TT_FATAL(
            inputB_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
            "Input tensor B must have a width that is divisible by the tile width");
        inB_N_tiles_per_core = inputB_shard_shape[1] / tt::constants::TILE_WIDTH;
    }
```

- [ ] **Step 4: Replace the in1 CB descriptor with the GCB-backed variant**

Still in `full_width_sharded_program_factory.cpp`, add the two new CB indices next to the existing ones (currently lines 127-130):

```cpp
    constexpr uint32_t in0_cb_index = CBIndex::c_0;
    constexpr uint32_t in1_cb_index = CBIndex::c_1;
    constexpr uint32_t out_cb_index = CBIndex::c_2;
    constexpr uint32_t full_in0_cb_index = CBIndex::c_3;
    // GCB path only: c_4 carries "compute is done reading in1" back to the reader so it can
    // release the GCB page; c_31 is the remote (GCB) index aliased onto the local in1 CB.
    constexpr uint32_t sync_cb_index = CBIndex::c_4;
    constexpr uint32_t remote_cb_index = CBIndex::c_31;
```

Replace the in1 `desc.cbs.push_back(...)` block (currently lines 143-153) with:

```cpp
    // One GCB page is a receiver's entire [K, N/num_receivers] weight slab. The local alias
    // (in1_cb_index) is tile-paged so the compute kernel can index tiles as it does today;
    // the remote index is slab-paged so one page-credit == one whole slab.
    const uint32_t in1_slab_num_tiles = K_tiles * inB_N_tiles_per_core;
    const uint32_t in1_slab_bytes = in1_slab_num_tiles * in1_tile_size;
    if (use_global_cb) {
        const auto& gcb = *operation_attributes.global_cb;
        // Round the window down to a whole number of slabs; the remote CB requires its total
        // size to be a multiple of its page size.
        const uint32_t gcb_window_bytes = (gcb.size() / in1_slab_bytes) * in1_slab_bytes;
        TT_FATAL(
            gcb_window_bytes >= in1_slab_bytes,
            "full_width_sharded matmul_decode with global_cb needs a GCB of at least one weight slab per receiver "
            "({} B), but the GCB holds {} B",
            in1_slab_bytes,
            gcb.size());
        desc.cbs.push_back(CBDescriptor{
            .total_size = gcb_window_bytes,
            .core_ranges = inputB_core_range_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = in1_cb_index,
                .data_format = in1_data_format,
                .page_size = in1_tile_size,
                .tile = in1_tile_desc,
            }}},
            .remote_format_descriptors = {{CBFormatDescriptor{
                .buffer_index = remote_cb_index,
                .data_format = in1_data_format,
                .page_size = in1_slab_bytes,
            }}},
            .global_circular_buffer = std::addressof(gcb),
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = tt::tt_metal::hal::get_l1_alignment(),
            .core_ranges = inputB_core_range_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = sync_cb_index,
                .data_format = tt::DataFormat::UInt32,
                .page_size = tt::tt_metal::hal::get_l1_alignment(),
            }}},
        });
    } else {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in1_slab_bytes,
            .core_ranges = all_compute_cores_with_bbox,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = in1_cb_index,
                .data_format = in1_data_format,
                .page_size = in1_tile_size,
                .tile = in1_tile_desc,
            }}},
            .buffer = input_tensor_b.buffer(),
        });
    }
```

Add the HAL include at the top of the file if it is not already pulled in transitively:

```cpp
#include <tt-metalium/hal.hpp>
```

- [ ] **Step 5: Wire the new args and define into the kernels**

Extend `reader_compile_time_args` (currently ends at line 224 with `K_tiles * inB_N_tiles_per_core,`) by appending two entries:

```cpp
        in1_cb_index,
        in1_slab_num_tiles,
        remote_cb_index,
        sync_cb_index,
    };
```

(The first two lines above replace the existing `in1_cb_index,` / `K_tiles * inB_N_tiles_per_core,` entries — reuse `in1_slab_num_tiles` so the value is defined once.)

In `build_reader_kernel`, set the define and add the fourth runtime arg. Replace the `reader_kernel_desc.config = ...` assignment and the runtime-arg loop with:

```cpp
        reader_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = noc,
        };
        if (use_global_cb) {
            reader_kernel_desc.defines["ENABLE_GLOBAL_CB"] = "1";
        }

        reader_kernel_desc.runtime_args.reserve(cores.size());
        for (const auto& core : cores) {
            const auto it = sender_id_by_core.find(core);
            const bool is_sender = it != sender_id_by_core.end();
            const uint32_t sender_id = is_sender ? it->second : 0;
            // The reader runs on the merged A-and-B bounding box, but only B cores are GCB
            // receivers and only they have the in1 / sync CBs configured.
            const bool is_in1_receiver = inputB_core_range_set.contains(core);
            reader_kernel_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    static_cast<uint32_t>(is_sender),
                    sender_id,
                    static_cast<uint32_t>(role_of(core)),
                    static_cast<uint32_t>(is_in1_receiver)});
        }
```

Extend the compute kernel descriptor (currently lines 331-340):

```cpp
    compute_kernel_desc.compile_time_args = {
        M_tiles,
        K_tiles,
        inB_N_tiles_per_core,
        inA_K_tiles_per_core,
        sync_cb_index,
    };
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    if (use_global_cb) {
        compute_kernel_desc.defines["ENABLE_GLOBAL_CB"] = "1";
    }
```

- [ ] **Step 6: Update the reader kernel**

Replace `reader_full_width_sharded.cpp` in full:

```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#ifdef ENABLE_GLOBAL_CB
#include "api/remote_circular_buffer.h"
#endif

// Gathers width(K)-sharded A onto every core via two-hub gather/broadcast.
//
// in1 (weights) arrive one of two ways:
//   - default: the in1 CB is globally allocated over the L1-resident weight shard, so the
//     reader only has to declare its tiles available.
//   - ENABLE_GLOBAL_CB: the weights are pushed into a DRAM-sender GlobalCircularBuffer by the
//     tensor prefetcher. Exactly one remote page per invocation carries this receiver's whole
//     [K, N/num_receivers] slab. The reader waits for that page, hands the tiles to compute
//     through the local alias CB, and releases the page only after compute signals (via the
//     sync CB) that it has finished reading -- releasing earlier would let the prefetcher
//     overwrite weights still in use.
void kernel_main() {
    constexpr uint32_t in0_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t full_in0_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t shard_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(5);
    uint32_t mcast_x_start = get_compile_time_arg_val(6);
    uint32_t mcast_y_start = get_compile_time_arg_val(7);
    uint32_t mcast_x_end = get_compile_time_arg_val(8);
    uint32_t mcast_y_end = get_compile_time_arg_val(9);
    constexpr uint32_t stage_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t hub0_noc_x = get_compile_time_arg_val(12);
    constexpr uint32_t hub0_noc_y = get_compile_time_arg_val(13);
    constexpr uint32_t hub1_noc_x = get_compile_time_arg_val(14);
    constexpr uint32_t hub1_noc_y = get_compile_time_arg_val(15);
    constexpr uint32_t split_H = get_compile_time_arg_val(16);
    constexpr uint32_t in1_cb_index = get_compile_time_arg_val(17);
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t remote_cb_index = get_compile_time_arg_val(19);
    constexpr uint32_t sync_cb_index = get_compile_time_arg_val(20);

    const uint32_t is_sender = get_arg_val<uint32_t>(0);
    const uint32_t sender_id = get_arg_val<uint32_t>(1);
    const uint32_t role = get_arg_val<uint32_t>(2);
    const uint32_t is_in1_receiver = get_arg_val<uint32_t>(3);

    constexpr uint32_t full_num_tiles = num_senders * shard_num_tiles;
    const uint32_t shard_size_bytes = shard_num_tiles * tile_size_bytes;

    // NOC_1 uses an inverted coordinate system.
    if (noc_index == 1) {
        std::swap(mcast_x_start, mcast_x_end);
        std::swap(mcast_y_start, mcast_y_end);
    }

    Noc noc;
    CircularBuffer in0_cb(in0_cb_index);
    CircularBuffer full_in0_cb(full_in0_cb_index);
    Semaphore<> stage_sem(stage_sem_id);
    Semaphore<> done_sem(done_sem_id);
    UnicastEndpoint hub;

#ifdef ENABLE_GLOBAL_CB
    if (is_in1_receiver) {
        CircularBuffer in1_cb(in1_cb_index);
        in1_cb.reserve_back(in1_num_tiles);
        experimental::remote_cb_wait_front(remote_cb_index, 1);
        in1_cb.push_back(in1_num_tiles);
    }
#else
    {
        CircularBuffer in1_cb(in1_cb_index);
        in1_cb.reserve_back(in1_num_tiles);
        in1_cb.push_back(in1_num_tiles);
    }
#endif
    full_in0_cb.reserve_back(full_num_tiles);

    const bool is_hub0 = (role == 1);
    const bool is_hub1 = (role == 2);

    if (is_sender) {
        const bool owned_by_hub0 = sender_id < split_H;
        const uint32_t hub_x = owned_by_hub0 ? hub0_noc_x : hub1_noc_x;
        const uint32_t hub_y = owned_by_hub0 ? hub0_noc_y : hub1_noc_y;
        const uint32_t dst_offset_bytes = sender_id * shard_size_bytes;

        // full_in0_cb is at the same L1 offset on every core, so the local write ptr is the remote dst addr.
        const uint32_t dst_l1_addr = full_in0_cb.get_write_ptr() + dst_offset_bytes;
        noc.async_write(
            in0_cb, hub, shard_size_bytes, {.offset_bytes = 0}, {.noc_x = hub_x, .noc_y = hub_y, .addr = dst_l1_addr});
        noc.async_write_barrier();
        stage_sem.up(noc, hub_x, hub_y, 1);
        noc.async_atomic_barrier();
    }

    if (is_hub0 || is_hub1) {
        const uint32_t region_first = is_hub0 ? 0 : split_H;
        const uint32_t region_count = is_hub0 ? split_H : (num_senders - split_H);
        const uint32_t region_offset_bytes = region_first * shard_size_bytes;
        const uint32_t region_size_bytes = region_count * shard_size_bytes;

        if (region_count > 0) {
            stage_sem.wait(region_count);

            noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                use<CircularBuffer::AddrSelector::WRITE_PTR>(full_in0_cb),
                full_in0_cb,
                region_size_bytes,
                num_receivers,
                {.offset_bytes = region_offset_bytes},
                {.noc_x_start = mcast_x_start,
                 .noc_y_start = mcast_y_start,
                 .noc_x_end = mcast_x_end,
                 .noc_y_end = mcast_y_end,
                 .offset_bytes = region_offset_bytes});
            noc.async_write_barrier();
        }

        // inc_multicast excludes the sender; self must use atomic NOC up() (local up() can race with the other hub).
        const uint32_t self_noc_x = is_hub0 ? hub0_noc_x : hub1_noc_x;
        const uint32_t self_noc_y = is_hub0 ? hub0_noc_y : hub1_noc_y;
        done_sem.inc_multicast(noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, 1, num_receivers - 1);
        done_sem.up(noc, self_noc_x, self_noc_y, 1);
        noc.async_atomic_barrier();
    }

    done_sem.wait(2);
    full_in0_cb.push_back(full_num_tiles);

#ifdef ENABLE_GLOBAL_CB
    if (is_in1_receiver) {
        // Compute signals here once it has finished reading every in1 tile of this slab.
        CircularBuffer sync_cb(sync_cb_index);
        sync_cb.wait_front(1);
        sync_cb.pop_front(1);
        experimental::remote_cb_pop_front(remote_cb_index, 1);
        // Persist the remote read pointer so the next invocation resumes at the right ring offset.
        experimental::update_remote_cb_config_in_l1(remote_cb_index);
        noc.async_atomic_barrier();
    }
#endif

    noc.async_write_barrier();
    noc.async_read_barrier();
}
```

- [ ] **Step 7: Update the compute kernel**

Replace `compute_full_width_sharded.cpp` in full:

```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

// C = A @ B per core. full_in0 is sender-major. matmul_block does not reduce over kt_dim; K is accumulated in the loop.
//
// With ENABLE_GLOBAL_CB the in1 tiles arrive through a GCB-backed circular buffer instead of a
// globally-allocated one, so this kernel must actually wait on them and, when done, tell the
// reader (via the sync CB) that the GCB page can be released.
using namespace ckernel;
void kernel_main() {
    constexpr uint32_t out_block_w = 1;

    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t sync_cb_id = get_compile_time_arg_val(4);

    constexpr uint32_t out_block_h = M_tiles;
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;

    constexpr uint32_t in0_num_tiles = M_tiles * K_tiles;
    constexpr uint32_t in1_num_tiles = K_tiles * N_tiles_per_core;
    constexpr uint32_t num_senders = K_tiles / inA_K_tiles_per_core;
    constexpr uint32_t sender_slice_tiles = M_tiles * inA_K_tiles_per_core;

    CircularBuffer in0_cb(in0_cb_id);
    CircularBuffer out_cb(out_cb_id);
#ifdef ENABLE_GLOBAL_CB
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer sync_cb(sync_cb_id);
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, out_cb_id);

    in0_cb.wait_front(in0_num_tiles);
#ifdef ENABLE_GLOBAL_CB
    in1_cb.wait_front(in1_num_tiles);
#endif

    matmul_block_init(in0_cb_id, in1_cb_id, false, out_block_w, out_block_h, in0_block_w);

    out_cb.reserve_back(M_tiles * N_tiles_per_core);
    for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
        tile_regs_acquire();
        for (uint32_t sender = 0; sender < num_senders; ++sender) {
            const uint32_t in0_base = sender * sender_slice_tiles;
            for (uint32_t kc = 0; kc < inA_K_tiles_per_core; ++kc) {
                const uint32_t in0_tile = in0_base + kc;
                const uint32_t k_global = sender * inA_K_tiles_per_core + kc;
                const uint32_t in1_tile = k_global * N_tiles_per_core + bw;
                matmul_block(in0_cb_id, in1_cb_id, in0_tile, in1_tile, 0, false, out_block_w, out_block_h, in0_block_w);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t mt = 0; mt < out_block_h; ++mt) {
            pack_tile<true>(mt, out_cb_id, mt * N_tiles_per_core + bw);
        }
        tile_regs_release();
    }
    out_cb.push_back(M_tiles * N_tiles_per_core);

    in0_cb.pop_front(in0_num_tiles);
#ifdef ENABLE_GLOBAL_CB
    // Every in1 tile has been read; release the local alias and let the reader ack the GCB page.
    in1_cb.pop_front(in1_num_tiles);
    sync_cb.reserve_back(1);
    sync_cb.push_back(1);
#endif
}
```

- [ ] **Step 8: Build**

Run: `./build_metal.sh`
Expected: build succeeds.

- [ ] **Step 9: Run the end-to-end test**

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py -v`
Expected: all three tests PASS.

**If the test hangs** (no output, no failure), it is a credit mismatch, not a math bug. Kill it and diagnose in this order:
1. Confirm the prefetcher pushed exactly one page per receiver: the request must be `(weight, 1)`, i.e. `block_count = 1`. A `block_count` of anything else makes the reader's `remote_cb_wait_front(remote_cb_index, 1)` and the sender's page count disagree.
2. Confirm `gcb.size()` is at least one slab and that `in1_slab_bytes` in the factory matches `K_tiles * N_tiles_per_core * tile_size` for the weight's dtype.
3. Re-run with watcher enabled to see which core is stuck:
   ```bash
   TT_METAL_WATCHER=10 pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py::test_matmul_decode_prefetched_weights_pcc -v
   ```
   Then read `generated/watcher/watcher.log`.

**If PCC is wrong but nothing hangs**, the receiver ordering is off: the weight shard that landed on ring position `p` is not the N-column block that the core at row-major index `p` computes. Verify `bank_receivers_strided` was called with `ring_cols` equal to the B-grid width and that the weight used `ROUND_ROBIN_1D`.

- [ ] **Step 10: Verify no regression**

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode.py -v`
Expected: PASS. This is the important gate — the `#else` branches must preserve today's behavior exactly.

- [ ] **Step 11: Commit**

```bash
git add ttnn/cpp/ttnn/operations/experimental/matmul_decode/device
git commit -m "feat: feed matmul_decode in1 from a DRAM-sender global circular buffer

The full width-sharded factory can now back its in1 CB with a GCB filled
by the DRISC tensor prefetcher: one page per receiver carries that
receiver's whole weight slab, the reader waits on it, and compute signals
through a sync CB before the page is released."
```

---

### Task 4: Host-side prefetch + matmul_decode pairing helper

Mirrors `ttnn/ttnn/_experimental/tensor_prefetcher_matmul.py` so callers cannot drift the GCB, the block count, and the consumer apart.

**Files:**
- Create: `ttnn/ttnn/_experimental/tensor_prefetcher_matmul_decode.py`
- Test: `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py`

**Interfaces:**
- Produces: `prefetch_and_matmul_decode(input_tensor_a, weight, *, global_cb, cq_id=None, **matmul_kwargs) -> ttnn.Tensor`
- Produces: `make_matmul_decode_gcb(device, weight, bank_to_receivers, *, num_slabs=2) -> ttnn.GlobalCircularBuffer`

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
def test_prefetch_and_matmul_decode_helper(device):
    """The paired helper issues the prefetch and the matmul against the same GCB."""
    from ttnn._experimental.tensor_prefetcher_matmul_decode import prefetch_and_matmul_decode

    m, k, n = 32, 1024, 2048
    pt_a, pt_b, a, weight, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        out = prefetch_and_matmul_decode(a, weight, global_cb=gcb)
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result, 0.99)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py::test_prefetch_and_matmul_decode_helper -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ttnn._experimental.tensor_prefetcher_matmul_decode'`.

- [ ] **Step 3: Write the helper module**

Create `ttnn/ttnn/_experimental/tensor_prefetcher_matmul_decode.py`:

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Combined DRAM-core prefetch + consuming ``matmul_decode``.

``ttnn.experimental.queue_tensor_prefetcher_request`` (fills a DRAM-sender
GlobalCircularBuffer over NOC, off the command queue) and the
``ttnn.experimental.matmul_decode`` that drains that GCB are always issued as a
pair, against the *same* GCB. As two separate calls the caller has to (a) hand
both the same ``global_cb`` and (b) pass a prefetch ``block_count`` that matches
what the matmul expects -- two couplings nothing enforces.

``prefetch_and_matmul_decode`` issues the pair from one call site so they cannot
drift. ``block_count`` is always 1: the full width-sharded ``matmul_decode``
compute kernel indexes in1 tiles by absolute position across the whole
``[K, N/num_receivers]`` slab (its N loop is outermost, so every K tile is
re-read once per output column), which means the entire slab must be resident
for the duration and therefore has to arrive as a single GCB page.

This is a host-side composition, not a device-level fusion: the prefetch runs on
the DRAM-core (DRISC) path off the command queue while the matmul is dispatched
normally. The pairing composes with trace capture -- pass the recording CQ as
``cq_id`` and the request is captured (and replayed) alongside the matmul.
"""

import ttnn

# One GCB page per receiver per invocation, carrying that receiver's whole weight slab.
_BLOCK_COUNT = 1

# Default GCB depth in slabs. Two lets the prefetcher land the next invocation's weights
# while the current matmul is still computing; one serializes them.
_DEFAULT_NUM_SLABS = 2


def _slab_bytes(weight, num_receivers):
    """Bytes of one receiver's [K, N/num_receivers] weight slab, in whole tiles."""
    k = weight.shape[-2]
    n = weight.shape[-1]
    if n % num_receivers != 0:
        raise ValueError(f"weight N={n} must be divisible by the receiver count {num_receivers}")
    # Verified against this build: ttnn.Tensor exposes `.tile` and `.dtype`; there is no
    # `.tensor_spec` and no `ttnn.datatype_to_dataformat_converter` in Python.
    # `tile.get_tile_size(dtype)` returns 2048 for bfloat16, 1088 for bfloat8_b, 576 for bfloat4_b.
    tile = weight.tile
    tile_bytes = tile.get_tile_size(weight.dtype)
    k_tiles = k // tile.tile_shape[0]
    n_tiles_per_recv = (n // num_receivers) // tile.tile_shape[1]
    return k_tiles * n_tiles_per_recv * tile_bytes


def make_matmul_decode_gcb(device, weight, bank_to_receivers, *, num_slabs=_DEFAULT_NUM_SLABS):
    """Build a DRAM-sender GCB sized to hold ``num_slabs`` weight slabs per receiver.

    Args:
        device: the mesh device.
        weight: the DRAM ND-sharded (receiver-contiguous) weight this GCB will carry.
            Only its shape/dtype are read here.
        bank_to_receivers: list of ``(dram_bank_id, ttnn.CoreRangeSet)`` pairs. The
            receiver at ring position ``p`` must be the B core whose row-major index in
            the matmul's B grid is ``p``, and the weight's shard ``p`` must be that
            receiver's N-column block. Build it with ``bank_receivers_strided`` for
            ``ROUND_ROBIN_1D`` weights or ``bank_receivers_contiguous`` for
            ``CONTIGUOUS_1D``.
        num_slabs: GCB depth in whole slabs. Must be at least 1; 2 (the default) lets the
            prefetch of the next invocation overlap the current matmul.

    Returns:
        A ``ttnn.GlobalCircularBuffer`` to pass as ``global_cb`` to both the prefetch
        request and ``matmul_decode``.
    """
    if num_slabs < 1:
        raise ValueError(f"num_slabs must be >= 1, got {num_slabs}")
    num_receivers = sum(crs.num_cores() for _, crs in bank_to_receivers)
    size = num_slabs * _slab_bytes(weight, num_receivers)
    return ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(device, bank_to_receivers, size)


def prefetch_and_matmul_decode(
    input_tensor_a,
    weight,
    *,
    global_cb,
    cq_id=None,
    **matmul_kwargs,
):
    """Queue a DRAM-core prefetch of ``weight`` into ``global_cb``, then run the
    ``matmul_decode`` that consumes it.

    Args:
        input_tensor_a: activation (in0), L1 width-sharded along K.
        weight: DRAM ND-sharded (receiver-contiguous) weight (in1), one
            ``[K, N/num_receivers]`` shard per GCB receiver.
        global_cb: DRAM-sender GlobalCircularBuffer shared by the prefetch and the matmul.
        cq_id: command queue for the prefetch request. When that CQ is mid trace-capture
            the request is captured into the trace. Defaults to the current command queue.
        **matmul_kwargs: forwarded to ``ttnn.experimental.matmul_decode``
            (e.g. ``dtype``, ``output_mem_config``).

    Returns:
        The ``matmul_decode`` output tensor.
    """
    device = input_tensor_a.device()
    ttnn.experimental.queue_tensor_prefetcher_request(
        device,
        [(weight, _BLOCK_COUNT)],
        global_cb=global_cb,
        cq_id=cq_id,
    )
    return ttnn.experimental.matmul_decode(
        input_tensor_a,
        weight,
        global_cb=global_cb,
        **matmul_kwargs,
    )
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py::test_prefetch_and_matmul_decode_helper -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ttnn/ttnn/_experimental/tensor_prefetcher_matmul_decode.py tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py
git commit -m "feat: add prefetch_and_matmul_decode host pairing helper

Issues the prefetch request and the consuming matmul_decode from one call
site against the same GCB, so the block count and buffer cannot drift."
```

---

### Task 5: Coverage across shapes, repeat invocations, and a perf comparison

Proves the feature holds up beyond one shape and that the GCB ring pointer survives back-to-back invocations (the thing most likely to be subtly wrong).

**Files:**
- Modify: `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py`

**Interfaces:**
- Consumes: `_make_gcb_and_operands` (Task 2), `prefetch_and_matmul_decode` (Task 4).

- [ ] **Step 1: Write the failing tests**

Append to the test file:

```python
@pytest.mark.parametrize(
    "m, k, n",
    [
        (1, 1024, 2048),
        (8, 1024, 2048),
        (32, 1024, 2048),
        (32, 2048, 2048),
        (32, 1024, 4096),
    ],
)
def test_matmul_decode_prefetched_shapes(device, m, k, n):
    """PCC across the decode M range and a couple of K/N shapes."""
    from ttnn._experimental.tensor_prefetcher_matmul_decode import prefetch_and_matmul_decode

    pt_a, pt_b, a, weight, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        out = prefetch_and_matmul_decode(a, weight, global_cb=gcb)
        result = ttnn.to_torch(out).float()

    assert_with_pcc(ref, result, 0.99)


def test_matmul_decode_prefetched_repeated_invocations(device):
    """Back-to-back prefetch+matmul pairs against one GCB.

    Each pair consumes exactly one page per receiver, so the GCB read and write
    pointers must stay in lockstep across invocations. A drift here shows up as a
    hang on iteration 2 or 3, or as iteration N returning iteration N-1's data.
    """
    from ttnn._experimental.tensor_prefetcher_matmul_decode import prefetch_and_matmul_decode

    m, k, n = 32, 1024, 2048
    pt_a, pt_b, a, weight, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        for i in range(4):
            out = prefetch_and_matmul_decode(a, weight, global_cb=gcb)
            result = ttnn.to_torch(out).float()
            assert_with_pcc(ref, result, 0.99)
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py -v`
Expected: all PASS.

If `test_matmul_decode_prefetched_repeated_invocations` hangs on iteration 2, the remote read pointer is not persisting: verify `experimental::update_remote_cb_config_in_l1(remote_cb_index)` is called at the end of the reader (Task 3 Step 6) and that the GCB window is a whole number of slabs.

- [ ] **Step 2b: Add negative-path tests for the rejection and validation branches**

Tasks 1 and 2 added `TT_FATAL`s that no test exercises: the factory-scope rejections (Task 1) and the in1 layout checks (Task 2). Both are reachable — a reviewer confirmed `compute_output_specs` sources the output grid purely from `global_cb->receiver_cores()` on the GCB path, so it cannot crash ahead of validation. Cover the two that are cheap to construct and most likely to be hit by a real user:

```python
def test_matmul_decode_global_cb_rejects_partial_width_sharded(device):
    """global_cb is only wired through the full width-sharded factory."""
    m, k, n = 32, 1024, 2048
    _, _, a, weight, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)

    with pytest.raises(RuntimeError, match="only supported by the full width-sharded"):
        ttnn.experimental.matmul_decode(a, weight, partial_width_sharded=True, global_cb=gcb)


def test_matmul_decode_global_cb_rejects_l1_weight(device):
    """With a global_cb the weight must be the DRAM ND-sharded tensor the prefetcher reads."""
    m, k, n = 32, 1024, 2048
    pt_a, pt_b, a, _, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)

    num_b_cores = n // 64
    b_grid = _num_cores_to_rectangle_core_range_set(num_b_cores, device)
    l1_weight = ttnn.from_torch(
        pt_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (k, n // num_b_cores),
            core_grid=b_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    with pytest.raises(RuntimeError, match="ND_SHARDED"):
        ttnn.experimental.matmul_decode(a, l1_weight, global_cb=gcb)
```

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py -k rejects -v`
Expected: both PASS.

If `TT_FATAL` surfaces in Python as an exception type other than `RuntimeError`, match on the actual type rather than loosening the `match=` pattern — the message assertion is the point of the test.

- [ ] **Step 3: Add the perf comparison test**

Append:

```python
def test_matmul_decode_prefetched_vs_l1_resident_perf(device):
    """Profiler-visible comparison: prefetched weights vs. today's per-call DRAM->L1 copy.

    Run under the profiler to compare the two signposted regions:
        pytest ... -k prefetched_vs_l1_resident --profiler
    """
    from tracy import signpost
    from ttnn._experimental.tensor_prefetcher_matmul_decode import prefetch_and_matmul_decode

    m, k, n = 32, 1024, 2048
    num_b_cores = n // 64
    pt_a, pt_b, a, weight, gcb, _ = _make_gcb_and_operands(device, m, k, n, num_a_cores=32)
    ref = pt_a.to(torch.float32) @ pt_b.to(torch.float32)

    # Baseline operands: the same weight as a DRAM-interleaved tensor that must be copied
    # into L1 width-sharded form before every matmul -- what LinearDecode does today.
    b_grid = _num_cores_to_rectangle_core_range_set(num_b_cores, device)
    l1_weight_config = ttnn.create_sharded_memory_config(
        (k, n // num_b_cores),
        core_grid=b_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    dram_weight = ttnn.from_torch(
        pt_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        signpost("matmul_decode_prefetched")
        for _ in range(4):
            prefetched_out = prefetch_and_matmul_decode(a, weight, global_cb=gcb)
        signpost("matmul_decode_l1_copy")
        for _ in range(4):
            l1_weight = ttnn.to_memory_config(dram_weight, l1_weight_config)
            baseline_out = ttnn.experimental.matmul_decode(a, l1_weight)
            l1_weight.deallocate()
        signpost("stop")

        prefetched = ttnn.to_torch(prefetched_out).float()
        baseline = ttnn.to_torch(baseline_out).float()

    assert_with_pcc(ref, prefetched, 0.99)
    assert_with_pcc(ref, baseline, 0.99)
```

- [ ] **Step 4: Run the full test file**

Run: `pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_prefetcher.py
git commit -m "test: cover prefetched matmul_decode across shapes, repeats, and perf

Adds the shape sweep, a back-to-back invocation test that catches GCB ring
pointer drift, and a signposted comparison against the per-call DRAM->L1
weight copy the feature replaces."
```

---

### Task 6: Document the second GCB consumer

`prefetcher_matmul_design.md` currently describes exactly one receiver (the gather_in0 matmul) and states invariants that a reader will otherwise assume are universal — notably "the receiver `num_blocks` compile arg equals ring_size", which `matmul_decode` deliberately violates by using `block_count = 1`.

**Files:**
- Modify: `tt_metal/impl/buffers/prefetcher_matmul_design.md`

**Interfaces:**
- Consumes: the on-device contract produced by Task 3.

- [ ] **Step 1: Add the new receiver to the overview**

In `tt_metal/impl/buffers/prefetcher_matmul_design.md`, after the paragraph ending "Receivers see the same byte layout regardless of which prefetcher produced it." (around line 39), insert:

```markdown
A second receiver exists: `ttnn.experimental.matmul_decode` (full width-sharded
factory only), see
`ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/full_width_sharded_program_factory.cpp`.
It is not a ring — every core receives the full activation via a two-hub
gather/broadcast — so there is no per-receiver rotation and no rotated
sub-ringbuffer. Each receiver is sent exactly **one** page per invocation
carrying its entire `(K, N/num_receivers)` weight slab (`block_count = 1`),
because that compute kernel loops N on the outside and re-reads every K tile
once per output column, so the whole slab must stay resident. It always uses the
receiver-contiguous layout (§6).
```

- [ ] **Step 2: Scope invariant 2 to the gather_in0 receiver**

Replace invariant 2 in §8 ("Cross-component invariants", around line 464):

```markdown
2. **`num_blocks` matches between prefetcher and receiver.** For the gather_in0
   receiver, `num_blocks == ring_size`, enforced because both prefetchers derive
   `num_blocks = num_senders * num_receivers_per_sender` from the GCB. For the
   `matmul_decode` receiver, `num_blocks == 1` (one whole-slab page per receiver
   per invocation). The invariant is that the pushed page count per receiver
   equals the count the receiver waits on — not that either equals ring_size.
```

- [ ] **Step 3: Scope invariant 4 the same way**

Replace invariant 4:

```markdown
4. **All tensors in one prefetcher request share `block_count`** as far as the
   consuming matmul is concerned — the receiver's wait count is a compile-time
   arg, so a request feeding one consumer must not mix per-tensor block counts
   that consumer does not expect. (The prefetcher itself permits per-tensor
   `block_count`; the constraint lives on the receiver side.)
```

- [ ] **Step 4: Verify the doc renders and links resolve**

Run:
```bash
grep -n "matmul_decode" tt_metal/impl/buffers/prefetcher_matmul_design.md
ls ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/full_width_sharded_program_factory.cpp
```
Expected: the grep shows the new references, and the `ls` confirms the referenced path exists.

- [ ] **Step 5: Commit**

```bash
git add tt_metal/impl/buffers/prefetcher_matmul_design.md
git commit -m "docs: document matmul_decode as a second GCB receiver

Records the block_count = 1 whole-slab contract and scopes the
num_blocks == ring_size invariant to the gather_in0 receiver."
```

---

## Out of scope (deliberately)

Do not attempt these as part of this plan. They are follow-on work.

- **Streaming (`block_count = K_blocks`).** Requires reordering `compute_full_width_sharded` to K-block-outer with accumulation held in dst across blocks, plus a rotation table on the prefetch request. Feasible for decode (M is one tile, `N_tiles_per_core` is small, so the accumulator fits in dst) and it is what shrinks the GCB from a full slab to two K-blocks — but it is a separate change with its own correctness risk.
- **`PartialWidthSharded` and `BatchedWidthSharded`.** Both have one contiguous weight shard per B core, so the same receiver-contiguous mapping applies, but each has its own factory, reader, and (for partial) a cross-core reduction to re-derive the grid for. `BatchedLinearDecode` is the highest-value target after this plan lands, since it does the per-call `to_memory_config` copy explicitly.
- **Migrating `models/experimental/deepseek_v4_flash/tt/layers.py`.** `LinearDecode` and `BatchedLinearDecode` should adopt this only after the op-level feature is proven, and adoption needs a `start_tensor_prefetcher` lifecycle at model init plus a decision about where `wait_for_cq_on_tensor_prefetcher` fences go.
- **Dual senders per bank.** `create_global_circular_buffer_for_tensor_prefetcher`'s `support_multi_receiver_shards=False` doubles per-bank bandwidth and is compatible with receiver-contiguous weights, but it changes the bank→receiver mapping and should be tuned once correctness is established.

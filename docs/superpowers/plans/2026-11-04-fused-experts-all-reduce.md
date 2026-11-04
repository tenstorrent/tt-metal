# Fused Experts + TP All-Reduce Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `ttnn.experimental.deepseek.moe.fused_experts` all-reduce its per-chip TP partial over one mesh axis itself, so the one-row TP decode MoE no longer launches a separate `width_sharded_all_reduce` after it.

**Architecture:** On the 6-expert TP path, `fused_experts` already leaves its per-chip partial on group 0's 16 down cores, `(0..1, 0..7)`, in exactly the width-sharded layout that `width_sharded_all_reduce` consumes. The fused op runs that all-reduce's three roles in the same program:

- **Group-0 writers** put the partial straight into this chip's slot of a persistent L1 scratch instead of the output tensor. They then signal one **fabric sender core per link** outside the expert grid.
- Each **sender** line-multicasts its cores' slots to every chip's matching slot, using the existing `all_reduce_async` `worker_writer`.
- The group-0 writer then waits on the global "out ready" semaphore. The group-0 compute sums the `ring_size` slots into the output buffer.

The op becomes a `WorkloadDescriptor` factory (one `ProgramDescriptor` per mesh coordinate) that owns the global semaphore and the scratch, following the pattern in `ccl/broadcast` and `all_gather_async_llama_sharded`.

**Tech Stack:** C++ (TT-Metal `ProgramDescriptor` / `WorkloadDescriptor`, tt-fabric 1D line multicast), Tensix dataflow and compute kernels (JIT-compiled), nanobind, Python (ttnn, pytest).

**Spec:** None separate. The design below is the spec. Sources it relies on:
- `ttnn/cpp/ttnn/operations/experimental/deepseek/width_sharded_all_reduce/device/width_sharded_all_reduce_program_factory.cpp` (the op being folded in)
- `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/fused_experts_program_factory.cpp` (writer descriptor at 1654-1739, semaphores at 817-861)
- `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/dataflow/write_gate_up.cpp:348-363` (group-0 output write)
- `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels/compute/matmul_gate_up.cpp:544-577` (group-0 cross-group reduce)
- `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:310-330` (`create_workload_descriptor` contract)
- `models/experimental/deepseek_v4_flash/tt/decode/moe.py:1003-1016, 1180-1201, 1472-1478` (Python caller)

## Global Constraints

- **Device commands need approval.** Every command that opens the device (any `pytest` below, tracy captures) must be shown to the user as a copy-pasteable command and run only after they approve. The card is shared.
- **Build only with** `cd /home/tt-admin/smanoj/metal_1 && ./build_metal.sh`. Kernel `.cpp` edits under `.../kernels/` are JIT-compiled and need no rebuild; host `.cpp`/`.hpp` and nanobind edits do.
- **No thin wrapper functions** (workspace rule): a helper must carry real logic, not argument plumbing.
- **Scope of the fused path:** one token row (`t == 1`), the 6-expert parallel path (12-wide grid, Blackhole), the output L1 WIDTH_SHARDED on `(0,0)-(1,7)` COL_MAJOR with shard `[1, H/16]`, a ROW_MAJOR input, and `ring_size > 1` on a FABRIC_1D mesh. Every other call keeps today's single-chip program, unchanged.
- **One axis only.** The op all-reduces over `all_reduce_cluster_axis`. TP12 on 3x4 (`_tp_cluster_axes == (1, 0)`) still runs `width_sharded_all_reduce` for axis 0 after the op.
- **Numerics:** the reduction order matches `width_sharded_all_reduce`: slots summed `0 .. ring_size-1` sequentially in DEST, packed to bf16. The fused compute kernel may run with `fp32_dest_acc_en`, so results agree with the two-op path to within bf16 rounding, not bit-for-bit. Every chip of the ring must produce bit-identical output, because all of them sum the same slots in the same order.
- `DEEPSEEK_V4_FUSE_ALL_REDUCE=0` must restore the two-op path, like the existing `DEEPSEEK_V4_FUSE_SHARED_EXPERT` / `DEEPSEEK_V4_FUSE_ROUTER` switches.

---

## Design

### Data flow on one chip of a ring of `R` chips (TP4: `R = 4`, `num_links = 2`)

```
group 1..5 down cores ──NoC──► group-0 down core (x∈{0,1}, y∈0..7)         [unchanged]
                                   │ compute: cross-group reduce -> cb_down_out
                                   ▼
                               writer: cb_down_out -> scratch[slot = my_chip]   (local, 8 faces * 64 B)
                                   │ sem_ar_ready.up(sender of my link)
                                   ▼
sender core (x = 12, y = link)  reader: wait sem_ar_ready == 8, read 8 cores' slot[my_chip] -> cb_ar_src
                                writer (all_reduce_async/worker_writer.cpp):
                                   fabric line-mcast slot[my_chip] -> same address on every chip,
                                   + local copy, + out_ready (GlobalSemaphore) += 1 on each core
                                   ▼
group-0 down core               writer: wait out_ready == R, push cb_ar_scratch (R * 8 faces), reset to 0
                                compute: out = Σ_{r=0}^{R-1} scratch[r]  -> cb_ar_out (aliased on the output buffer)
```

### Why each piece is shaped this way

- **No intermediate partial tensor.** The writer puts `cb_down_out` straight into the chip's own scratch slot. `worker_writer` already writes the local slot as well as the remote ones, so here that local write rewrites identical bytes in place and is harmless. The reader's "input address" is the chip's own slot, `scratch + my_chip * kLocalTiles * 64`.
- **Senders come from outside the op's cores.** They are taken from `worker_cores(sub_device)` minus the expert grid, the router row and the replica grid, scanning column `GRID_X` (x = 12) first. Those cores run no fused-experts kernel, so the stock `worker_writer.cpp` runs there unchanged. The reader is forked into `fused_experts/.../ar_sender_reader.cpp` only to add the "partials ready" wait. The standalone op needs no such wait because the op boundary orders it.
- **The receiver role lives in `write_gate_up.cpp`.** On group-0 down cores the writer is idle after its scratch write, and every RISC of those cores is already taken by fused-experts kernels, so `reduction_receiver.cpp` cannot be added as a separate kernel. The sum is appended to `matmul_gate_up.cpp`'s `is_root` branch for the same reason (TRISC is taken).
- **Workload-scoped resources.** `create_workload_descriptor` allocates the `GlobalSemaphore` (on the 16 output cores plus the senders) and the scratch tensor (`[R * 1, H/16]` per core, preferring L1_SMALL as `make_scratch_tensor` does), then calls `Synchronize` before building the per-coordinate programs. The program hash already includes every weight address, so each MoE layer gets its own cached workload and its own scratch and semaphore. A layer's scratch is therefore only ever re-entered on the next decode step, by which point every chip has long finished reading it.
- **Link split.** The output cores are ordered `corerange_to_cores(grid, row_wise=true)`, the order `width_sharded_all_reduce` uses, and chunked `ceil(16 / num_links)` per link. A core's link is its index in that order divided by the chunk size. The writer gets its link's sender NoC coordinate as a runtime argument.

### Risks to check while implementing

1. **The factory variant mixes a `create_descriptor` factory (`MultiCore`) with a `create_workload_descriptor` factory (`MultiCoreAllReduce`).** If `DescriptorMeshWorkloadAdapter` rejects the mix at compile time, give `MultiCore` a `create_workload_descriptor` that loops the coordinates and calls `build_program_descriptor(..., nullptr)`.
2. **L1_SMALL capacity.** There are 43 MoE layers. Each layer's scratch is `R * H/16 * 2` bytes per output core (TP4, H=4096: 2 KiB), plus one global semaphore. If `largest_free_block` in L1_SMALL runs out, the scratch falls back to L1, which fragments it. The demo already logs this, so watch the "CCL semaphores and width_sharded_all_reduce's scratch" accounting near `test_full_model_decode_demo.py:564`.
3. **Sender-core collisions.** Anything persistent on column 12 (another sub-device, a prefetcher ring) would collide with the senders. The selection uses the op's sub-device worker cores and asserts with a clear message.
4. **Trace capture.** The workload must be created (cache miss, which allocates and calls `Synchronize`) before `begin_trace_capture`. The model's warm-up run already guarantees this for the standalone all-reduce.

---

## File Structure

| File | Change |
|---|---|
| `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/fused_experts_device_operation_types.hpp` | add `FusedAllReduceParams`, `operation_attributes_t::all_reduce` |
| `.../fused_experts/device/fused_experts_device_operation.hpp` / `.cpp` | `MultiCoreAllReduce` factory, `select_program_factory`, validation, hash, `invoke` / `prim::fused_experts` args |
| `.../fused_experts/device/fused_experts_program_factory.cpp` | `create_descriptor` body moved into `build_program_descriptor(..., const AllReduceEpilogue*)`, AR epilogue (CBs, semaphores, sender kernels, extra writer/compute args), `MultiCoreAllReduce::create_workload_descriptor` |
| `.../fused_experts/device/kernels/dataflow/ar_sender_reader.cpp` | **new**: `worker_reader.cpp` + wait on `sem_ar_ready` |
| `.../fused_experts/device/kernels/dataflow/write_gate_up.cpp` | `FE_ALL_REDUCE`: write to scratch slot, signal sender, receiver wait |
| `.../fused_experts/device/kernels/compute/matmul_gate_up.cpp` | `FE_ALL_REDUCE`: sum the `R` slots into `cb_ar_out` |
| `.../fused_experts/fused_experts.hpp` / `.cpp` / `fused_experts_nanobind.cpp` | three new kwargs |
| `tests/ttnn/nightly/unit_tests/operations/experimental/test_fused_experts_all_reduce.py` | **new** op test |
| `models/experimental/deepseek_v4_flash/tt/decode/moe.py` | `FUSE_ALL_REDUCE`, pass the axis, drop that axis from the block's loop |

---

### Task 1: Failing op-level test for the fused all-reduce

**Files:**
- Create: `tests/ttnn/nightly/unit_tests/operations/experimental/test_fused_experts_all_reduce.py`

**Interfaces:**
- Consumes: `_expert_weights`, `_shared_expert_weights` from `test_fused_experts.py` (same directory).
- Produces: the acceptance test every later task runs, `test_fused_experts_all_reduce[...]`.

- [ ] **Step 1: Write the test**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``fused_experts`` with its TP all-reduce fused in, against the same op followed by
``width_sharded_all_reduce``.

Each chip of a 1xR line gets its own token row, so the per-chip partials differ and the sum is
non-trivial. The fused output must match the two-op path on every chip (within bf16 rounding of one
sum) and be bit-identical across the chips of the ring.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.experimental.test_fused_experts import (
    _expert_weights,
    _shared_expert_weights,
)

TILE = 32
DOWN_CORES = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7))})


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "hidden, intermediate",
    [(4096, 512), (5120, 576)],
    ids=("v4_tp4", "v41_tp4"),
)
@pytest.mark.parametrize("num_links", (1, 2), ids=lambda n: f"links{n}")
@pytest.mark.parametrize("iterations", (3,))
def test_fused_experts_all_reduce(mesh_device, hidden, intermediate, num_links, iterations):
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    ring = submesh.get_num_devices()
    grid = submesh.compute_with_storage_grid_size()
    grid_y = 8 if (intermediate // TILE) % 16 == 0 else 9
    if grid.x < 13 or grid.y < grid_y:
        pytest.skip(f"needs the 12x{grid_y} expert grid plus a free sender column, got {grid.x}x{grid.y}")

    torch.manual_seed(0)
    num_experts, top_k = 16, 6
    limit, scaling, eps = 7.0, 2.5, 1e-20
    ids = torch.arange(top_k, dtype=torch.int64).reshape(1, 1, 1, top_k)
    scores = torch.rand((1, 1, 1, num_experts), dtype=torch.bfloat16).float() + 0.5
    x = (torch.rand((ring, 1, 1, hidden), dtype=torch.bfloat16) - 0.5).float()

    _, _, gate_up_tt, down_tt = _expert_weights(submesh, hidden, intermediate, num_experts)
    _, _, shared_gate_up_tt, shared_down_tt = _shared_expert_weights(submesh, hidden, intermediate)

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )
    ids_tt = ttnn.from_torch(
        ids.to(torch.int32), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scores_tt = ttnn.from_torch(
        scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(DOWN_CORES, [1, hidden // 16], ttnn.ShardOrientation.COL_MAJOR),
    )

    def run(**all_reduce):
        return ttnn.experimental.deepseek.moe.fused_experts(
            x_tt,
            routing_indices=ids_tt,
            routing_scores=scores_tt,
            gate_up_weights=gate_up_tt,
            down_weights=down_tt,
            num_experts=top_k,
            intermediate_size=intermediate,
            swiglu_limit=limit,
            top_k=top_k,
            routed_scaling_factor=scaling,
            routing_eps=eps,
            shared_gate_up_weights=shared_gate_up_tt,
            shared_down_weights=shared_down_tt,
            memory_config=out_mem,
            **all_reduce,
        )

    def per_chip(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)).float().reshape(ring, hidden)

    partial = run()
    expected = per_chip(
        ttnn.experimental.deepseek.width_sharded_all_reduce(
            partial, cluster_axis=1, num_links=num_links, topology=ttnn.Topology.Linear
        )
    )
    assert torch.allclose(per_chip(partial).sum(0).expand(ring, -1), expected, rtol=2e-2, atol=2e-2)

    # Several back-to-back calls exercise the semaphore reset and the persistent scratch.
    for i in range(iterations):
        fused = run(all_reduce_cluster_axis=1, all_reduce_num_links=num_links, all_reduce_topology=ttnn.Topology.Linear)
        assert fused.memory_config() == out_mem
        got = per_chip(fused)
        for chip in range(1, ring):
            assert torch.equal(got[0], got[chip]), f"iteration {i}: chip {chip} differs from chip 0"
        passing, msg = comp_pcc(expected, got, pcc=0.9999)
        assert passing, f"iteration {i}: fused all-reduce vs two-op path: {msg}"
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)
```

- [ ] **Step 2: Ask the user to run it and confirm it fails**

Ask the user to run (this opens the device):

```
pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_fused_experts_all_reduce.py -x --tb=short -k "v4_tp4 and links2"
```

Expected: FAIL with `TypeError: ... unexpected keyword argument 'all_reduce_cluster_axis'`. The two-op `expected` and the `allclose` sanity check above the loop already pass.

- [ ] **Step 3: Commit**

```bash
git add tests/ttnn/nightly/unit_tests/operations/experimental/test_fused_experts_all_reduce.py
git commit -m "test(fused_experts): two-op reference for a fused TP all-reduce"
```

---

### Task 2: Move `create_descriptor` into a builder that can take an all-reduce epilogue (no behaviour change)

**Files:**
- Modify: `.../fused_experts/device/fused_experts_program_factory.cpp:119-140, 1799`
- Modify: `.../fused_experts/device/fused_experts_device_operation.hpp:82-90`

**Interfaces:**
- Produces (in the factory `.cpp`, namespace `fused_experts`):

```cpp
// Everything the all-reduce epilogue adds to one chip's program; built per mesh coordinate by
// MultiCoreAllReduce::create_workload_descriptor.
struct AllReduceEpilogue {
    uint32_t device_index;            // this chip's position on the ring (its scratch slot)
    uint32_t ring_size;
    uint32_t num_links;
    tt::tt_metal::Buffer* scratch;    // [ring_size * 1, H/16] per output core, L1 / L1_SMALL
    uint32_t out_ready_sem_addr;      // GlobalSemaphore on the output cores + senders
    ttnn::MeshCoordinate coord;
    std::optional<ttnn::MeshCoordinate> forward_coord;
    std::optional<ttnn::MeshCoordinate> backward_coord;
    uint32_t num_targets_forward;
    uint32_t num_targets_backward;
    std::vector<uint32_t> forward_mcast_args;   // get_forward_backward_line_mcast_configuration
    std::vector<uint32_t> backward_mcast_args;
};

ProgramDescriptor build_program_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const AllReduceEpilogue* all_reduce);
```

- [ ] **Step 1: Do the move**

Rename the body of `FusedExpertsDeviceOperation::MultiCore::create_descriptor` (lines 119-1800) to `build_program_descriptor` with the extra `const AllReduceEpilogue* all_reduce` parameter, which this task leaves unused (`(void)all_reduce;`). Then define:

```cpp
ProgramDescriptor FusedExpertsDeviceOperation::MultiCore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    return build_program_descriptor(operation_attributes, tensor_args, tensor_return_value, nullptr);
}
```

`create_descriptor` is the framework entry point, so this is not a thin wrapper in the rule's sense: the framework requires that exact signature.

- [ ] **Step 2: Build**

Run: `cd /home/tt-admin/smanoj/metal_1 && ./build_metal.sh`
Expected: builds cleanly.

- [ ] **Step 3: Ask the user to run the existing op tests (regression)**

```
pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_fused_experts.py -x --tb=short -k "shared_expert or replicated_input or ranking_scores_match_topk_ids"
```

Expected: the same pass/skip set as before the change.

- [ ] **Step 4: Commit**

```bash
git add ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/
git commit -m "refactor(fused_experts): build the program through build_program_descriptor"
```

---

### Task 3: API, attributes, validation and hash for the fused all-reduce

**Files:**
- Modify: `.../fused_experts/device/fused_experts_device_operation_types.hpp:17-67`
- Modify: `.../fused_experts/device/fused_experts_device_operation.hpp:82-153`
- Modify: `.../fused_experts/device/fused_experts_device_operation.cpp:16-18, 54-85 (validation), 582-653 (hash), 688-796 (invoke / prim)`
- Modify: `.../fused_experts/fused_experts.hpp`, `.../fused_experts/fused_experts.cpp`, `.../fused_experts/fused_experts_nanobind.cpp:130-151`

**Interfaces:**
- Produces:

```cpp
// fused_experts_device_operation_types.hpp
struct FusedAllReduceParams {
    uint32_t cluster_axis{};
    uint32_t ring_size{};   // resolved in invoke via ttnn::ccl::get_topological_dimension
    uint32_t num_links = 1;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear;
};
// in operation_attributes_t:
    std::optional<FusedAllReduceParams> all_reduce;
```

- Produces new trailing args on `invoke`, `ttnn::prim::fused_experts` and `ttnn::experimental::deepseek::moe::fused_experts`:

```cpp
    std::optional<uint32_t> all_reduce_cluster_axis = std::nullopt,
    uint32_t all_reduce_num_links = 1,
    tt::tt_fabric::Topology all_reduce_topology = tt::tt_fabric::Topology::Linear
```

- Produces Python kwargs `all_reduce_cluster_axis=None, all_reduce_num_links=1, all_reduce_topology=ttnn.Topology.Linear`.
- Produces `struct MultiCoreAllReduce` and `using program_factory_t = std::variant<MultiCore, MultiCoreAllReduce>;`. Its body is written in Task 4.

- [ ] **Step 1: Attributes and invoke**

In `invoke`, after `attributes` is built:

```cpp
    if (all_reduce_cluster_axis.has_value()) {
        attributes.all_reduce = FusedAllReduceParams{
            .cluster_axis = *all_reduce_cluster_axis,
            .ring_size = ttnn::ccl::get_topological_dimension(input_tensor, all_reduce_cluster_axis),
            .num_links = all_reduce_num_links,
            .topology = all_reduce_topology,
        };
    }
```

`select_program_factory` returns `MultiCoreAllReduce{}` when `attributes.all_reduce.has_value()`, otherwise `MultiCore{}`.

- [ ] **Step 2: Validation (in `validate_on_program_cache_miss`)**

```cpp
    if (attributes.all_reduce.has_value()) {
        const auto& ar = *attributes.all_reduce;
        const auto& mem = attributes.output_memory_config;
        const auto& input = tensor_args.input_tensor;
        const auto grid = input.device()->compute_with_storage_grid_size();
        const uint32_t hidden = static_cast<uint32_t>(input.logical_shape()[-1]);
        TT_FATAL(ar.ring_size > 1, "fused_experts: all_reduce needs more than one chip on axis {}", ar.cluster_axis);
        TT_FATAL(ar.num_links >= 1 && ar.num_links <= kDownCoresPerExpert,
                 "fused_experts: all_reduce num_links must be in [1, {}], got {}", kDownCoresPerExpert, ar.num_links);
        TT_FATAL(fused_experts_input_rows(input) == 1, "fused_experts: all_reduce runs on one token row only");
        TT_FATAL(input.layout() == Layout::ROW_MAJOR, "fused_experts: all_reduce needs a ROW_MAJOR input");
        TT_FATAL(
            mem.buffer_type() == BufferType::L1 && mem.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                mem.shard_spec().has_value() &&
                mem.shard_spec()->grid == CoreRangeSet(CoreRange({0, 0}, {1, kGridY - 1})) &&
                mem.shard_spec()->orientation == ShardOrientation::COL_MAJOR &&
                mem.shard_spec()->shape == std::array<uint32_t, 2>{1, hidden / kDownCoresPerExpert},
            "fused_experts: all_reduce needs the L1 WIDTH_SHARDED [1, H/16] output on (0,0)-(1,7), COL_MAJOR");
        // The 6-expert path (num_experts == 6 on a >= 12-wide grid) is the only one with that output.
        TT_FATAL(attributes.num_experts == kParallelExperts && grid.x >= 12,
                 "fused_experts: all_reduce needs the 6-expert path");
    }
```

`kGridY`, `kDownCoresPerExpert` and `kParallelExperts` are currently in the factory's anonymous namespace. Move them into `fused_experts_device_operation.hpp` (namespace `fused_experts`) so both files use the same constants.

- [ ] **Step 3: Hash**

Append to the `hash_operation<FusedExpertsDeviceOperation>(...)` argument list in `compute_program_hash`:

```cpp
        attributes.all_reduce.has_value(),
        attributes.all_reduce.has_value() ? attributes.all_reduce->cluster_axis : 0u,
        attributes.all_reduce.has_value() ? attributes.all_reduce->ring_size : 0u,
        attributes.all_reduce.has_value() ? attributes.all_reduce->num_links : 0u,
        attributes.all_reduce.has_value() ? static_cast<uint32_t>(attributes.all_reduce->topology) : 0u
```

- [ ] **Step 4: Nanobind**

After `nb::arg("router_bias") = std::nullopt`:

```cpp
        nb::arg("all_reduce_cluster_axis") = std::nullopt,
        nb::arg("all_reduce_num_links") = 1,
        nb::arg("all_reduce_topology") = tt::tt_fabric::Topology::Linear);
```

Document the three kwargs in the docstring next to `router_bias`. The text: "all-reduce the per-chip partial over this mesh axis inside the op; requires the one-row 6-expert path and the L1 WIDTH_SHARDED `[1, H/16]` output on (0,0)-(1,7). The result has the same spec, replicated over the axis."

- [ ] **Step 5: Build, then make `MultiCoreAllReduce` fail loudly until Task 4**

Declare `MultiCoreAllReduce` with the `create_workload_descriptor` signature. Its body for now: `TT_THROW("fused_experts: all_reduce is not implemented yet");`. This is the one deliberate stub, and Task 4 replaces it.

Run: `./build_metal.sh`. Expected: builds.

- [ ] **Step 6: Ask the user to rerun Task 1's test**

Same command as Task 1. Expected: FAIL with `fused_experts: all_reduce is not implemented yet`, which shows the kwargs reach the factory. Then commit:

```bash
git add ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/
git commit -m "feat(fused_experts): all_reduce kwargs, attributes, validation and hash"
```

---

### Task 4: The fused all-reduce (host epilogue + kernels)

**Files:**
- Modify: `.../fused_experts/device/fused_experts_program_factory.cpp` (epilogue inside `build_program_descriptor`, plus `MultiCoreAllReduce::create_workload_descriptor`)
- Create: `.../fused_experts/device/kernels/dataflow/ar_sender_reader.cpp`
- Modify: `.../fused_experts/device/kernels/dataflow/write_gate_up.cpp:116-139, 348-363`
- Modify: `.../fused_experts/device/kernels/compute/matmul_gate_up.cpp:139-146, 544-577`

**Interfaces:**
- Consumes: `AllReduceEpilogue`, `build_program_descriptor` (Task 2); `FusedAllReduceParams` (Task 3).
- Produces these constants (all new, in the factory):

```cpp
constexpr uint32_t sem_ar_ready_id = 6;      // senders: group-0 cores of the link bump it once their slot is written
constexpr uint32_t sem_ar_mcast_id = 7;      // senders: worker_writer's local reduction semaphore
constexpr uint32_t cb_ar_scratch = CBIndex::c_19;  // output cores + senders, aliased on the scratch
constexpr uint32_t cb_ar_out     = CBIndex::c_20;  // output cores, aliased on the output buffer
constexpr uint32_t cb_ar_src     = CBIndex::c_21;  // senders: reader -> fabric writer
constexpr uint32_t cb_ar_headers = CBIndex::c_22;  // senders: fabric packet headers
```

- [ ] **Step 1: `create_workload_descriptor`**

```cpp
tt::tt_metal::WorkloadDescriptor FusedExpertsDeviceOperation::MultiCoreAllReduce::create_workload_descriptor(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& ar = *attributes.all_reduce;
    auto* mesh_device = tensor_args.input_tensor.device();
    tt::tt_metal::WorkloadDescriptor workload;

    // The scratch: ring_size slots of the output shard on each output core. In L1_SMALL when it fits,
    // so like the semaphore it lives as long as the cached workload without fragmenting L1.
    const auto& shard = output.shard_spec().value();
    auto logical = output.logical_shape();
    logical[-2] *= ar.ring_size;
    const ShardSpec scratch_shard{shard.grid, {shard.shape[0] * ar.ring_size, shard.shape[1]}, shard.orientation};
    const auto* allocator = mesh_device->allocator().get();
    const uint64_t bytes_per_bank = tt::align(
        uint64_t{scratch_shard.shape[0]} * scratch_shard.shape[1] * output.element_size(),
        allocator->get_alignment(BufferType::L1_SMALL));
    const auto scratch_type = allocator->get_bank_size(BufferType::L1_SMALL) > 0 &&
                                      allocator->get_statistics(BufferType::L1_SMALL).largest_free_block_bytes >=
                                          bytes_per_bank
                                  ? BufferType::L1_SMALL
                                  : BufferType::L1;
    auto scratch = std::make_shared<Tensor>(create_device_tensor(
        TensorSpec(
            logical,
            TensorLayout(
                output.dtype(),
                PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, scratch_type, scratch_shard})),
        mesh_device));
    workload.buffers.push_back({scratch, scratch->buffer()});

    const auto sem_type = allocator->get_bank_size(BufferType::L1_SMALL) > 0 ? BufferType::L1_SMALL : BufferType::L1;
    const auto subdevice = mesh_device->get_sub_device_ids().at(0);
    workload.semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(
        mesh_device, mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, subdevice), 0, sem_type));
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, {subdevice});

    for (const auto& coord : tensor_coords.coords()) {
        const auto& input = tensor_args.input_tensor;
        const uint32_t device_index =
            ttnn::ccl::get_linearized_index_from_physical_coord(input, coord, ar.cluster_axis);
        auto forward = ttnn::ccl::get_physical_neighbor_from_physical_coord(input, coord, 1, ar.topology, ar.cluster_axis);
        auto backward = ttnn::ccl::get_physical_neighbor_from_physical_coord(input, coord, -1, ar.topology, ar.cluster_axis);
        auto [n_fwd, n_bwd] =
            ttnn::ccl::get_forward_backward_line_mcast_distance(ar.ring_size, device_index, ar.topology, true);
        auto [fwd_args, bwd_args] = ttnn::ccl::get_forward_backward_line_mcast_configuration(
            coord, forward, backward, n_fwd, n_bwd, mesh_device);
        const AllReduceEpilogue epilogue{
            .device_index = device_index,
            .ring_size = ar.ring_size,
            .num_links = ar.num_links,
            .scratch = scratch->buffer(),
            .out_ready_sem_addr = static_cast<uint32_t>(workload.semaphores[0].address()),
            .coord = coord,
            .forward_coord = forward,
            .backward_coord = backward,
            .num_targets_forward = n_fwd,
            .num_targets_backward = n_bwd,
            .forward_mcast_args = {fwd_args.begin(), fwd_args.end()},
            .backward_mcast_args = {bwd_args.begin(), bwd_args.end()},
        };
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord), build_program_descriptor(attributes, tensor_args, output, &epilogue)});
    }
    return workload;
}
```

Add the includes `width_sharded_all_reduce_program_factory.cpp` uses: `ttnn/operations/ccl/ccl_common.hpp`, `ttnn/global_semaphore.hpp`, `<tt-metalium/experimental/fabric/fabric.hpp>`, `<tt-metalium/tt_align.hpp>`, `<tt-metalium/workload_descriptor.hpp>`. In `fused_experts/CMakeLists.txt`, link whatever target `width_sharded_all_reduce/CMakeLists.txt` links for CCL (copy its `target_link_libraries` line).

- [ ] **Step 2: Epilogue in `build_program_descriptor`**

Insert just before `return desc;` (current line 1799). Guard it with `if (all_reduce != nullptr)`, and in the same `if` also:

- Add `"FE_ALL_REDUCE"` to `compute_desc.defines` and to each writer's `defines`. Kernel descriptors are pushed earlier, so either set a `const bool fuse_ar = all_reduce != nullptr;` before the compute and writer descriptors are built and add the define there, or patch `desc.kernels` entries by `kernel_source`. Prefer the former.
- Append three compute CT args after `compute_ct_args`' last entry (index 29 in the kernel): `cb_ar_scratch, cb_ar_out, all_reduce->ring_size`.
- Append three writer CT args after `TensorAccessorArgs(*input_buffer).append_to(writer_ct_args)`: `sem_ar_ready_id, cb_ar_scratch, all_reduce->ring_size`.
- Append five writer runtime args after `input_buffer` in `make_writer`: `all_reduce->scratch` (as a `Buffer*` binding), `all_reduce->device_index`, the link's sender NoC x, the link's sender NoC y, `all_reduce->out_ready_sem_addr`. Pass zeros on cores that are not group-0 down cores.

Then the epilogue itself:

```cpp
    if (all_reduce != nullptr) {
        const auto face = tt::tt_metal::Tile({1, TILE_DIM});
        const auto out_df = datatype_to_dataformat_converter(output_tensor.dtype());
        const uint32_t face_bytes = face.get_tile_size(out_df);
        const CoreRangeSet out_cores = operation_attributes.output_memory_config.shard_spec()->grid;
        const auto out_cores_vec = corerange_to_cores(out_cores, std::nullopt, /*row_wise=*/true);
        const uint32_t shard_pages = (operation_attributes.output_memory_config.shard_spec()->shape[1]) / TILE_DIM;
        const uint32_t cores_per_link = tt::div_up(static_cast<uint32_t>(out_cores_vec.size()), all_reduce->num_links);

        // Senders: the first num_links worker cores outside everything this op runs on, column GRID_X first.
        const auto sub_device_cores = device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().at(0));
        const CoreRangeSet taken = run_router ? all_cores.merge(router_cores) : all_cores;
        std::vector<CoreCoord> senders;
        for (uint32_t x = GRID_X; x < grid.x && senders.size() < all_reduce->num_links; ++x) {
            for (uint32_t y = 0; y < grid.y && senders.size() < all_reduce->num_links; ++y) {
                const CoreCoord c{x, y};
                if (sub_device_cores.contains(c) && !taken.contains(c)) {
                    senders.push_back(c);
                }
            }
        }
        TT_FATAL(
            senders.size() == all_reduce->num_links,
            "fused_experts: all_reduce needs {} free sender cores right of the {}x{} expert grid, found {}",
            all_reduce->num_links, GRID_X, GRID_Y, senders.size());
        CoreRangeSet sender_set;
        for (const auto& c : senders) {
            sender_set = sender_set.merge(CoreRangeSet(CoreRange(c, c)));
        }

        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = sem_ar_ready_id, .core_type = CoreType::WORKER, .core_ranges = sender_set, .initial_value = 0});
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = sem_ar_mcast_id, .core_type = CoreType::WORKER, .core_ranges = sender_set, .initial_value = 0});

        // worker_writer reads the scratch base from its own cb_ar_scratch, so the senders declare it too.
        desc.cbs.push_back(CBDescriptor{
            .total_size = all_reduce->ring_size * shard_pages * face_bytes,
            .core_ranges = out_cores.merge(sender_set),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_ar_scratch, .data_format = out_df, .page_size = face_bytes, .tile = TileDescriptor(face)}}},
            .buffer = all_reduce->scratch,
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = shard_pages * face_bytes,
            .core_ranges = out_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_ar_out, .data_format = out_df, .page_size = face_bytes, .tile = TileDescriptor(face)}}},
            .buffer = out_buffer,
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = cores_per_link * shard_pages * face_bytes,
            .core_ranges = sender_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_ar_src, .data_format = out_df, .page_size = face_bytes, .tile = TileDescriptor(face)}}},
        });
        constexpr uint32_t kPacketHeaders = 8;
        const uint32_t header_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        desc.cbs.push_back(CBDescriptor{
            .total_size = kPacketHeaders * header_bytes * 2,
            .core_ranges = sender_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_ar_headers, .data_format = tt::DataFormat::RawUInt32, .page_size = header_bytes}}},
        });

        const uint32_t slot_offset = all_reduce->device_index * shard_pages * face_bytes;
        KernelDescriptor reader;
        reader.kernel_source = std::string(kKernelDir) + "/dataflow/ar_sender_reader.cpp";
        reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader.core_ranges = sender_set;
        reader.compile_time_args = {cb_ar_src, face_bytes, sem_ar_ready_id};
        reader.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};

        std::vector<uint32_t> writer_ct = {
            all_reduce->device_index, cb_ar_headers, kPacketHeaders, cb_ar_src,
            std::max<uint32_t>(1, tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*device) / face_bytes),
            face_bytes, all_reduce->num_targets_forward, all_reduce->num_targets_backward};
        writer_ct.insert(writer_ct.end(), all_reduce->forward_mcast_args.begin(), all_reduce->forward_mcast_args.end());
        writer_ct.insert(writer_ct.end(), all_reduce->backward_mcast_args.begin(), all_reduce->backward_mcast_args.end());
        KernelDescriptor fabric_writer;
        fabric_writer.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_writer.cpp";
        fabric_writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        fabric_writer.core_ranges = sender_set;
        fabric_writer.compile_time_args = writer_ct;
        fabric_writer.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
        fabric_writer.common_runtime_args = {all_reduce->out_ready_sem_addr};

        for (uint32_t link = 0; link < all_reduce->num_links; ++link) {
            const uint32_t first = link * cores_per_link;
            const uint32_t count = std::min<uint32_t>(cores_per_link, out_cores_vec.size() - first);
            std::vector<uint32_t> xs, ys;
            for (uint32_t i = first; i < first + count; ++i) {
                const auto noc = device->worker_core_from_logical_core(out_cores_vec[i]);
                xs.push_back(noc.x);
                ys.push_back(noc.y);
            }
            const CoreCoord s = senders[link];
            const auto s_noc = device->worker_core_from_logical_core(s);

            // Reader: wait for `count` slots, then read them; the address is this chip's slot of the scratch.
            std::vector<uint32_t> r_args = {all_reduce->scratch->address() + slot_offset, shard_pages, count * shard_pages, 0u, count};
            r_args.insert(r_args.end(), xs.begin(), xs.end());
            r_args.insert(r_args.end(), ys.begin(), ys.end());
            reader.runtime_args.emplace_back(s, KernelDescriptor::CoreRuntimeArgs(r_args.begin(), r_args.end()));

            // Fabric writer: worker_writer's runtime-arg layout (width_sharded_all_reduce_program_factory.cpp:381-422).
            // worker_writer reads the mcast range list but never multicasts on it, so none is passed
            // (num_mcast_cores = num_mcast_ranges = 0).
            std::vector<uint32_t> w_args = {
                cb_ar_scratch, shard_pages, count * shard_pages, 0u, count, 0u,
                static_cast<uint32_t>(s_noc.x), static_cast<uint32_t>(s_noc.y),
                all_reduce->ring_size, sem_ar_mcast_id, 0u, link};
            w_args.insert(w_args.end(), xs.begin(), xs.end());
            w_args.insert(w_args.end(), ys.begin(), ys.end());
            w_args.push_back(all_reduce->forward_coord.has_value());
            if (all_reduce->forward_coord.has_value()) {
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    device->get_fabric_node_id(all_reduce->coord), device->get_fabric_node_id(*all_reduce->forward_coord),
                    link, desc, s, w_args);
            }
            w_args.push_back(all_reduce->backward_coord.has_value());
            if (all_reduce->backward_coord.has_value()) {
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    device->get_fabric_node_id(all_reduce->coord), device->get_fabric_node_id(*all_reduce->backward_coord),
                    link, desc, s, w_args);
            }
            fabric_writer.runtime_args.emplace_back(s, KernelDescriptor::CoreRuntimeArgs(w_args.begin(), w_args.end()));
        }
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(fabric_writer));
    }
```

The writer runtime arguments in `make_writer` need each output core's sender. Compute `link_of(core) = index_of(core in out_cores_vec) / cores_per_link` and select `senders[link_of(core)]`. That means the sender selection and `out_cores_vec` / `cores_per_link` must be computed **before** `make_writer` runs: hoist them out of the `if` above to just after `writer_ct_args` is built, still guarded by `all_reduce != nullptr`.

The reader's `scratch->address()` and `out_ready_sem_addr` are raw addresses. The scratch and semaphore belong to the workload and never move, so they need no binding. The scratch passed to the group-0 writers goes in as the `Buffer*` binding so it is listed in `workload.buffers`.

- [ ] **Step 3: `ar_sender_reader.cpp`**

```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include <cstdint>

// The fused all-reduce's reader on one fabric sender core: worker_reader.cpp (all_reduce_async), except
// that the slots it forwards are written by this same program, so it first waits for the link's
// group-0 down cores to bump sem_ar_ready once each.
void kernel_main() {
    constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t sem_ar_ready_id = get_compile_time_arg_val(2);

    size_t arg_idx = 0;
    const uint32_t slot_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pages_per_core = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pages_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t first_core_page_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));

    Semaphore<>(sem_ar_ready_id).wait(num_cores);
    Semaphore<>(sem_ar_ready_id).set(0);

    Noc noc;
    CircularBuffer cb0(cb0_id);
    UnicastEndpoint src_ep;
    uint32_t pages_read = 0;
    uint32_t page = first_core_page_offset;
    for (uint32_t core = 0; pages_read < pages_to_read; ++core) {
        const uint32_t n = std::min(pages_per_core - page, pages_to_read - pages_read);
        cb0.reserve_back(n);
        noc.async_read(
            src_ep, cb0, n * page_size,
            {.noc_x = core_noc_x[core], .noc_y = core_noc_y[core], .addr = slot_addr + page * page_size}, {});
        noc.async_read_barrier();
        cb0.push_back(n);
        pages_read += n;
        page = 0;
    }
}
```

- [ ] **Step 4: `write_gate_up.cpp`**

After `constexpr auto input_args = TensorAccessorArgs<kSharedArg + 10>();` (line 127):

```cpp
#ifdef FE_ALL_REDUCE
    constexpr uint32_t kArArg = input_args.next_compile_time_args_offset();
    constexpr uint32_t sem_ar_ready_id = get_compile_time_arg_val(kArArg);
    constexpr uint32_t cb_ar_scratch_id = get_compile_time_arg_val(kArArg + 1);
    constexpr uint32_t ar_ring_size = get_compile_time_arg_val(kArArg + 2);
    const uint32_t ar_scratch_addr = get_arg_val<uint32_t>(10);
    const uint32_t ar_slot = get_arg_val<uint32_t>(11);
    const uint32_t ar_sender_noc_x = get_arg_val<uint32_t>(12);
    const uint32_t ar_sender_noc_y = get_arg_val<uint32_t>(13);
    const uint32_t ar_out_ready_addr = get_arg_val<uint32_t>(14);
#endif
```

Replace the group-0 branch at lines 350-351:

```cpp
        if (expert_group == 0) {
#ifdef FE_ALL_REDUCE
            // This chip's partial goes to its own slot of the all-reduce scratch (reduce_noc_* is this
            // core on group 0); the link's sender forwards it to every chip from there.
            noc.async_write(
                cb_down_out,
                UnicastEndpoint{},
                kLocalTiles * out_tile_bytes,
                {.offset_bytes = 0},
                {.noc_x = reduce_noc_x, .noc_y = reduce_noc_y, .addr = ar_scratch_addr + ar_slot * kLocalTiles * out_tile_bytes});
            noc.async_write_barrier();
            Semaphore<>(sem_ar_ready_id).up(noc, ar_sender_noc_x, ar_sender_noc_y, 1);
#else
            write_output(kLocalTiles);
#endif
        } else {
```

After `cb_down_out.pop_front(kLocalTiles);` (line 363), still inside `if constexpr (num_expert_groups > 1)`:

```cpp
#ifdef FE_ALL_REDUCE
        if (expert_group == 0) {
            // Every chip's slot (this one's included, by the sender's local copy) bumps out_ready once.
            auto* out_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ar_out_ready_addr);
            noc_semaphore_wait(out_ready, ar_ring_size);
            CircularBuffer(cb_ar_scratch_id).push_back(ar_ring_size * kLocalTiles);
            noc_semaphore_set(out_ready, 0);
        }
#endif
```

`out_tile_bytes` is the 64-byte 1x32 bf16 face here, the same page `worker_writer` uses, and `kLocalTiles == shard_pages`. Add `static_assert`s where both are known:

```cpp
#ifdef FE_ALL_REDUCE
    static_assert(out_tile_bytes == 64, "fused all-reduce slots are 1x32 bf16 faces");
#endif
```

- [ ] **Step 5: `matmul_gate_up.cpp`**

After line 139 (`shared_first_local`):

```cpp
#ifdef FE_ALL_REDUCE
    constexpr uint32_t cb_ar_scratch_id = get_compile_time_arg_val(30);
    constexpr uint32_t cb_ar_out_id = get_compile_time_arg_val(31);
    constexpr uint32_t ar_ring_size = get_compile_time_arg_val(32);
#endif
```

After `reduce_cb.pop_front(kRemoteGroups * kLocalTiles);` (line 576), inside `if (is_root)`:

```cpp
#ifdef FE_ALL_REDUCE
            // Sum the ring's slots in chip order, as width_sharded_all_reduce does, so every chip packs
            // the same bits.
            CircularBuffer ar_in(cb_ar_scratch_id);
            CircularBuffer ar_out(cb_ar_out_id);
            ar_in.wait_front(ar_ring_size * kLocalTiles);
            ar_out.reserve_back(kLocalTiles);
            reconfig_full_operand(cb_ar_scratch_id, cb_ar_scratch_id);
            pack_reconfig_data_format(cb_ar_out_id);
            for (uint32_t t = 0; t < kLocalTiles; ++t) {
                copy_init(cb_ar_scratch_id);
                tile_regs_acquire();
                copy_tile(cb_ar_scratch_id, t, 0);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_ar_scratch_id);
                for (uint32_t r = 1; r < ar_ring_size; ++r) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_ar_scratch_id, r * kLocalTiles + t, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_ar_out_id, t);
                tile_regs_release();
            }
            ar_out.push_back(kLocalTiles);
            ar_in.pop_front(ar_ring_size * kLocalTiles);
#endif
```

`reconfig_full_operand` is this kernel's existing helper (used at lines 216-481). If its two-operand form insists on distinct CBs, use the `_srca` variant plus `reconfig_data_format_srcb(cb_ar_scratch_id)`. `cb_ar_scratch` is aliased on the persistent scratch, so popping it only advances this program's read pointer.

- [ ] **Step 6: Build**

Run: `./build_metal.sh`. Expected: builds. The kernels only compile when the test runs.

- [ ] **Step 7: Ask the user to run Task 1's test, all parameterizations**

```
pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_fused_experts_all_reduce.py -x --tb=short
```

Expected: PASS for `v4_tp4` and `v41_tp4` with `links1` and `links2`, all 3 iterations.

If it hangs, ask the user to rerun with `TT_METAL_WATCHER=1` and read which core is stuck:
- A group-0 writer stuck in `noc_semaphore_wait` means the senders never forwarded.
- A sender stuck in `Semaphore::wait` means a group-0 writer had the wrong sender coordinate.

- [ ] **Step 8: Ask the user to rerun the single-chip regression from Task 2, Step 3**

Expected: unchanged.

- [ ] **Step 9: Commit**

```bash
git add ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/
git commit -m "feat(fused_experts): all-reduce the TP partial over one mesh axis in the op"
```

---

### Task 5: Use it from the model

**Files:**
- Modify: `models/experimental/deepseek_v4_flash/tt/decode/moe.py:633-641, 1003-1016, 1180-1201, 1472-1478`

**Interfaces:**
- Consumes: the `fused_experts` kwargs from Task 3.
- Produces: `FUSE_ALL_REDUCE` (module flag), `DeepSeekV4PreloadedExperts.tp_axes: tuple[int, ...]`, and `DeepSeekV4PreloadedExperts.fused_all_reduce_axis: Optional[int]`.

- [ ] **Step 1: Flag**

After `FUSE_ROUTER` (line 638):

```python
# All-reduce the TP partial over the first TP axis inside ``fused_experts`` wherever its output is the
# width-sharded partial (see ``output_memory_config``). ``DEEPSEEK_V4_FUSE_ALL_REDUCE=0`` leaves it to
# ``width_sharded_all_reduce``.
FUSE_ALL_REDUCE = os.environ.get("DEEPSEEK_V4_FUSE_ALL_REDUCE", "1") not in ("0", "", "false", "False")
```

- [ ] **Step 2: Experts**

After the `output_memory_config` block (after line 1016):

```python
        # The mesh axes the TP partials are summed over, in order; the op takes the first itself when it can.
        self.tp_axes = _tp_cluster_axes(device, tp_size) if tp_size > 1 else ()
        self.fused_all_reduce_axis = (
            self.tp_axes[0] if self.output_memory_config is not None and FUSE_ALL_REDUCE else None
        )
```

In `_run_fused`'s `fused_experts(...)` call, after `memory_config=...` (line 1200):

```python
                all_reduce_cluster_axis=self.fused_all_reduce_axis if t == 1 else None,
                all_reduce_num_links=2,
                all_reduce_topology=CCL_TOPOLOGY,
```

- [ ] **Step 3: Block**

In `DeepSeekV4SparseMoeBlock.decode_static`, replace:

```python
        if t == 1 and self.experts.output_memory_config is not None:
            for axis in _tp_cluster_axes(self.device, self.tp_size):
```

with:

```python
        if t == 1 and self.experts.output_memory_config is not None:
            done = 1 if self.experts.fused_all_reduce_axis is not None else 0
            for axis in self.experts.tp_axes[done:]:
```

- [ ] **Step 4: Ask the user to run the model-level TP tests**

```
pytest models/experimental/deepseek_v4_flash/tests/decode/test_moe_pcc_tp.py -x --tb=short
pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe.py -x --tb=short -k "test_moe_decode_tp"
DEEPSEEK_V4_FUSE_ALL_REDUCE=0 pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe.py -x --tb=short -k "test_moe_decode_tp"
```

Expected:
- All pass, with the same PCC as before.
- TP12 exercises the remaining axis-0 `width_sharded_all_reduce`.
- The `=0` run confirms the old path still works.
- `test_moe_decode_tp` logs "max abs spread over the chips", which must stay 0.

- [ ] **Step 5: Commit**

```bash
git add models/experimental/deepseek_v4_flash/tt/decode/moe.py
git commit -m "feat(deepseek_v4): fuse the MoE TP all-reduce into fused_experts"
```

---

### Task 6: Trace, full-model check and perf

**Files:**
- None, unless the numbers call for a change.

- [ ] **Step 1: Ask the user to run the traced full-model demo at TP4 and compare tokens with the flag on and off**

```
pytest models/experimental/deepseek_v41_flash/tests/decode/test_full_model_decode_demo.py -x --tb=short
DEEPSEEK_V4_FUSE_ALL_REDUCE=0 pytest models/experimental/deepseek_v41_flash/tests/decode/test_full_model_decode_demo.py -x --tb=short
```

Expected:
- Both complete, which shows trace capture works with the workload-owned semaphore and scratch.
- The greedy token streams match, or diverge only late (bf16 rounding of the sum).
- Throughput (tok/s/u) with the flag on is at least as high as with it off.
- Check the L1 / L1_SMALL usage the demo logs around `test_full_model_decode_demo.py:564`. There must be no new "fell back to L1" scratch and no CB clash.

- [ ] **Step 2: Ask the user for a tracy capture of the MoE TP region with the flag on and off**

```
python -m tracy -r -p -v -m pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe.py -k "test_moe_decode_tp and tp4"
DEEPSEEK_V4_FUSE_ALL_REDUCE=0 python -m tracy -r -p -v -m pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe.py -k "test_moe_decode_tp and tp4"
```

Compare the `MOE_TP4_START` .. `MOE_TP4_END` span: the fused op's device time against `fused_experts` + the op-to-op gap + `width_sharded_all_reduce`. Record the numbers in the commit message of any follow-up change.

- [ ] **Step 3: Follow-ups, recorded here and not done in this plan**

- Fuse the second TP axis as well (TP12: reduce over axis 1, then forward the reduced shard over axis 0 from the same senders, using a second scratch and semaphore).
- Let each sender start as soon as its own link's 8 cores are ready. It already does, so measure whether `num_links = 4` (4 cores per link) shortens the tail.

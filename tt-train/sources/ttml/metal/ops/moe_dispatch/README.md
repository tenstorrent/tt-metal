# MoE Streaming Dispatch Kernel

## What it does

Single op that dispatches sorted token tile-rows across a `[1, 32]` galaxy mesh
via fabric unicast and computes `matmul(tokens, W_up)` on each device's local
experts — all in **one program launch** across all 32 devices.

```
Python:  result = ttml.ops.moe.dispatch(sorted_hidden, w_up, cluster_axis, offsets, counts, E_local)
```

## Architecture

One program per device with **4 kernels on 2 cores** running concurrently:

```
Core (0,0) — Sender                   Core (0,1) — Receiver + Compute
┌──────────────────────────┐          ┌──────────────────────────────────┐
│ RISCV_0 (writer/NOC_0)   │          │ RISCV_0 (reader)                 │
│                          │          │  • noc_semaphore_wait(barrier)    │
│ for each expert e:       │          │  • read dispatched tiles from     │
│   dest = e / E_local     │          │    output buffer (DRAM)           │
│   if dest == me:         │          │  • read W_up tiles from DRAM      │
│     noc_async_write      │          │  • push to compute CBs            │
│     (local DRAM→DRAM)    │          │                                   │
│   else:                  │          │ TENSIX (compute)                   │
│     fabric_send_unicast  │          │  • matmul_block + L1 acc          │
│     (→ remote output buf)│          │  • same proven kernel (PCC 0.999) │
│                          │          │                                   │
│ noc_semaphore_inc(barrier)│         │ RISCV_1 (writer)                  │
│                          │          │  • write matmul output to DRAM    │
└──────────────────────────┘          └──────────────────────────────────┘
```

## How the dispatch works (no sockets)

Follows the exact same pattern as `ttnn.all_to_all_dispatch`:

1. **Fabric connections**: `WorkerToFabricEdmSender` opened per neighbor direction.
   Routing info baked in as compile-time `#define DEST_CHIP_ID`, `DEST_MESH_ID`,
   `DIRECTIONS` — the kernel knows at compile time which fabric links to use.

2. **Unicast writes**: For each expert's token chunk, the sender reads tiles from
   local DRAM and either:
   - **Local expert**: `noc_async_write` to the output buffer on the same device
   - **Remote expert**: `fabric_send_chip_unicast_noc_unicast_1d` through the
     fabric to the destination device's output buffer

3. **Global semaphore barrier**: After all tokens dispatched, each device
   increments a `GlobalSemaphore`. The receiver reader does
   `noc_semaphore_wait(barrier, num_devices)` — blocks until ALL 32 devices
   have finished sending. Only then does it start reading from the output buffer.

4. **Fused matmul**: The receiver reader feeds received tiles + weight tiles to
   the compute engine in the matmul block order. The compute kernel is identical
   to the proven single-device matmul (PCC 0.9999).

## Data flow

```
Device 0                    Device 1                    Device 31
┌─────────┐                ┌─────────┐                ┌─────────┐
│sorted    │──fabric──────>│output   │                │         │
│hidden    │──fabric──────────────────────────>──────>│output   │
│[N,D]    │                │buffer   │                │buffer   │
└─────────┘                └────┬────┘                └────┬────┘
                                │                          │
                           semaphore                   semaphore
                           barrier                     barrier
                                │                          │
                           ┌────▼────┐                ┌────▼────┐
                           │ matmul  │                │ matmul  │
                           │ W_up[1] │                │ W_up[31]│
                           └────┬────┘                └────┬────┘
                                │                          │
                           [N_e1, ffn]              [N_e31, ffn]
```

## Files

```
moe_dispatch/
├── moe_dispatch.hpp                           # Public C++ API
├── moe_dispatch.cpp                           # Delegates to ttnn::prim
├── README.md                                  # This file
└── device/
    ├── moe_dispatch_types.hpp                 # MoeDispatchParams, MoeDispatchTensorArgs
    ├── moe_dispatch_device_operation.hpp       # DeviceOperation struct (MeshWorkload variant)
    ├── moe_dispatch_device_operation.cpp       # validate, compute_output_specs, create_output, hash
    ├── moe_dispatch_program_factory.hpp        # MeshWorkloadFactory with create_mesh_workload
    ├── moe_dispatch_program_factory.cpp        # Per-device program creation:
    │                                           #   - GlobalSemaphore for cross-device barrier
    │                                           #   - fabric connections via append_fabric_connection_rt_args
    │                                           #   - DEST_CHIP_ID/DEST_MESH_ID/DIRECTIONS as #defines
    │                                           #   - CBs: sender (data + packet header), receiver (in, w, out)
    │                                           #   - 4 kernels: sender, receiver_reader, compute, writer
    └── kernels/
        ├── dataflow/
        │   ├── sender.cpp                     # Reads sorted tiles from DRAM, dispatches via fabric
        │   │                                  # unicast (local: noc_async_write, remote:
        │   │                                  # fabric_send_chip_unicast_noc_unicast_1d).
        │   │                                  # Signals GlobalSemaphore when done.
        │   ├── receiver_reader.cpp            # noc_semaphore_wait(barrier, num_devices) then
        │   │                                  # reads dispatched tokens from output buffer +
        │   │                                  # streams W_up weight tiles to compute CB.
        │   └── receiver_writer.cpp            # Writes matmul output tiles to DRAM.
        └── compute/
            └── expert_matmul.cpp              # matmul_block + pack_l1_acc_block per tile-row.
                                               # Same kernel that achieved PCC 0.9999 on single device.
```

## Integration

### C++ build (CMakeLists.txt)
```cmake
${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/moe_dispatch/moe_dispatch.cpp
${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/moe_dispatch/device/moe_dispatch_device_operation.cpp
${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/moe_dispatch/device/moe_dispatch_program_factory.cpp
```

### Nanobind (nb_ops.cpp)
```python
ttml.ops.moe.dispatch(sorted_hidden, w_up, cluster_axis, expert_offsets, expert_counts, E_local)
```

### operations.hpp
```cpp
#include "ops/moe_dispatch/moe_dispatch.hpp"
```

## Key design decisions

1. **No sockets** — fabric unicast + global semaphore, same as `all_to_all_dispatch`.
   Sockets are for the `send_async`/`recv_async` pattern which is a different abstraction.

2. **Two-phase execution** within one program:
   - Phase 1 (sender core): fabric dispatch. All 32 devices send simultaneously.
   - Barrier: `noc_semaphore_wait` on receiver core blocks until all senders done.
   - Phase 2 (receiver cores): matmul on received tokens. All 32 compute simultaneously.

3. **MeshWorkload pattern** — `create_mesh_workload` iterates over `tensor_coords`,
   calls `create_at` per device. The `device_operation::launch` framework handles
   enqueueing the MeshWorkload. No manual `EnqueueMeshWorkload` call.

4. **Sorted tokens** — the sender already knows which expert each tile-row belongs to
   (from Phase 1 counting sort). No expert_mapping scan or routing lookup at runtime.
   Just iterate over expert offsets/counts and unicast each chunk to `dest = e / E_local`.

5. **Output buffer as dispatch target** — senders write directly into the output tensor's
   DRAM on the destination device. After the barrier, the receiver reader reads from
   this same buffer. No intermediate staging tensor.

## Comparison with all_to_all_dispatch

| Aspect | all_to_all_dispatch | moe_dispatch |
|---|---|---|
| Token routing | Per-token expert lookup at runtime | Pre-sorted, contiguous per expert |
| Output shape | `[EP, B, S, D]` padded (junk rows) | `[N_padded, ffn]` (only real tokens) |
| Compute fusion | None (separate matmul after) | Fused matmul in same program |
| Fabric pattern | Same (unicast + global semaphore) | Same |
| Program launches | 1 (dispatch) + E*3 (matmul+silu+matmul) | 1 (dispatch + matmul fused) |

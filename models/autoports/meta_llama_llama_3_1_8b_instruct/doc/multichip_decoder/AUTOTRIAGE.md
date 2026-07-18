# AUTOTRIAGE

## Diagnosis

- The TP4 fused `all_gather_matmul_async` program deadlocks because its generic program factory hardcodes `num_transfers = 4`, while `MatmulOpReceiver` interprets that as four transfers in each of two directions (eight tensor slices). The TP4 QKV matmul has only four gathered tensor slices. Both fused matmul data-movement kernels therefore fail their on-device slice/block ledger assertion (`8 slices * 4 K-blocks/slice == 16 K-blocks`, which is false) during receiver construction. With watcher/assert reporting disabled, the halted worker program surfaces as an indefinite host `synchronize_device`/mesh-command-queue wait.

## Triage Evidence

- A bounded reproduction on 2026-07-18 used:

  ```text
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  RUN_MULTICHIP_DECODER_TOPOLOGY_PROBE=1 \
  timeout 600 pytest -q -s --tb=short \
    models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
    -k fractured_residual_topology_probe
  ```

- The reproduction reached device/fabric initialization and the fused op's direct-matmul setup warnings, then made no further progress. The earlier separate all-gather plus matmul boundary had already completed on the same 1x4 ring with the same activation, persistent all-gather output, CCL semaphores, links, workers, buffers, and chunking policy. This localizes the failure to fused AG/MM coordination rather than reduce-scatter, distributed RMSNorm, the ring, or the standalone all-gather.
- Requester-provided GDB evidence placed a worker host thread in `FDMeshCommandQueue::finish_nolock` through `ttnn.synchronize_device`, while another host thread busy-polled. Those are host-side completion waiters after dispatch; they do not identify the first device-side stuck point.
- Focused live `tt-triage` was attempted before killing the repro:

  ```text
  timeout 180 tools/tt-triage.py --llm-output \
    --run=dump_callstacks --run=dump_running_operations \
    --run=check_eth_status --run=check_arc --dev=all
  ```

- `check_arc` directly read all four devices and reported live ARC heartbeats of approximately 9.997/s on devices 0-3. This refutes board disappearance and an ARC-service stall as the primary failure.
- Worker call stacks and running-operation aggregation could not be read: the installed triage/ttexalens reader called a `noc_read(..., memoryview)` signature incompatible with the loaded UMD binding. Ethernet status was skipped for the same ABI error. Inspector program metadata was also unavailable because `generated/inspector` is root-owned and the runtime could not initialize its logger. Consequently, this report does **not** claim a directly observed RISC-V PC, core, semaphore value, or watcher assertion.
- After evidence capture, only the reproduction PIDs were terminated. `timeout 180 tt-smi -r` reset PCI devices 0-3, `tt-smi -ls --local` showed all four p300c devices, and a `MeshShape(1, 4)` open/close smoke printed `MESH_SMOKE_OK`.

## Source Evidence

- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp:55-76` builds the matmul/all-gather fusion ledger. It hardcodes:

  ```cpp
  const uint32_t num_transfers = 4;
  ...
  matmul_fused_op_signaler->init_all_gather(
      num_transfers,
      ring_size,
      ...);
  ```

  `ring_size` is available but is not used to derive `num_transfers`.
- `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp:144-204` fixes `MatmulOpReceiver::num_directions` at two, computes `num_tensor_slices = num_transfers * num_directions`, computes `num_blocks_per_slice = tensor_slice_shape_width / tiles_per_block`, and asserts:

  ```cpp
  ASSERT(num_tensor_slices * num_blocks_per_slice == num_blocks);
  ```

- The TP4 QKV ledger is exact:

  | Quantity | Value | Source/derivation |
  |---|---:|---|
  | ring size | 4 | target 1x4 mesh |
  | global gathered K | 4096 elements = 128 tiles | four local 1024-element shards |
  | local slice width | 1024 elements = 32 tiles | `tensor_slicer.num_cols` |
  | matmul `in0_block_w` | 8 tiles | probe's 1D matmul program config |
  | matmul K-blocks | `128 / 8 = 16` | `matmul_multicore_reuse_mcast_1d_program_factory.cpp:2942` |
  | K-blocks per slice | `32 / 8 = 4` | `MatmulOpReceiver` |
  | receiver slices | `4 transfers * 2 directions = 8` | hardcoded factory value |
  | asserted equality | `8 * 4 == 16` | false (`32 != 16`) |

- Both fused matmul data-movement paths construct this receiver before their inner K loops:
  - `reader_bmm_tile_layout_in0_sender_padding.cpp:101-108` constructs it with `wait_for_op_signal=true`, `num_blocks_inner_dim`, and `in0_block_w`.
  - `reader_bmm_tile_layout_in1_sender_writer_padding.cpp:119-127` constructs it with `wait_for_op_signal=false`, the same `num_blocks_inner_dim`, and `in1_block_h` (the K-block height, also 8 here).
- The assertion mismatch is independent of the selected block width when valid divisibility is preserved: the factory advertises eight full tensor slices while TP4's gathered K dimension contains four. Changing `in0_block_w`, output cores, CCL links, worker count, buffers, chunks, core offset, or weight dtype cannot repair that slice-count mismatch.
- Existing generic fused-op coverage masks the defect: `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py` parametrizes `mesh_device` as `(1, 8)`. On TP8, the hardcoded ledger is `4 transfers * 2 directions = 8 slices`, so it matches the ring. The passing TP4 standalone all-gather does not instantiate `MatmulOpReceiver` and therefore never exercises this assertion.
- The direct-matmul `allowed_worker_cores` warnings are not the deadlock source. Current matmul code explicitly auto-populates the missing field and continues into program construction; moreover, that warning is emitted before the receiver ledger fails. Normalizing the program config is still appropriate cleanup, but it does not change `num_transfers` or the failed equality.

## Downstream Effects

- The first plausible stuck point is the failed `MatmulOpReceiver` constructor ledger on one or both matmul data-movement RISC-V kernels. Any all-gather workers waiting to synchronize or signal fused-op receiver semaphores, compute kernels waiting for CB data, dispatch completion, `FDMeshCommandQueue::finish_nolock`, and pytest teardown are downstream waiters or victims.
- The broad four-device host wait is expected because mesh synchronization waits for every device program to retire. It is not evidence that the physical ring, all four ARCs, or every Ethernet route independently failed.
- The separate AG plus matmul result and healthy ARC heartbeats specifically demote hypotheses involving the page table, QKV weight packing, BFP4 decoding, distributed norm, PCIe availability, and general fabric initialization.

## Proposed Fix

- In the generic `AllGatherMatmulAsyncMeshWorkloadFactory`, derive the number of equal bidirectional transfers from the actual ring size instead of hardcoding the TP8 value. For the current receiver protocol, the minimal fix is conceptually:

  ```cpp
  TT_FATAL(ring_size % MatmulOpReceiver::num_directions == 0, ...);
  const uint32_t num_transfers = ring_size / 2;
  ```

  Since `MatmulOpReceiver::num_directions` is device-kernel code and may not be host-visible, use a shared host/device constant or an explicitly named host constant rather than duplicating an unexplained literal. For TP4 this supplies `num_transfers=2`; the ledger becomes `2 * 2 * 4 == 16` and the four ring slices are consumed exactly once.
- Preserve or add a host-side validation for the actual fusion geometry before launch: an even ring under the current equal-two-direction receiver protocol, a slice width divisible by the matmul K-block width, and `ring_size * blocks_per_slice == total_K_blocks`. Odd rings need a receiver that supports unequal forward/backward transfer counts; silently rounding is unsafe.
- Normalize the matmul program config before invoking direct `MatmulDeviceOperation` helpers to remove the current warnings, but treat this as a separate cleanup rather than the hang fix.
- Add focused generic-op regression coverage for both TP4 and TP8, including a TP4 `[1,1,32,1024] -> [1,1,32,4096]` width all-gather followed by K=4096 matmul, persistent sharded AG output, traced replay, and numerical checks of both returned tensors.
- Focused verification/refutation sequence:
  1. Change only the transfer derivation and rerun the exact TP4 topology probe. Completion plus AG/MM PCC would verify the ledger diagnosis.
  2. In a separate watcher run, keep the old hardcoded value long enough to confirm the `MatmulOpReceiver` equality assertion, then reset; this is diagnostic only and is not required if the one-line ledger fix makes TP4 pass.
  3. Run the existing TP8 fused tests to prove the derived value preserves their four-transfers-per-direction behavior.
  4. If TP4 still stalls after the ledger fix, capture watcher or repaired tt-triage call stacks and inspect the next semaphore boundary; do not tune links, offsets, or matmul geometry before obtaining that evidence.

## Uncertainty

- The exact device core and RISC-V thread that hits the equality assertion were not directly captured because both available device introspection paths were unavailable (Inspector permissions and triage/UMD `noc_read` ABI mismatch). The diagnosis is therefore a source-contract inference, not a claimed PC-level observation.
- The source ledger is nevertheless high-confidence: it yields a deterministic false assertion for the exact TP4 shapes, explains why the same generic op is covered only on TP8, explains why standalone AG plus matmul passes, and predicts a minimal ring-size-derived fix with no dependence on performance-tuning parameters.
- Support for odd-sized rings remains outside the proposed minimal fix. The current two-direction receiver represents equal transfer counts and needs either an explicit even-ring validation or a broader protocol change.

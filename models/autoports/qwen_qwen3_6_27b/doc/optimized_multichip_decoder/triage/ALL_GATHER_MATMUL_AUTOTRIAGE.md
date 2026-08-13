# AUTOTRIAGE: TP4 fused all-gather + matmul stall

## Diagnosis

- `all_gather_matmul_async` hard-codes `num_transfers = 4`, and `MatmulOpReceiver` interprets that as four transfers in each of two directions: eight tensor slices. That contract is correct for the existing TP8 test but wrong for this TP4 ring. For the Qwen shape, the receiver constructs an 80-K-block ledger while the matmul has only 40 K blocks, trips its device-side assertion, and the host remains blocked in the first synchronization.

## Triage Evidence

- The supplied run completes mesh/fabric initialization and program construction, then emits only the fused matmul program-config warnings. It never reaches PCC, timing, JSON emission, or clean mesh close before the 90-second timeout.
- Watcher was attached to worker cores but had Ethernet inspection disabled and emitted no explicit kernel assertion. Therefore the log directly proves a silent device-side stall during the first fused invocation, not a Python/API validation failure and not a later trace/iteration problem.
- Reset/list recovery after terminating the process makes permanent hardware failure unlikely. The firmware-version warning is background uncertainty, not a source contract that explains the exact TP4-only geometry.

## Source Evidence

- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp` unconditionally sets `const uint32_t num_transfers = 4` before calling `MatmulFusedOpSignaler::init_all_gather(...)`.
- `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` defines `MatmulOpReceiver::num_directions = 2`, then computes:
  - `num_tensor_slices = num_transfers * num_directions`
  - `num_blocks_per_slice = tensor_slice_shape_width / tiles_per_block`
  - `ASSERT(num_tensor_slices * num_blocks_per_slice == num_blocks)`
- The Qwen probe has global K=5120, TP4 local K=1280, and `in0_block_w=4` tiles. Its ledger is:
  - local slice width: 1280 / 32 = 40 tiles
  - blocks per slice: 40 / 4 = 10
  - actual global matmul K blocks: (5120 / 32) / 4 = 40
  - hard-coded fused expectation: 4 transfers * 2 directions * 10 = 80 blocks
  - required TP4 expectation: 2 transfers * 2 directions * 10 = 40 blocks
- The checked-in nightly coverage in `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py` runs this fused API on a `(1, 8)` mesh. Its eight-slice geometry happens to match the hard-coded value, explaining why the defect survives existing tests.
- The probe's all-gather core offset `(0,6)` is patterned after that passing test, and the 8x6 matmul occupies y=0..5. Core overlap is therefore a weaker hypothesis. Likewise, the warning that `allowed_worker_cores` was auto-populated is actionable API cleanup but does not change the proven 80-versus-40 receiver invariant.
- A currently present unrelated worktree edit adds an atomic barrier only for `fuse_op_reduce_scatter`; it does not alter the all-gather receiver ledger and is not the missing fix here.

## Downstream Effects

- The broad host timeout at `ttnn.synchronize_device` is downstream of the fused matmul receiver's invalid slice/block contract.
- No PCC or latency result can be inferred because the first fused operation never completes.
- The all-gather fabric workers and eventual host/device teardown are victims/waiters. The available log does not support blaming fabric routing, persistent-buffer reuse, or repeated semaphore epochs.

## Proposed Fix

- Replace the hard-coded transfer count with a ring-derived count for the current bidirectional receiver protocol: `num_transfers = ring_size / 2` for an even ring. Add an explicit host-side fatal check for unsupported odd ring sizes unless the receiver is extended to represent unequal directional transfer counts.
- Add TP4 fused coverage with the exact invariant-sensitive shape class (local width 1280 tiles feeding global K 5120) and retain TP8 coverage. A host-side validation of `num_tensor_slices * num_blocks_per_slice == num_blocks` would turn future mismatches into an immediate diagnostic instead of a silent device assertion.
- Normalize the matmul program config before fused factory use to remove the `allowed_worker_cores` warnings, but treat that as separate hardening rather than the stall fix.
- Verification should run the TP4 probe once under Watcher with Ethernet visibility enabled, then repeat enough iterations to cover semaphore reuse. Require first-call completion, PCC, watcher-clean exit, and compare against the separate async all-gather + matmul control. Also retain a TP8 regression to prove the generalized count preserves the existing case.

## Uncertainty

- The supplied artifact is not a full `tt-triage --llm-output` capture and Watcher did not print the device assertion, so the exact RISC stop site is inferred from the source invariant at the first fused invocation. The arithmetic mismatch is deterministic for this configuration and is the first concrete source-side stuck point.
- Odd ring sizes are not representable by the present receiver's equal two-direction `num_transfers * 2` model. Supporting them needs a protocol change, not integer rounding.

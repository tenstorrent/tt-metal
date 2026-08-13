# AUTOTRIAGE

## Diagnosis

- The fused matmul/reduce-scatter program exits the matmul BRISC kernel with outstanding non-posted NoC semaphore atomics issued by `OpSignaler::synchronize_workers_and_signal_op()`. The kernel ends with `noc.async_write_barrier()`, which does not drain the non-posted atomic transaction class. The minimal source-side fix is to execute an atomic barrier on every signaling matmul worker after its final `OpSignaler` use and before kernel return (most narrowly, `noc.async_atomic_barrier()` under `fuse_op_reduce_scatter` in this kernel). This diagnosis is high confidence for the watcher assertion, but it does not yet prove that the fused operation is otherwise correct or performant.

## Triage Evidence

- Two independent preserved watcher runs reproduce the identical assertion:
  - `logs/fused_ccl_o_b32_w3_watcher_second.log`, lines 64-72
  - `logs/fused_ccl_o_b32_w3_fused_only_watcher.log`, lines 64-72
- Both runs identify device 0 worker core logical `(0,0)`, virtual `(1,2)`, BRISC, in `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`. Watcher states that the kernel completed with pending NoC transactions and specifically reports a missing non-posted-atomics flushed barrier.
- The fused-only reproduction rules out the earlier separate-path Python invocation error as the cause. The first `*_watcher.log` failed before device execution because it used obsolete keyword arguments for `reduce_scatter_minimal_async`; the later fused-only run contains no separate reduce-scatter call and still trips the same device assertion.
- The failure happens after fused program construction and dispatch. The four repeated `allowed_worker_cores` warnings are host-side normalization warnings, not the asserted transaction class; the helper auto-populates the field and proceeds to launch.
- The preserved watcher stdout gives the useful stop-site and running kernel set, but no live `tt-triage` capture exists because the process is no longer live. `generated/watcher/watcher.log` is not evidence for this failure now: it has subsequently been overwritten by a healthy mesh-smoke/blank-kernel watcher attachment (`k_id[0]: blank`).
- The broad process abort, core dump, and host stack in watcher polling are downstream effects of Watcher deliberately stopping the device after detecting the BRISC invariant violation.

## Source Evidence

- `matmul_reduce_scatter_async_program_factory.cpp:create_at()` constructs a `ReduceScatterFusedOpSignaler`, passes its receiver core coordinates/semaphores into a `MatmulFusedOpSignaler`, and passes that signaler to `matmul_multi_core_reuse_mcast_2d_optimized_helper()`. Thus the fused path, unlike ordinary matmul, enables the reduce-scatter signaling branch in the matmul dataflow kernel.
- `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` reads compile-time argument 31 into `fuse_op_reduce_scatter`, constructs `OpSignaler` from runtime arguments, and after each batch calls `op_signaler.synchronize_workers_and_signal_op(0)`.
- `worker_sync_utils.hpp:OpSignaler::synchronize_workers_and_signal_op()` issues remote semaphore increments using `Semaphore::up(noc, x, y, 1)`:
  - each non-master worker issues one atomic increment to the master's worker-sync semaphore, then waits for release;
  - the master issues one atomic increment to each reduce-scatter receiver semaphore (or one selected receiver when not multicast mode);
  - when multiple matmul workers participate, the master also issues one atomic increment to each slave's release semaphore.
- `noc_semaphore.h:Semaphore::up(const Noc&, ...)` calls `noc_semaphore_inc`, an asynchronous remote atomic. Neither `OpSignaler::synchronize_workers_and_signal_op()` nor its caller performs `async_atomic_barrier()`.
- At kernel exit, `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` only calls `noc.async_write_barrier()`. The NoC API exposes a separate `async_atomic_barrier()` which waits on `ncrisc_*noc_nonposted_atomics_flushed`; therefore the existing final write barrier cannot satisfy the exact invariant Watcher checks.
- The asserted core `(0,0)` is consistent with the master side of this ledger: it can issue receiver-start atomics plus slave-release atomics immediately before returning. The ledger also requires a barrier on slaves because their sync-to-master increment is likewise a non-posted atomic; placing the barrier in the caller on every fused matmul worker is safer and narrower than guarding only the observed master core.
- The proposed behavior is absent in the prepared source. Other kernels that issue remote synchronization atomics explicitly call `noc.async_atomic_barrier()` (for example the matmul ring-all-gather kernels and reduce-scatter writers), while this fused matmul sender/writer does not.
- The related regression test `tests/ttnn/nightly/unit_tests/operations/matmul/test_dram_sharded_then_1d_matmul.py` is not a competing explanation for this assertion. It documents persistent custom read-VC state inherited *after a preceding DRAM-width-sharded matmul*, causing a bandwidth regression. This probe starts a fresh process and its `[6144,5120]` per-mesh weight is K-sharded across devices but DRAM-interleaved within each device, so this kernel's `IN1_DRAM_WIDTH_SHARDED`/`set_async_read_state<CUSTOM_VC>` branch is not selected. More decisively, stale read VC explains performance, whereas Watcher reports an outstanding non-posted atomic and the fused-only path demonstrably issues such atomics through `OpSignaler`.

## Downstream Effects

- The host `TT_THROW`, abort, core dump, and timeout message are consequences of Watcher stopping the device, not separate root causes.
- NCRISC and TRISC kernel names in the watcher report describe the concurrently launched matmul program. There is no preserved evidence that they are the first fault; the precise invariant violation is owned by BRISC.
- Reduce-scatter workers may receive the start signal and run, wait, or race with kernel teardown depending on atomic completion timing. Those states are victims of the missing completion contract and must not be diagnosed as a fabric-route or CCL deadlock from the current evidence.
- The `allowed_worker_cores` normalization warnings should be cleaned up independently before that warning becomes a hard error, but they do not explain a pending non-posted atomic on BRISC.
- The known stale-custom-VC issue from a prior DRAM-sharded matmul is also orthogonal here: there is no preceding DRAM-sharded matmul in the fused-only fresh-process reproduction, and that issue changes read bandwidth rather than leaving semaphore atomics pending at kernel completion.

## Proposed Fix

- Minimal candidate: after the last `op_signaler.synchronize_workers_and_signal_op(0)` use, execute `noc.async_atomic_barrier()` on every matmul sender/writer core before it can return. For the current kernel, a narrow form is an atomic barrier inside the `fuse_op_reduce_scatter` branch after signaling (per batch), or once after the batch loop guarded by `fuse_op_reduce_scatter`. A once-after-loop barrier is sufficient for kernel-exit safety; a per-batch barrier is only needed if semaphore reuse across batches requires completion before the next batch.
- Do not replace the existing `noc.async_write_barrier()`: it still owns ordinary output/multicast write completion. Atomic and write transaction classes need their respective barriers.
- Minimal verify/refute experiment:
  1. Add only the guarded atomic barrier, rebuild the affected device kernel, reset/mesh-smoke per `tt-device-usage`, and rerun the exact fused-only B32, `in0_block_w=3` command under Watcher. Require clean watcher termination, synchronization, PCC >= 0.999, and a result JSON.
  2. Repeat with at least 20 iterations so program-cache reuse and persistent semaphore reuse are exercised. A one-shot clean launch is insufficient.
  3. Run `--mode separate` under Watcher with the probe's current `persistent_output_buffers=[intermediate, persistent]` API as a negative control. It should remain clean and confirms the assertion is introduced by matmul-to-reduce-scatter fusion signaling.
  4. If the guarded atomic barrier does not clear the assertion, instrument or capture fresh `tt-triage`/Watcher ringbuffer evidence and compare master versus slave cores. The next boundary to inspect is runtime construction of `workers_noc_coords`, receiver coordinates, and semaphore IDs/counts in the matmul fused-op signaler—not fabric teardown.
- Placement discriminator: first test a caller-local barrier in this kernel. If that works, test moving the barrier into `OpSignaler::synchronize_workers_and_signal_op()` only if all callers require completion semantics; otherwise keep it at the kernel lifecycle boundary to avoid serializing intended overlap in callers that continue useful work.

## Uncertainty

- No live tt-triage call stacks, semaphore values, or transaction counters were captured, so the report cannot distinguish which of the master's receiver-start or slave-release atomics remained pending. That distinction does not change the required kernel-exit atomic barrier.
- The current evidence proves an unsafe kernel-completion contract; it does not establish whether there is an additional receiver-count, core-coordinate, or semaphore-reuse bug after the barrier is added. PCC and multi-iteration watcher runs are required.
- The failure is observed on device 0 core `(0,0)` in both preserved runs. Watcher stops on the first detected assert, so absence of reported assertions on other devices/cores is not proof that only that core is affected.
- `generated/watcher/watcher.log` has been overwritten and cannot strengthen the historical diagnosis. Preserve the next failing or passing watcher file under this stage directory immediately after the experiment.

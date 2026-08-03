# AUTOTRIAGE

## Diagnosis

- The hang is the known single-core `sparse_matmul` in0-multicast deadlock, triggered by the sweep candidate `bfp4_geo_u1_d2_k1_1`; it is not a BFP4 numerical issue and the captured stop site does not match an `nnz` mismatch. The candidate makes the packed up/gate projection a 1x1-grid operation, but the current sparse factory does not define `SKIP_MCAST` for its in0 sender. With no receiver core, that sender nevertheless issues a zero-destination multicast and stalls forever in the Blackhole NoC flush before it can publish in0 to the compute CB.

## Triage Evidence

- Failing command:

  ```text
  GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_GEOMETRY=1 TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q 'models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_sparse_geometry_sweep[blackhole-bfp4-sliding_attention-device_params0-mesh_device0]'
  ```

- `triage/geometry_bfp4_hang.txt` identifies op 85, the first `SparseMatmulDeviceOperation`, as the only running model op on device 3/core `(0,0)`. Its tensors are packed-up/gate shapes: A `[1,1,1,2816]` BF16, B `[1,128,2816,1408]` BFP4, sparsity `[1,1,1,128]` BF16. Op 92, the later down sparse matmul, has not started.
- The BRISC is stopped at `reader_bmm_tile_layout_in0_sender_padding.cpp:377`, in `noc.async_writes_flushed()`. This is exactly the Blackhole flush immediately following the in0 multicast. The NoC check reports a one-count nonposted-write/ack mismatch on the same worker (`noc_nonposted_writes_num_issued=3530809137`, acked `3530809138`), consistent with the malformed degenerate multicast, not an idle board.
- The compute unpack thread waits at `bmm_large_block_zm_fused_bias_activation.cpp:297` on `in0_cb.wait_front(in0_block_num_tiles)`: the in0 producer never reaches its later `cb_in0.push_back`. The in1 reader is stopped at `cb_in1.reserve_back` because compute no longer drains the weight CB. Dispatch/prefetch waits are downstream host/queue backpressure.
- ARC heartbeat, DDR, Ethernet status, inactive-CB check, and core-magic checks passed. The broad binary-integrity and broken-core output is a consequence of inspecting/stopping live cores during triage and does not explain why the sole running op stopped at the sparse reader's multicast flush.

## Source Evidence

- `optimized_decoder.py:137-143` enumerates two one-up/gate-core rows for both BFP4 and BFP8: `(1,2,1,1)` and `(1,2,22,22)`. `test_optimized_decoder.py:211-226` sorts candidate names, constructs `names[0]`, and executes it first. For BFP4 that is `bfp4_geo_u1_d2_k1_1`, matching the reported candidate attribution.
- `_sparse_program_config` (`optimized_decoder.py:163-191`) maps that row to:

  ```text
  N = 1408 -> Nt = 44 tiles
  grid = (1,1)
  per_core_N = 44
  out_block_w = 44
  out_subblock_w = 4
  in0_block_w = 1
  per_core_M = 1 tile
  mcast_in0 = true
  ```

- The sparse program factory computes one output block and one working core for this geometry (`sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:128-148,208-223`). Its producer/consumer ledger is:

  | Resource | Producer | Consumer | Required count | Actual one-core behavior |
  |---|---|---|---:|---|
  | in0 multicast receiver-ready semaphore | in0 receiver kernel | in0 sender BRISC | `num_cores - 1 = 0` | no receiver kernel is launched |
  | in0 multicast payload | sender BRISC | receiver BRISCs | `receiver_grid_size - 1 = 0` | sender still calls multicast on the self-only bounding box |
  | CB0 in0 blocks | sender BRISC | compute unpack thread | 88 K blocks per selected expert batch (`Kt=88`, `in0_block_w=1`) | sender stalls at the first multicast flush before `push_back`; compute waits for CB0 |
  | CB1 weight blocks | in1 NCRISC | compute unpack thread | same fixed sparse-compute loop | compute cannot progress, so the producer eventually blocks reserving CB1 |

- The in0 dataflow kernel gates the multicast handshake, multicast write, Blackhole flush, and receiver semaphore multicast only under `#ifndef SKIP_MCAST` (`reader_bmm_tile_layout_in0_sender_padding.cpp:348-389`). The current sparse factory sets `SKIP_MCAST` only for the in1 sender (`sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:401-435`), so the one-core in0 sender takes the invalid path.
- The dense 1D-mcast factory already has the missing guard: `if (in0_mcast_receiver_num_cores == 1) ... ["SKIP_MCAST"] = "1"` (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:540-546`).
- Repository history contains the exact sparse fix in commit `341ffae7862` (`Fix sparse_matmul single-core mcast deadlock (TP=8 sharded weight)`), but that commit is not an ancestor of the current HEAD and its guard is absent from the prepared source. Its recorded failure mechanism and fix exactly match this triage: one-core grid, zero in0 receivers, sender still runs multicast, permanent NoC/semaphore deadlock.
- This is not the known wrong-`nnz` trigger. Both packed expert calls use the inherited exact `TOP_K_EXPERTS=8` contract, and the fused/default path uses the same routing mask and `nnz`. More importantly, the captured BRISC has advanced past the per-entry `num_valid_batches <= num_batch_compute` assertion and is stopped specifically in the multicast flush inside a valid batch. A too-small count would assert earlier; a too-large count would leave receiver/compute waiting after the sender finishes its valid batches, not leave the sender itself at the first zero-destination multicast flush.

## Downstream Effects

- Compute's CB0 wait and the in1 reader's CB1 reserve wait are victims of the stopped in0 sender, not independent CB sizing defects.
- Queue dispatch/prefetch waits are expected once op 85 never completes. The later down projection and trace replay are not implicated because they were never reached.
- The trigger is geometry, not dtype: the BFP8 sweep has the same one-core rows and would hit the same kernel contract if run. BFP4 merely ran first in the failing parametrization.

## Proposed Fix

- Framework-level fix: apply the already-proven guard from commit `341ffae7862` in the sparse factory so a one-core in0 sender is compiled with `SKIP_MCAST`. That is the semantically complete fix, but it is outside this stage's authorized model/test/doc file scope.
- Smallest safe model-side action for this stage: exclude every explicit `expert_up_gate_cores == 1` candidate from `POLICIES`/the geometry sweep (both `k1` and `k22`, for BFP4 and BFP8), or reject one-core sparse configs before device dispatch and record them as `kernel_single_core_mcast_blocked`. Continue the precision-locked sweep with the 2-core and 4-core up/gate candidates; the default 4-core up/gate geometry is unaffected. Do not silently remap a candidate named `u1` to two or four cores, because that would make the geometry evidence false.
- Add a host-only regression asserting the optimized candidate set cannot dispatch an explicit one-core sparse 1D-mcast config while this checkout lacks the sparse-factory `SKIP_MCAST` guard. If the framework commit is later integrated, replace that exclusion with the upstream single-core sparse-matmul regression and re-enable the one-core measurement.

## Uncertainty

- No new hardware experiment was run, by instruction. Verification should be performed after the model-side exclusion by rerunning the exact geometry selector and confirming that at least two BFP4 rows complete, then repeating BFP8; this is a repair-loop task, not part of this inspection-only report.
- The report does not claim that one-core geometry is intrinsically unsupported or slow. It is specifically unsafe in this checkout because the known sparse-factory guard is missing. Once `341ffae7862` or an equivalent framework change is present, it becomes a valid candidate again.

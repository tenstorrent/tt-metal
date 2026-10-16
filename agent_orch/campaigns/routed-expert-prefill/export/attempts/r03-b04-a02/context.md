## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: the job, the metric, allowed paths and the precision rules.
- $DREAM_HOME/routed-expert-prefill/history.md: the index and leaderboard.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp (the whole file): the per-K-block gate/up handshake.
  - Receivers ack, the sender (gy = group base) reads gate on NoC 0 and the writer multicasts.
  - The row-shared x owner issues the x(kb+2) row-major read at the top of the iteration and barriers it at the end, before the step-3 weight push.
  - The gate sender's `noc_read.async_read_barrier()` is global, so it also waits for an x read issued on the same core.
  - Every core of a group's first row is a gate sender, so one sender per block is hit.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp: the UP_SPLIT `up` read and the IN1_WRITER_MCAST gate+up multicast on NoC 1, the DOWN_SPLIT half, and the cb_out drain.
- kernels/compute/fused_swiglu.cpp, `matmul_phase_fused_gu`: the owner tilizes x(b+1) before its matmul of block b (`tilize_owned_x_block`). That fixes the latest point at which cb_x_rm(j) must be pushed: before the compute's matmul j-1.
- kernels/weight_runs.hpp: a gate K-row slice is one NoC read through `noc.async_read`, so it carries the sticky read-cmd-buf tag.
- device/unified_routed_expert_ffn_program_factory.cpp, lines 600-760: UP_SPLIT, DOWN_SPLIT and IN1_WRITER_MCAST are on by default, the ring depths, and Reader/Writer config descriptors.
- tt_metal/hw/inc/api/dataflow/noc.h, dataflow_api.h, internal/tt-1xx/blackhole/noc_nonblocking_api.h:
  - `noc_async_read_set_trid` is a sticky NOC_PACKET_TAG on read_cmd_buf.
  - `noc_async_read_barrier_with_trid` / `ncrisc_noc_read_with_transaction_id_flushed` poll NIU_MST_REQS_OUTSTANDING_ID(trid).
  - Noc::async_read<TXN_ID> throttles per id, so the raw sticky call is used instead.
- tt_metal/hw/inc/api/semaphore.h: `Semaphore::value()` is a non-blocking poll.
- ttnn/.../moe_fused_swiglu/device/kernels/moe_fused_swiglu_reader.cpp: an in-tree precedent for the same sticky-trid pattern, including a separate trid for an x look-ahead read (NEXT_X_TRID).

## Nodes consulted
Every node's reflection, plus the proposals of the round-3 and lineage nodes. Most relevant:
- r03-b04-a01 (parent): planner trim. Its next #3 asks to measure the per-expert fixed cost on the compute/stream-bound shapes with NCRISC zones, which record reliably (TRISC_0 zones do not).
- r03-b02-a01 (best, 1.5887): already combines a planner trim with the PACK pipeline, so porting the pipeline here would only repeat it.
- r02-b04-a03: the group-tail lottery is a gate/up delivery effect. Sometimes one row of a group is slow, and that "row's own x stream is the suspect" (its next #3 b/c).
- r02-b02-a02 / r02-b02-a03 / r01-b04-a03 / r01-b01-a03: request VCs, down look-ahead, ring depth and pacing do not fix the tail. What they say about aggregate DRAM sharing frames the expected size.
- r01-b03-a01 / r01-b04-a01: weight read-ahead before the multicast lost. This node moves no weight read.
- r01-b03-a02: x batching made x burstier and lost. This node keeps x size and gives it more lead time.
- r01-b01-a02 / r01-b01-a04: the row-shared tilize design, and the cross-unit look-ahead (neutral).
- r02-b01-a02 / r02-b04-a02: phase zones. GU on mistral is ~117 µs per expert (8 blocks) against ~55 µs of LoFi compute.

## Docs / external references
- None beyond the in-tree headers above.

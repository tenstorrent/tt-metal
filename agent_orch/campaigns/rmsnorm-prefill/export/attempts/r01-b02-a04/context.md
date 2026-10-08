## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml: process, allowed paths, accuracy gate.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: the deep_drain loop (cumulative wait per block, one
  flush plus a padded-row pop per row, all on BRISC/NoC0). This is where the even/odd split and the drain_sem wait
  before the pop go.
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp: the reader's row loop (deep trid input read, then the
  single-barrier gamma batch). After the loop NCRISC has nothing left to do, so the drain of the odd blocks is
  appended there. Common args 0-5 were in use, so output addr is arg 6.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp: the reader is created for both the mux and TP=1 paths.
  The worker writer's CT args end with deep_drain. Read num_tile_rows_per_worker, block_major_post, the semaphore
  creation, the common-arg asserts (reader 6 -> 7) and override_runtime_arguments (refresh reader_common[6]).
- tt_metal/hw/inc/api/dataflow/dataflow_api.h (cb_wait_front / cb_pop_front): tiles_acked is a shared register, so
  if the writer popped before the reader passed its cumulative wait_front, the reader's wait would go wrong. Hence
  the drain_sem handshake before the pop.
- tt_metal/hw/inc/api/semaphore.h, api/dataflow/semaphore_dm_impl.h: a bare-id Semaphore up(noc, x, y) is a NoC
  atomic inc, and wait_min is noc_semaphore_wait_min, so a self-targeted inc from NCRISC on NoC1 with
  my_x/my_y[noc id] works.
## Nodes consulted
- All 12 nodes (proposal + reflection). Key ones:
- r01-b02-a03 (parent): drain end trails TRISC end by 1.2-3.5 µs on h7168. NCRISC is idle after ~8.5 µs.
- r01-b02-a02: the per-block flush is not the drain bottleneck (neutral).
- r01-b03-a02 / r01-b03-a03: ~200 GB/s write ceiling regardless of core count. Bank de-phasing barely moves the
  drain. The drain end grows with core x/y, which suggests NoC0 path congestion and motivates using NoC1.
- r01-b04-a01 / r01-b04-a03: the drain is the critical path once x*gamma hides under the AG. They also suggest a
  dual-NoC drain.
- r01-b01-a02 / r01-b04-a02: don't put extra traffic on NCRISC before its input read finishes. This node only uses
  NCRISC after all reads are done.
## Docs / external references
- none

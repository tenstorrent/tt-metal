## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml: the job, the metric and the allowed paths.
- $DREAM_HOME/rmsnorm-prefill/ledger/rounds/r01/summary.md, rounds/r02/manifest.json: round root = r01-b04-a04 (a89c280).
- tests/ttnn/nightly/unit_tests/operations/fused/test_fused_rms_norm_prefill.py: 640 rows (20 tile-rows), 1 link, Linear, 4 chips.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp: 20 workers placed row-major (logical rows 0-1) + 1 forwarder.
  Worker writer = WriterDataMovementConfig (BRISC, NoC0). Reader = ReaderDataMovementConfig (NCRISC, NoC1).
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: W_GAMMA, W_PUSH, W_AGWAIT, stats read, W_DRAIN
  (one 2 KB NoC0 write per tile, flush per 4-tile block).
- kernels/dataflow/dit_rmsnorm_fused_reader.cpp: NCRISC only issues reads (trid-pipelined input). No NoC
  writes/atomics, so the NoC1 write/ack counters are free for BRISC once the input read is done.
- dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp (read-only, outside allowed paths): AG protocol.
- tt_metal/hw/firmware/src/tt-1xx/brisck.cc: in dedicated-NoC mode the kernel prologue calls
  noc_local_state_init(NOC_INDEX) only, so BRISC must sync its NoC1 counters itself.
- tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h: noc_local_state_init, noc_init (inits cmd bufs on
  all NoCs), BRISC_WR_CMD_BUF == NCRISC_WR_CMD_BUF == 0.
- tt_metal/hw/inc/api/dataflow/noc.h, api/tensor/noc_traits.h: Noc(noc_id) object, and the accessor address uses the Noc's id.
- tt_metal/soc_descriptors/blackhole_140_arch.yaml: DRAM columns at physical x=0 (ch 0-3) and x=9 (ch 4-7),
  per-NoC endpoints. Grid 17x12.
- umd blackhole_coordinate_manager.cpp (eval checkout): translated DRAM x = 17 for the west banks and 18 for the
  east banks (no DRAM harvest). Translated tensix x = NoC0 physical x of the logical column.

## Nodes consulted
- r01-b04-a04 (root): the timeline. The drain is throughput-bound from the AG end; drain tail 1-4 µs after TRISC.
- r01-b02-a04: the position-blind 50/50 NoC split regressed. Its per-core x gradient is the evidence for NoC0 row-link
  congestion, and its suggestion is a destination-aware NoC choice.
- r01-b03-a04: 50/50 dual-NoC on 80 cores. The gradient flipped instead of flattening, so the NoC links are directional.
- r01-b03-a03: bank de-phasing barely touched the drain, so it isn't a DRAM-bank queue problem.
- r01-b01-a01..a04, r01-b02-a01..a03, r01-b03-a01/a02, r01-b04-a01..a03: lineage history; all read.

## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml — job + gate + allowed paths
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — W_PUSH / go / stick read / W_DRAIN loop; drain issues every tile through one cmd buf per NoC (Noc::async_write -> write_cmd_buf)
- tt_metal/hw/inc/api/dataflow/noc.h (Noc::async_write) and dataflow_api.h (noc_async_write) — default path = ncrisc_noc_fast_write_any_len on write_cmd_buf, NOC_UNICAST_WRITE_VC, non-posted
- tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h — DM_DYNAMIC_NOC cmd buf map (BRISC: 0 = writes, 1 = reads+atomics, on both NoCs; NCRISC 2/3), dynamic_noc_init (cmd buf 0 TARG coord = local, cmd buf 1 RET coord = local), ncrisc_noc_fast_write (writes CTRL/TARG_LO/RET_LO/RET_COORD/LEN/CMD_CTRL, not TARG_COORD), fast_read (dynamic: rewrites CTRL; relies on preset RET_COORD), noc_fast_atomic_increment (rewrites TARG coord, relies on RET coord), dynamic counters per (risc, noc) not per cmd buf
- tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc.h — noc_command_ready: "no pending request that is being backpressured by the NOC" => a cmd buf is busy while its write injects
- tt_metal/tools/profiler/kernel_profiler.hpp — profiler flush uses write_cmd_buf with save/restore, so cmd buf 1 is untouched by it
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — dual_noc_drain implies writer noc_mode = DM_DYNAMIC_NOC
## Nodes consulted
- all 31 nodes (proposal + reflection), see history.md
- r03-b04-a02 — drain is per-core-bound (10 lone writers drain at the same per-core rate as 20)
- r02-b02-a01 — path-aware dual-NoC drain rule kept unchanged here; BRISC is NoC1 by default on BH
- r01-b02-a02 — flush-per-row neutral (so the per-block flush is not the cap)
- r01-b02-a04 / r01-b03-a04 / r02-b03-a01 / r02-b01-a01 — NoC-split variants that changed routing (congestion flips); this node does not change routing
- r03-b02-a01/a02 — parent chain; C_COMB/C_POST TRISC zones used in drain2.py
## Analysis
- drain.py / drain2.py (this dir) on reports/r03-b02-a02: drain per core ~10.5 tiles/µs from the first POST tile, uniform across cores at h3584; pack POST produces 14 tiles/µs. Output in drain2_parent_out.txt.
- drain_out.txt / drain2_out.txt — same scripts on this node's report (per-core drain rate fell uniformly ~8%)
- r03-b02-a02/ag.py on this node's report — pre-drain path (F_COLLECT, go, stick read) unchanged

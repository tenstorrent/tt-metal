## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml — job definition, allowed paths, gate.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — W_DRAIN loop (per-block wait/pop on NoC0, rotated
  output column), stick push / go relay; BRISC is the only output consumer in the parent.
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp — NCRISC finishes input + gamma reads before the AG and is
  idle after; has col_start / row_stride / col_rot RT args (same output column mapping can be reproduced).
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — col_split_ok gating (one row per row-worker), output_cb
  = 2 padded rows (whole-row resident, no wrap), reader/writer CT/common args, override_runtime_arguments.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — output_cb pushed per block of block_size (padded).
- tt_metal/hw/inc/api/semaphore.h — local Semaphore set/wait semantics (plain store / polled read).
## Nodes consulted
- All 12 nodes (b01..b04, a01..a03): proposals, reflections, summaries.
- r01-b03-a03 (parent) — drain is the tail, not bank-bound; recommends splitting drain across NoCs.
- r01-b03-a02 — drain end grows with core x/y; ~200 GB/s aggregate drain regardless of core count.
- r01-b02-a02 — one flush per row in the drain was neutral (not flush serialization).
- r01-b04-a01 / r01-b04-a03 — also suggest NCRISC draining half the row on NoC1.
## Profiling
- /tmp/r01b03a04/percore.py on reports/r01-b03-a03 profile_log_device.csv: per-core median W_DRAIN end - AG end
  (h7168) rises smoothly from 3.6 µs at (x=1,y=2) to 10.2 µs at (15,8); TRISC end - AG is 3.3-5.2 µs everywhere.

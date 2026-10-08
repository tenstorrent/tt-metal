## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — rules, metric, allowed paths
- kernels/dataflow/dit_rmsnorm_fused_reader.cpp — `read_input_pass` barriers every block_size(4) tiles (latency bound);
  weight read barriered per block; schedules INPUT_FIRST/SPLIT/DEFER_ALL
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — W_PUSH / W_AGWAIT / gather read / W_DRAIN; drain flushes
  + pops per 4-tile block
- kernels/compute/dit_rmsnorm_fused_compute.cpp (PRE section) — PRE waits cumulatively on input, block-granular,
  so pushes may arrive in any block granularity
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — input_cb = 2 whole rows (resident), output_cb =
  2 padded rows unless block_major_post (then 2*block_size); worker-writer CT arg layout; reader on NOC1, writer NOC0
- tt_metal/hw/inc/api/dataflow/noc.h, dataflow_api.h, blackhole noc_nonblocking_api.h — `async_read<TXN_ID>`
  sets the sticky NOC_PACKET_TAG, `async_read_barrier<TXN_ID>` polls NIU_MST_REQS_OUTSTANDING_ID(trid); 16 trids,
  255 outstanding per trid
- reports/r01-b02-a01/.logs/profile_log_device.csv — zone timeline for the parent (table in proposal.md)

## Nodes consulted
- r01-b01-a01 — x*gamma under AG; reflection: input read latency bound (~0.5 µs / 8 KB block), drain tail
- r01-b04-a01 — same compute idea; reflection: per-block `async_writes_flushed` makes drain latency bound
- r01-b02-a01 (parent) — column split never engaged (34-stick packet cap); plumbing harmless at col_split=1
- r01-b03-a01 — column split with leader combine; 60 cores read 2.24 MB in ~5.5 µs (DRAM headroom estimate);
  cross-chip launch skew when more cores / kernel groups are dispatched

## Docs / external references
- none beyond the tt-metal headers above

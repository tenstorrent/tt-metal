## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml: the rules, metric, allowed paths
- kernels/dataflow/dit_rmsnorm_fused_reader.cpp: `read_input_pass` barriers after every 4-tile block (one block in
  flight). The broadcast gamma is read after the input row, also with a per-block barrier.
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: stick push, AG wait, W_DRAIN with a per-block flush
- kernels/compute/dit_rmsnorm_fused_compute.cpp: PRE waits on input cumulatively per block. The parent's prescale x*gamma
  pass waits on weight cumulatively, and POST is a single pass.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp: block_size = dst_reg_count (4), input_cb = 2 rows,
  weight_cb = num_tile_cols, output_cb = 2 padded rows
- tt_metal/hw/inc/api/dataflow/noc.h, dataflow_api.h: `async_read<NocOptions::TXN_ID>`,
  `async_read_barrier<TXN_ID>`, `noc_async_read_set_trid` (BH trids 0..15)
- ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_reader.cpp: a prior example of trid-pipelined reads
## Nodes consulted
- r01-b04-a01 (parent): x*gamma pre-pass. Its reflection shows the gain shrinking with width. Re-profiled it: at h7168 the
  gamma read ends 15.4-16.8 µs, after the AG wait ends (~14.1), so the pre-pass stalls on gamma.
- r01-b01-a01: same pre-pass plus a batched gamma read (1.109). Its reflection names input-read pipelining as the
  biggest untried lever.
- r01-b02-a01, r01-b03-a01: column split. b03's 60-core read shows ~460 GB/s per chip is achievable (vs ~315 GB/s here).
## Docs / external references
- Zone timelines from reports/r01-b04-a01 and r01-b01-a01 profile_log_device.csv (call ids 61441 = h7168,
  15361 = h3584, chip 1)

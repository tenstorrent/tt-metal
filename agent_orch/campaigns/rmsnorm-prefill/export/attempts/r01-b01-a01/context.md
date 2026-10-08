## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml — job + campaign definition (allowed paths = the op dir, recursively via fnmatch).
- tests/ttnn/nightly/unit_tests/operations/fused/test_fused_rms_norm_prefill.py — seq 640 (20 tile-rows), TP=4, broadcast bf16 weight, no bias/RoPE, num_links=1, default compute config (HiFi4, fp32 dest).
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — worker cap / compute_sizing (20 workers, 1 row each, 1 forwarder), CB sizing (intermediate_cb = whole padded row fp32, weight_cb = num_tile_cols, output_cb = 2 rows), block_size = dst count (4).
- device/dit_fused_distributed_rmsnorm_device_operation.cpp — default compute config = HiFi4, fp32_dest_acc_en.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE (x^2 + l1-acc + row reduce, transpose stick), AG wait, POST sub-phase 1 (x*rsqrt -> fp32 intermediate) and sub-phase 2 (*gamma -> output).
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — stick push / AG wait / gathered read / drain (zones W_PUSH, W_AGWAIT, W_DRAIN).
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp — input pass is per-block barriered (comment claims deep read; code barriers every block_size tiles); broadcast weight read is per-block barriered after the input row.
- ttnn/.../dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp (read-only, outside allowed paths) — one packet per round, max payload 4352 B = 34 x 128 B sticks per forwarder.
- tt_metal/fabric/erisc_datamover_builder.hpp — default packet payload 4352 B.
## Nodes consulted
- none (first node of the campaign; history empty).
## Profiler evidence
- $DREAM_HOME/rmsnorm-prefill/reports/baseline_1/reports/*/profile_log_device.csv parsed per zone (script kept outside the repo).
  h7168: R_INPUT ~0-8 us, AG done ~12.6 us, TRISC end ~22 us, W_DRAIN end ~23-26 us, NCRISC (weight read) ends ~15.5 us.
  h3584: R_INPUT ~0-4.5 us, AG done ~8.8 us, TRISC end ~14.5 us, W_DRAIN end ~15-17 us.
## Docs / external references
- none

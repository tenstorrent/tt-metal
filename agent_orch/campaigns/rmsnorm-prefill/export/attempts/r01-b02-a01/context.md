## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job definition, shapes (20 tile-rows x 28/32/48/56 local tile-cols), gate.
- tests/ttnn/nightly/unit_tests/operations/fused/test_fused_rms_norm_prefill.py — TP=4 linear, num_links=1, bf16 weight broadcast, no rope/bias, DRAM interleaved in/out.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — worker cap (derive_worker_cap: BH knee 64 for ring<=4), one worker per tile-row => only 20 workers + 1 forwarder; stats DRAM geometry depends only on forwarders*max_rounds (so col split keeps it unchanged); RT/CT arg layouts.
- kernels/dataflow/dit_rmsnorm_fused_reader.cpp — reads input in 4-tile blocks with a barrier per block (latency bound), broadcast weight face-row reads.
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — stick push to forwarder slot, go-sem wait, gather read of ring_size sticks from DRAM, output drain.
- kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE sum(x^2), packed AG sum of stats_tiles_cols gathered tiles + transpose_dest + 1/H + eps + rsqrt, POST x*rms (fp32 intermediate) then *weight. Handles non-multiple-of-block_size widths. 1/H uses num_tile_cols*32*stats_tiles_cols, so local cols x (ring*col_split) still equals full hidden.
- ccl/dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp (read-only, outside allowed paths) — coalesces group_size sticks, single fabric mcast of pc*128 B; generic in group size.
- device/dit_fused_distributed_rmsnorm_device_operation.cpp — validate/compute_output_specs use compute_sizing only for stats buffer geometry.
## Nodes consulted
- none exist (first round); baseline profiler logs in $DREAM_HOME/rmsnorm-prefill/reports/baseline_1 used for a per-zone timeline (R_INPUT ~8us, AG ~4us, POST ~9us, drain tail ~3us on kimi-k2-7).
## Docs / external references
- none

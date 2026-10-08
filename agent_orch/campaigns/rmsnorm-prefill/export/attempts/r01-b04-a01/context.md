## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml — rules, shapes, gate
- tests/ttnn/nightly/unit_tests/operations/fused/test_fused_rms_norm_prefill.py — 640 rows (20 tile-rows), TP=4, bf16, broadcast weight, no bias/rope, default compute config (HiFi4, fp32 dest acc)
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — worker count (20 workers = 1 row each, 1 forwarder), CB sizes (intermediate_cb = padded whole row fp32), streaming/block-major decisions (both off for these shapes)
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE (x^2 + row reduce), transpose stat, wait gathered, POST sub-phase 1 (x*1/rms) and 2 (*gamma)
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — W_PUSH / W_AGWAIT / W_DRAIN zones
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp — input row read first, then broadcast weight (face-row reads), so weight is resident before PRE ends
## Nodes consulted
- none (round 1, first node in history)
## Profiler data
- $DREAM_HOME/rmsnorm-prefill/reports/baseline_1/reports/*/profile_log_device.csv — per-zone timeline (R_INPUT, W_PUSH, F_FABRIC, W_AGWAIT, W_DRAIN); script /tmp/r01b04/zones2.py

## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml: the job, the shapes (20 tile-rows x 28/32/48/56 tile-cols per chip), the gate.
- tests/ttnn/nightly/unit_tests/operations/fused/test_fused_rms_norm_prefill.py: bf16 input, broadcast bf16 weight, no bias/rope, num_links=1, Linear, 1x4 mesh.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp: derive_worker_cap / pick_num_workers_tp_gt_1 (one worker per tile-row, so 20 workers + 1 forwarder);
  compute_sizing (stats buffer depends only on num_workers -> max_rounds, so a column split that keeps the leader count leaves it unchanged); CB sizing; kernel CT/RT arg layout; override_runtime_arguments.
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp: input tile index = row*num_tile_cols, broadcast weight face-row reads, per-block barriers.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: stick push to forwarder packet slot, go-sem wait, gathered-stick DRAM read, output drain index math.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp: PRE (mul x*x + l1-acc + reduce), transpose stat to row 0, ring-sum via add_tiles + transpose_dest, 1/H_full constant from num_tile_cols*32*ring, POST sub-phases.
- ttnn/.../dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp (read-only, outside allowed_paths): forwarder protocol. It waits for present_count arrivals, mcasts pc*stick_bytes, and ups go only on the first pc worker coords. It can't take more sticks per packet than the fabric payload allows.
- tt_metal/fabric/erisc_datamover_builder.hpp: default fabric payload 4352 B, so 34 RMS sticks per packet. This is why followers can't push their own sticks and why the leader-combine is needed.
- tt_metal/hw/inc/api/compute/eltwise_binary.h: add_reuse_dest_init / add_reuse_dest_tiles<DEST_TO_SRCA> for the leader's in-DST peer sum.

## Nodes consulted
- none: round 1, history empty (only dream/rmsnorm-prefill/root).

## Profiling
- $DREAM_HOME/rmsnorm-prefill/reports/baseline_1 profile_log_device.csv: per-zone timeline (R_INPUT, W_PUSH, F_COLLECT, F_FABRIC, W_AGWAIT, W_DRAIN), table in proposal.md.
  POST is ~200 ns/tile/core and read+PRE ~130 ns/tile/core on 20 cores, with 100 cores idle.

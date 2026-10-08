## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml: job definition, allowed paths, gate.
- $DREAM_HOME/rmsnorm-prefill/history.md: index of all 55 nodes.
- Reflections of all 55 nodes (dumped via `git show <tag>:<node>/reflection.md`).
- r05-b01-a01/analysis/{waves.py,fwd.py,*_out.txt}: per-wave timeline of the root. The A/B drain overlap
  (A 9.05-14.21, B 11.31-15.59 at h7168) and the aggregate-bound write window are the basis of this proposal.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp: `compute_sizing` (wave_slots / wave_span / page
  size), the worker -> (wave, slot, row, half) lambdas, reader/writer RT args, forwarder CT/RT args.
- device/dit_fused_distributed_rmsnorm_device_operation_types.hpp: sizing struct.
- kernels/dataflow/dit_rmsnorm_wave_forwarder.cpp: 16-bit per-wave arrival/out_ready fields, F_SEND/F_GO loop.
  It assumed group = 2 x wave_slots.
- kernels/dataflow/dit_rmsnorm_fused_reader.cpp: start_sem gate (wave role, single partner, lead 2 blocks).
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: stick_off / pair_off / arrival_inc are RT args, so no
  change is needed for an uneven split.
- kernels/compute/dit_rmsnorm_fused_compute.cpp: gathered tile view is relative to the pair read, so it is
  slot-independent; no change needed.
- ttnn/cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp: fused_write_atomic... (local write + fwd/bwd
  fabric sends).
- tests/.../test_fused_rms_norm_prefill.py: seq_len 640 = 20 tile-rows for every shape.

## Nodes consulted
- r05-b01-a01 (root): equal 10/10 two-wave split; its timeline shows the overlapping drains.
- r05-b03-a01 #3: suggested an uneven split (B smaller). I argue for A smaller.
- r05-b01-a02/a03, r05-b03-a02/a03: 4 waves lose (serial go releases, drain overlap, chain cost). So stay at 2 waves.
- r03-b04-a02: waves of 10 cores are per-core bound. Each wave here keeps 18-22 cores.
- r04-b04-a01, r03-b02-a03, r03-b03-a03, r04-b03-a03: the drain is aggregate/path-bound, not issue/VC/bank bound.
- r05-b04-a01: the ~2.3 µs fabric round is a floor.

## Docs / external references
- none

## After eval
- reports/r06-b03-a01: waves.py / fwd.py outputs in analysis/. Per-device kernel means come from eval/ops.csv. The
  parent's per-device numbers are from its reflection (its ops.csv is not committed).

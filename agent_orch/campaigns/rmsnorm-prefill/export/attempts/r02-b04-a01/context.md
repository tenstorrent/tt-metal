## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml: process, allowed paths (whole op dir), metric.
- history.md, and the reflection of all 16 r01 nodes: what was tried, why the drain and read tails remain.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp: worker/forwarder placement (row-major
  `all_cores_vec`), per-core RT args (forwarder coords per worker, worker coords per forwarder), CB sizing
  (output_cb = 2 padded rows), override_runtime_arguments (no core lookups).
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: stick push to the forwarder by RT coords, gamma on BRISC,
  per-block drain.
- kernels/compute/dit_rmsnorm_fused_compute.cpp: PRE (DST-accumulated x^2), x*gamma under the AG, single POST pass.
- dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp (read-only): go-sem is per-worker unicast
  from RT coords, so any placement works.
- tt_metal/soc_descriptors/blackhole_140_arch.yaml: DRAM channels at NoC0 x=0 (ch0-3) and x=9 (ch4-7), endpoints
  per NoC.
## Nodes consulted
- r01-b04-a04 (round root): per-core zone table from reports/r01-b04-a04, which shows the x/y gradient of the read
  and drain ends.
- r01-b02-a04 and r01-b03-a04: dual-NoC drain flips the gradient. Placement was suggested and not tried.
- r01-b03-a03: bank de-phasing fixes the read hot spot only for bank-aligned slices. The drain is link-bound.
- r01-b03-a01 and a02: dispatch cost from multiple kernel groups. Keep the worker set as few CoreRanges.
## Docs / external references
- /tmp/r01b04a03/zones.py (zone timeline logic), reused per core.

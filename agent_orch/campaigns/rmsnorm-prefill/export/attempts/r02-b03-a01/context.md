## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job definition, allowed paths, metric.
- ledger/rounds/r02/manifest.json — round root is r01-b04-a04 (a89c280e84a).
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — W_GAMMA on BRISC, stick push, AG wait, per-block drain.
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp — CT-arg layout (weight_from_writer last), reader ends after input.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE (DST accumulate), x*gamma under AG, single POST pass pushes
  block_size-padded blocks to output_cb.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — worker cores = first 20 row-major cores; output_cb = 2 padded
  rows (resident); common-arg layout + override_runtime_arguments.
- tt_metal/soc_descriptors/blackhole_140_arch.yaml — DRAM columns x=0 (ch 0-3) and x=9 (ch 4-7), per-NoC worker endpoints.
- tt_metal/impl/device/firmware/risc_firmware_initializer.cpp — dram_bank_to_noc_xy uses get_preferred_worker_core_for_dram_view per NoC.
- umd blackhole_coordinate_manager.cpp — translated coords compact harvested tensix columns; DRAM translated x=17/18.
- tt_metal/hw/inc/internal/tt-1xx/risc_common.h — my_x/my_y come from NOC_ID_LOGICAL (translated), so physical
  coords must come from NOC_NODE_ID.
- reports/r02_root_1 profile_log_device.csv — worker cores are physical (x in 1..7,10..15, y=2,3); harvesting differs per chip;
  JIT compile line shows NUM_DRAM_BANKS=8 (no DRAM harvesting).
## Nodes consulted
- all 16 r01 nodes (reflections): drain is the tail on every 20-core node; PRE tail is fixed overhead (r01-b04-a04).
- r01-b02-a04 — position-blind odd/even block split; its diff (drain_sem protocol) reused; its regression explained by
  block parity == bank column (noc_drain_sim.py).
- r01-b03-a04 — per-position parity split on 80 cores, mirrored gradient.
- r01-b03-a03 — bank de-phasing: drain not bank-bound.
## Docs / external references
- noc_drain_sim.py (this dir) — torus link-load model of the drain for NoC0-only / parity / bank-side / shortest-path.
- drain.py (this dir) — per-core W_DRAIN/R_DRAIN end vs AG end and TRISC end from profile_log_device.csv (used for the reflection).

## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — rules, shapes, metric, allowed paths.
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp — input read (per 4-tile block barrier) and broadcast gamma face-row read index on col_start + col.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — W_DRAIN writes position col_tile+i to column col_start+col_tile+i.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — col_splits choice, per-core RT args, slice_start, eligibility.
- tests/ttnn/nightly/unit_tests/operations/fused/test_fused_rms_norm_prefill.py — input/weight/output are DRAM interleaved bf16 tiles.
- reports/r01-b03-a02/.logs/profile_log_device.csv via /tmp/percore_r01b03a02.py — per-core zones, h7168 drain tail grows with core x/y.

## Nodes consulted
- All 8 nodes (b01-a01/a02, b02-a01/a02, b03-a01/a02, b04-a01/a02): proposals, reflections, summaries.
- r01-b03-a02 (parent) — drain-bound, h4096 outlier, bank-camping hypothesis.
- r01-b02-a02 — drain tail is contention, not flush serialisation.
- r01-b04-a02 — gamma face-row reads are a same-bank hot spot.

## Docs / external references
- tt-metal interleaved buffer layout: page p -> DRAM bank p % num_banks.
- /tmp/r01b03a03/sim.py — bank collision simulation (original vs greedy rotation).

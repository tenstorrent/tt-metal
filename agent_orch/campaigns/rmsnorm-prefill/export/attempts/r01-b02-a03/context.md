## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/rmsnorm-prefill/campaign.yaml — process, allowed paths, gate.
- kernels/dataflow/dit_rmsnorm_fused_reader.cpp — parent's deep trid input read (`read_input_row_deep`), then the
  per-block-barriered broadcast weight read (col_offset page base from b02-a01). Replaced the weight read with
  b01-a01's single-barrier batch.
- kernels/compute/dit_rmsnorm_fused_compute.cpp — unchanged on this lineage vs root, so b01-a01's patch applied
  cleanly (`git diff n/r01-b01-a01~1 n/r01-b01-a01 -- <compute>`).
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — parent's deep drain waits cumulatively on
  `col_tile + block_size` and pops the padded row once; the new single POST pass pushes block_size per block, so it
  is compatible.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — weight_cb holds num_tile_cols (= weight_bcast_tiles),
  so a whole-row reserve/push fits.
## Nodes consulted
- r01-b01-a01 — source of the compute reorder and gamma batch (best, 1.1089); late gamma limited h6144/h7168.
- r01-b02-a02 (parent) — deep trid input read; recommends stacking x*gamma.
- r01-b01-a02, r01-b04-a02 — interleaving gamma reads with the input is a big regression; keep gamma after input.
- r01-b04-a01 — same compute idea; drain becomes the tail on wide shapes.
- r01-b02-a01, r01-b03-a01, r01-b03-a02 — column-split lineage (col_split=1 here; plumbing kept).
## Docs / external references
- none

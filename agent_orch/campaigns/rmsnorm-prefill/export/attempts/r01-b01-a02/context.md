## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job, allowed paths, accuracy gate.
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp — read_input_pass barriers every block_size tiles; a01's
  broadcast gamma batch is issued after the input row.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE waits input cumulatively per block; pre-AG x*gamma
  waits cb_weight cumulatively per block (whole-row push satisfies it).
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — resident input_cb = kInputCbChunks (2) whole rows,
  weight_cb = num_tile_cols; reader uses ReaderDataMovementConfig (its own NoC, separate from the writer).
- tt_metal/hw/inc/api/dataflow/noc.h — Noc::async_read<NocOptions::TXN_ID>(..., {.trid}) and
  async_read_barrier<TXN_ID>; per-trid outstanding throttle at 128.
- tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h — plain ncrisc_noc_fast_read does not touch
  NOC_PACKET_TAG (trid is sticky), per-trid flush = NIU_MST_REQS_OUTSTANDING_ID(trid)==0; BH trids 0..15.
## Nodes consulted
- r01-b01-a01 (parent) — reflection: R_INPUT latency bound (~7.5 us), gamma ~3 us late, drain tail.
- r01-b02-a01 — col split never engaged (34-stick packet cap); also flags the per-block read barrier.
- r01-b03-a01 — col split works but cross-chip launch skew eats it; avoid heavier dispatch.
- r01-b04-a01 — same compute reorder as a01; drain is the critical path at wide shapes.
## Docs / external references
- padded_slice reader (ttnn/.../padded_slice_reader_rm_interleaved_start_id.cpp) — existing trid read pattern.

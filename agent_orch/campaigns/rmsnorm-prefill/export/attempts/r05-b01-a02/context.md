## Files read
- `agent_orch/WORKER.md`, `agent_orch/campaigns/rmsnorm-prefill/campaign.yaml`: the job, the metric, allowed paths,
  the accuracy gate. Plus the 2026-10-08 rules: HiFi4 everywhere, no approx modes.
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp`:
  - `compute_sizing`: where the col-split capability, wave span and page size are decided, shared with
    create_stats_buffer and validate;
  - `create_at`: worker decomposition, wave/slot/row/half lambdas, gathered CB, start_sem, reader/writer/forwarder
    CT + RT args, compute CT 17 (stats_tiles_cols) and 43-45.
- `device/dit_fused_distributed_rmsnorm_device_operation_types.hpp`: the sizing struct.
- `kernels/dataflow/dit_rmsnorm_fused_reader.cpp`: the wave start_sem gate, the signal block
  (kWaveSignalLeadBlocks = 2), the trid-pipelined input read (lookahead 4 blocks, so a 7-14 tile row is issued at
  once).
- `kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`: the stick push (slot offset, f01 at +1024, arrival_inc), the
  streamed gamma loop (handles chunks < 8 pages through `rot % chunk_pages`), the post-go pair read, and the drain
  col_offset.
- `kernels/compute/dit_rmsnorm_fused_compute.cpp`: gathered_tile indexing, the pairwise add loop over
  stats_tiles_cols, `cb_stats_reduce_src.wait_front(chunk_stats_tiles)` (≤ gathered_cb_pages, fine at 16), and
  recip_h_full = 1/(num_tile_cols*32*stats_tiles_cols).
- `kernels/dataflow/dit_rmsnorm_wave_forwarder.cpp`: the parent's two-wave forwarder. The wave count and field width
  are the only places that assume 2.
- `r05-b01-a01/analysis/waves.py`, `waves_out.txt`, `fwd_out.txt`: the parent's per-wave timeline used for the model.

## Nodes consulted
- r05-b01-a01 (parent, 1.5696): two-wave k=2 split. Its timeline gives the model inputs: a fixed ~5 µs chain from
  read end to drain start, the total read unchanged at ~6.3 µs, and the last wave's drain overlapped. Its reflection #2
  suggests more waves.
- r05-b03-a01 (1.5174): the independent two-wave variant. It confirms the per-wave AG floor (~2.9-3.1 µs send -> go)
  and that the post-go gather/combine grows with partial count. Its #2 suggests col_split 4 / 4 waves of 20.
- r05-b04-a01 (1.1939): multi-round on 20 workers with the stock forwarder serialized the rounds. The lesson is that
  the AGs must overlap, which the per-wave fields here provide.
- r03-b04-a02: 10-core waves are per-core capped, so keep 20 cores per wave.
- r01-b03-a01/a02/a03: k=4 / 81 cores is dispatchable with one kernel group, and all-80-at-once drains collapse.
  Here only 20 drain at once.
- r04-b02-a01/a02 and r04-b02-a03: the 64 B-page gathered CB trick, and synchronized drains contend (release
  staggering helps).
- r04-b04-a03, r05-b02-a01: the bistable launch-skew state and the host-bound narrow shapes, to judge chip means by.
- All other r01-r04 reflections were read for the ruled-out list: VC, cmd-buf, bank de-phase, read depth, dual-NoC
  splits, placement, multicast release.

## Docs / external references
- None beyond the code.

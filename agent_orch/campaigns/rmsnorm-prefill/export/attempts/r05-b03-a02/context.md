## Files read
- `agent_orch/WORKER.md`, `campaign.yaml`, `$DREAM_HOME/rmsnorm-prefill/history.md`: the rules, the metric, and the
  51-node index. Campaign rule (2026-10-08): HiFi4 everywhere, no new approx mode. This node changes no fidelity or
  approx setting.
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp`:
  - `compute_sizing`: the wave decision lives here because it sizes the stats scratch (one page per wave).
  - the core allocation and wave mapping lambdas (`worker_wave`, `worker_tile_row`, `worker_col_offset`): already
    general in K.
  - the CB setup (gathered CB, packet CB), writer/forwarder/compute CT args, and per-worker RT args.
  - `override_runtime_arguments`: touches only common args and forwarder rt[0..1], so the RT-arg changes are safe.
- `kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`: stick push (packet slot, face-row offsets, arrival inc), the
  post-go gathered-stick read (parent: 2 x 64 B reads per partial), and the drain addressing.
- `kernels/dataflow/dit_rmsnorm_fused_reader.cpp`: the trid-pipelined input read, the wave start_sem wait/signal, and
  `kWaveLeadBlocks` (parent: "2 blocks left", which signals after block 0 on 2-block rows).
- `kernels/dataflow/dit_rmsnorm_wave_forwarder.cpp`: the 2-wave, 16-bit-field poll loop; it sends `need * 128` B per wave.
- `kernels/compute/dit_rmsnorm_fused_compute.cpp`: the packed-AG combine (pairwise ELWADD over stats_tiles_cols
  gathered tiles, fused row-0 add_rsqrt, transpose_dest) and the `1/(num_tile_cols*32*stats_tiles_cols)` scale, which
  stays 1/H_full for any K.
- `tests/ttnn/nightly/.../test_fused_rms_norm_prefill.py`: seq 640 -> 20 tile rows; local widths 896/1024/1536/1792
  -> 28/32/48/56 tile cols, all divisible by 4 (quarter rows of 7/8/12/14 tiles).

## Nodes consulted
- r05-b03-a01 (parent, 1.5174): the 2-wave pipeline on 40 half-row workers. Its waves table gave the per-wave timeline.
  Reflection #1 (slow post-go gather: 16 x 64 B reads) and #2 (more waves, but fix the gather first) set this node.
- r05-b01-a01 (1.5696, sibling best): same idea, plus the tile-row-0 slot layout `L(j) = (j/16)*2048 + (j%16)*64`
  with one pair read per chip and the 64 B-page gathered CB with offset tile views, all validated on HW. Its waves
  table (A read end 3.59, A drain start 9.05, B drain end 15.59 at h7168) is the source of the
  `R/N + C + W` model in the proposal.
- r04-b02-a01: the original HW validation of the slot layout and the 64 B-page gathered CB.
- r03-b04-a02: the 2-wave forwarder fork and 16-bit fields. Its lesson: 10-core waves are per-core capped, so every
  wave here keeps 20 cores.
- r01-b03-a01/a02/a03: an 80-worker k=4 split with one kernel group runs fine on this grid. They lost on a leader
  hop and on 80 concurrent drainers; neither exists here.
- r05-b04-a01 (stock-forwarder rounds, -14%): rounds must overlap their AGs. The wave forwarder already does that.
- r05-b02-a01, r04-b01-*: the PRE tail is fixed per core (~0.8 µs at HiFi4), so it is part of the per-wave chain C.
- Every other node's reflection, for the ruled-out levers: drain per-core knobs, bank de-phasing, read depth,
  multicast release, and dual-NoC variants.

## Docs / external references
- none beyond the code.

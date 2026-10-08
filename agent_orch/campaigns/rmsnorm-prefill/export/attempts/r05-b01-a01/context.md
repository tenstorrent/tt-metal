## Files read
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp`: compute_sizing (stats page geometry; shape-only),
  create_at (worker count / placement, CB sizing, the CT/RT arg layouts of reader, worker writer, forwarder and
  compute, the forwarder group RT args, and override_runtime_arguments only touching rt[0]/rt[1]).
- `device/dit_fused_distributed_rmsnorm_device_operation_types.hpp`: the sizing struct.
- `device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`: push_stick (ack-free), streamed gamma, the 8
  gathered face-row reads after go, and the posted drain with its out_idx mapping.
- `device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp`: trid-pipelined resident input pass, the input page
  index `tile_row * num_tile_cols`.
- `device/kernels/compute/dit_rmsnorm_fused_compute.cpp`: PRE (ELWMUL into DST + ones*S^T matmul), x*gamma
  pre-pass, combine (ELWADD over stats_tiles_cols gathered tiles, recip_h = 1/(num_tile_cols*32*stats_tiles_cols)),
  POST. Checked that the ragged last block (half width 14 at h3584) is block-padded everywhere.
- `dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp`: the forwarder protocol (the base of the
  fork). Also `ccl/common/kernels/minimal_ccl_common.hpp` fused_write_atomic... (local write + fwd/bwd fused
  write+inc, arbitrary payload size/offset).
- The test (not traced; host enqueues d0..d3 ~1 µs apart, so the chip mean includes launch skew).

## Nodes consulted
- All 47 nodes' proposal/reflection/score (dumped via git show). The ones that shaped this attempt:
- r03-b04-a02: two-wave AG with 10 full-row workers per wave. Flawed because of per-core caps (read 273 GB/s,
  PRE 56 tiles/core). Its reflection #4 is this node. I reuse its 16-bit per-wave semaphore fields, the reader
  start_sem gate and the forwarder poll loop.
- r04-b02-a01: slot tile-row-0 layout `L(s) = (s/16)*2048 + (s%16)*64`, a 64 B-page gathered CB and compute
  addressing tiles at 64 B offsets (all validated on HW). Here it makes a row's two halves one contiguous read.
- r01-b03-a01..a04, r01-b02-a01: earlier column splits. One kernel group is needed (equal slices). The 34-stick
  packet limit. A row-leader combine hop costs 1-2 µs. 80 concurrent writers collapse the drain.
- r02-b02-a02: the diag x*x^T matmul removes the HiFi4 ELWMUL backlog (stat ready 0.41-0.56 vs 0.67-0.83 µs
  after the read), but extracting the diagonal costs about as much. Considered and not pursued.
- r04-b04-a02 (root), r04-b04-a03, r04-b03-a03: the read is aggregate-bound (~400 GB/s), and per-core read depth
  and bank phase don't matter.
- r04-b01-a02/a03, r04-b02-a03: HiFi2 PRE, now forbidden by the campaign rule.

## Measurements made for this proposal (root report reports/r04-b04-a02)
- `pre.py` (stat ready − own R_INPUT end per core) over r02-b02-a01/a02/a03, r03-b03-a02, r04-b04-a02,
  r04-b01-a03.
- `ag.py` per chip, per call: F_COLLECT end vs the chip's start, F_FABRIC duration. The min over chips is
  ~1.2-2 µs. F_COLLECT end relative to the chip's own start is the same on every chip.
- ops CSV host timestamps: d0..d3 are enqueued ~1 µs apart. In skewed calls d3's op-to-op is 2-4 µs.

## Docs / external references
- none beyond the code.

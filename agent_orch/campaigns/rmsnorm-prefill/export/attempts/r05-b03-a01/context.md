## Files read
- `agent_orch/WORKER.md`, `campaign.yaml`, `$DREAM_HOME/rmsnorm-prefill/history.md`: the rules, the metric, and the leaderboard. HiFi2 nodes are now forbidden_edit (the fidelity rule).
- Every node's reflection (all 47, via `git show <tag>:.../reflection.md`). They show that the read is DRAM-aggregate-bound at
  20 cores (r04-b03-a03, r04-b04-a03), that per-core drain knobs are exhausted (r03-b02-a03, r03-b03-a03, r04-b04-a01, r04-b03-a03),
  and that waves of 10 full-row cores fail on per-core rates (r03-b04-a02, whose #4 suggests k=2 + 2 waves of 20).
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp` (whole file): sizing (`compute_sizing`, `derive_worker_cap`),
  worker decomposition, CB sizing, CT/RT arg layouts, the forwarder args, and override_runtime_arguments.
- `device/dit_fused_distributed_rmsnorm_device_operation.cpp` (validate / compute_output_specs): the stats buffer shape
  must exactly equal `make_stats_tensor_spec(compute_sizing(...))`, so the wave decision has to live in `compute_sizing`.
- `kernels/compute/dit_rmsnorm_fused_compute.cpp` PRE / x*gamma / combine / POST: the combine sums `stats_tiles_cols`
  gathered tiles pairwise and scales by `1/(num_tile_cols*32*stats_tiles_cols)`. So passing W/2 and 2*ring keeps it
  exact, with no compute change. Tail blocks (14 = 4+4+4+2) are handled by `tiles_in_block`.
- `kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`, `dit_rmsnorm_fused_reader.cpp`: stick push, gathered-stick
  read, gamma stream, drain addressing, and the trid-pipelined input read.
- `dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp` (read-only) and r03-b04-a02's
  `dit_rmsnorm_wave_forwarder.cpp` (via git show): the fork base for the per-wave 16-bit fields.
- `ttnn/cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp`: the fused write+atomic helper. The header is
  flushed before return, so back-to-back wave sends can reuse it.
- `tests/ttnn/nightly/.../test_fused_rms_norm_prefill.py`: eager loop, 3 warmup + 10 measured, ping-pong semaphores and stats buffers.

## Nodes consulted
- r03-b04-a02: two waves of 10 full-row cores, -5.3%. Its plumbing (wave forwarder, reader start_sem gate, 16-bit
  fields) is reused; its failure mode (per-core rates) is what the column split addresses.
- r01-b03-a01/a02/a03: column split k=3/4. Lessons: one kernel group (equal slices), and the leader-combine hop is
  expensive. Here the partial sticks are summed in the existing post-AG combine instead.
- r04-b04-a02 (parent/root): its `tl.py` timeline is the cost model (read 3.2-6.4 µs, F_FABRIC end -> C_POST 1.6 µs,
  drain ~93 ns/tile/core).
- r04-b04-a03: launch-skew analysis. `analysis/startgap.py` here shows that each core restarts a fixed ~2.0 µs after
  its own previous kernel end (profiler/FW turnaround), and the next go arrives ~0.5 µs after the last core ends.
  So the "bistable skew" is the previous call's per-core end spread. I considered an end-of-kernel barrier and
  rejected it: it would cut the measured kernel time without making the op faster (a profiler artifact).
- r04-b01-a02/a03, r04-b02-a03: HiFi2 PRE wins, now forbidden. The root is the HiFi4 r04-b04-a02.

## Docs / external references
- none beyond the code.

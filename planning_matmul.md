# Planning: Matmul Beyond DeepSeek

## Objective

Improve matmul efficiency in shared paths that can benefit multiple models on non-Galaxy hardware (N150/T3K), using the DeepSeek N150 finding as a clue:
- Short-M prefill shapes can improve when compute grid Y is right-sized instead of always using full grid height.

## Current Status

- [x] Baseline insight captured from N150 DeepSeek experiments
- [x] Experiment 1: Shared TTNN auto-config grid right-sizing (C++ matmul config path)
- [x] Experiment 1 validation on N150 (model-traced matmul protocol)
- [x] Experiment 2: Common MLP1D prefill config right-sizing (shared Python module)
- [x] Experiment 2 validation on N150 (targeted benchmark/protocol)
- [ ] Optional quick T3K sanity workflow recommendation update
- [x] Summarize findings + keep/revert decision per experiment

## Experiment 1 Plan (Shared C++ auto path)

- [x] Update `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp`
  - Scope: `create_simple_matmul_program_config(...)`
  - Change: cap active `compute_with_storage_grid_size` by work tiles (`num_blocks_x`, `num_blocks_y`) for all-DRAM-interleaved 2D mcast path.
  - Goal: avoid over-parallelizing short-M/N shapes.
- [x] Validate no obvious correctness/perf regressions on N150 with existing benchmark protocol subset.

## Experiment 2 Plan (Shared Python MLP path)

- [x] Update `models/common/modules/mlp/mlp_1d.py`
  - Scope: default prefill program-config builders (`prefill_w1_w3_prg_config`, `prefill_w2_prg_config`)
  - Change: derive an effective grid per `seq_len` bounded by tile demand.
  - Goal: bring same right-sizing behavior to non-DeepSeek shared MLP path.
- [x] Fast unit sanity for shared MLP module (`test_mlp_1d_config_*`)
- [x] Device perf/correctness validation for shared MLP path on N150/T3K
- [x] Validate on N150 with targeted run(s) and compare to pre-change baseline.

## Idea Backlog (Next)

- Add stricter no-regression benchmark policy for new experiments:
  - require both `overall_p50` and `overall_p95` to decrease across repeated runs.
  - auto-revert experiment candidates that fail this gate.
- Add a separate exploration track (non-merge gate):
  - preserve mixed but informative candidates in an explicit scoreboard.
  - keep strict `p50+p95` gate only for merge decisions.
- Explore memory-traffic bottlenecks around matmul boundaries:
  - reduce unnecessary DRAM round-trips for intermediate tensors.
  - prefer scoped, shape-gated changes with safe fallbacks.
- Expand coverage with additional N150/T3K-safe shape families before broad rollout.
- Add regime-aware kernel microbench workflow:
  - maintain a fixed small regime manifest (6-10 shapes).
  - use low-noise microbench runs to screen candidates before protocol A/B.
- Add alternating-run variance controls:
  - baseline/candidate interleaving in fresh processes for A/B fairness.
  - track both per-run and per-shape variance.

## Notes

- Keep changes narrowly scoped; avoid model-specific hand-tuning.
- Prefer improvements that can benefit generic `ttnn.matmul` / `ttnn.linear` usage.

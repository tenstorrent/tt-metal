# Minimal-host solve experiments — plan & autonomous decisions

Overnight autonomous run. Goal: compare ways to make the inter-mesh minimal-host solve faster, verify each
lands on the minimal host count, across shapes / stage counts / host sizes. Greedy-only is a required arm.

## Key finding that shaped this (2x4 64-stage baseline)
The **soft descent** (shave one host at a time, 24→16) is the bottleneck — it times out (>250s). The warm_solve
itself is ~5ms. So the experiments target the descent, not the warm_solve.

## Experiments (branches off checkpoint cc0e88940e1; selected at runtime by `TT_TOPO_SAT_MIN_MODE`)
- **baseline** (`exp/baseline`, MIN_MODE=0): warm + soft descent + hard-cap lock (original).
- **skip-descent** (`exp/skip-descent`, MIN_MODE=1): warm + hard-cap lock only; jump straight to k_min.
- **greedy-only** (`exp/greedy-only`, MIN_MODE=2): constructive host-fill (backtracking DFS, pruned to ≤ k_min
  groups, host-fill heuristic) + verify by unit-propagation. No SAT search. Fails cleanly if it can't construct
  (never emits a wrong mapping).

Autonomous decision: all three modes live in one build behind the env flag (efficient — one build, no per-arm
rebuild), and each is also a git branch for isolation/reproducibility. The original is preserved (MIN_MODE=0 ==
checkpoint behavior).

## Matrix
- Shapes/stages: 2x4 {16,32,64,80,96,112,128,144}, 4x4 {8,16,32,40,48,56,64,72}.
- Host sizes (mocks): SC36 (36 hosts / 1152 ASIC), SC20 (20 / 640), SC16 (16 / 512).
- Filter: only run (mock, stage) where the MGD's ASIC count fits the mock (2x4=8·stage, 4x4=16·stage).
- Enumeration ON (`-a -n 20`) so it also finds more solutions; 150s per-run wall cap.
- Order: modes 2 (greedy) and 1 (skip-descent) first (fast), baseline last (times out on hard cases).

## Metrics per run (RESULTS.tsv)
mode, shape, stage, mock, k_min (arithmetic min hosts), success/TIMEOUT, greedy_ok, occupied (hosts landed),
intermesh_ms, wall_ms, solutions_found.

## Other speedup ideas noted (recommendations, not all run)
- **greedy → skip-descent fallback** (practical best): the table already implies it — use greedy where
  `greedy_ok`, else the skip-descent number. A 4th mode could encode this.
- Warm-start the hard-cap SAT with a greedy phase-hint (for cases greedy can't fully construct).

## Reproduce a single run
`TT_TOPO_SAT_MIN_MODE=<0|1|2>` in the producer env; direct 36/20/16-rank `generate_rank_bindings` on the
sweep MGD + mock (see scratchpad/run_one.sh). Debug profile needs `-DTT_METAL_ENABLE_LOGGING=ON` + `TT_LOGGER_LEVEL=debug`.

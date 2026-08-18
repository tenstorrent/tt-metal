# segfusion1 — segmented-fusion / valid_length-rebalance / routing-boundary evidence (2026-08-18)

Per-cell Tracy device-kernel sums (33 or 13 iters, steady-state opcodes only), p150a,
measured during the #53457/#53464 review cycle against the PR-stack build
(nkapre/topk-large-indices-multicore + nkapre/topk-large-k-routing heads) vs origin/main.

- width_sweep_main.txt / width_sweep_ours.txt: 160-row 5k..1M sweep, k=1536/2048,
  main vs pre-segmentation branch (ours = trees+hybrid+fused<=32).
- width_sweep_seg.txt: same cells after segmented fusion (+ 640-row tie-ins).
- pavle_cell_results.txt: the GLM 5.2 DSA-indexer calling shape (1M preallocated,
  valid_length=512k, k=2048, 160 rows) — main 5774.7us vs branch 2932.1us (1.97x)
  after tree-slice valid_length rebalancing.
- route_cells_results.txt: routing-boundary cells (pow2-4096 38x/70x; W=128 k=96
  5.9x; MoE gate at SHIPPED k_rounded=32 config: W=512 k=10 routed 8.0 vs stock
  87.9us), stock arm forced via sub_core_grids.
- mutation_results2.txt: LLK fused-e2e test sensitivity controls (wrong chunk-id
  stamp -> 9 cells fail; dropped segment base -> 9 cells fail; pristine green).

Benches: tests/ttnn/nightly/.../experimental/_topk_li_width_sweep.py and
tests/ttnn/unit_tests/operations/reduce/_topk_route_cells_bench.py (merged with #53457/#53464).

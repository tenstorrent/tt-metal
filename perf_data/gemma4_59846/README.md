# Issue #59846 data: Gemma4 ring SDPA utilization vs prefill chunk size

Work-in-progress data, not for merge. Drop this directory before opening a PR.

Measured 2026-10-09/10 on bh-glx-120-b09u08 (Blackhole galaxy, high-power host), 8x4 mesh, SP (CP) = 8, TP = 4,
110 SDPA cores per chip (11x10 grid, CCL column excluded), Linear topology. Branch
`mmanzoor/gemma4-sdpa-small-chunk-sweep` = main 6ba9d172b63 + PR #59484's test changes (c111f4d4801, cherry-picked,
not merged upstream) + the sweep tests (1b930d75ffb). C = request chunk before CP sharding; c = C/8 Q rows per rank.

## Files

- `global_sweep.csv`: global attention, one row per (C, prefix, tuning); 1205 rows. C in {256, 512, 1024, 2048, 4096},
  prefix in {0, 16K, 64K, 128K, final chunk of 256K}; Q chunk {32, 64, 96, 128} (<= c), K chunk {128, 256, 384, 512},
  max_k_splits {1-4}, segmented on/off, deduplicated on what the op runs, plus production's tuning. 6 calls per row,
  the first dropped as warm-up; median/min/max device ns (realtime profiler, max over the 32 chips); util = the
  PR's metric (rectangle + causal triangle FLOPs, 110 cores, LoFi peak = 2x the HiFi2 model). `reason` holds the
  op's rejection.
- `global_summary.txt`: `analyze.py` output (best vs production per point, rejections).
- `global_accuracy.csv`: best and production tuning per C, first and last chunk of a 64K sequence, packed K/V, LoFi,
  K/V mean offset 1.0, gate PCC >= 0.99 and RMSE < 0.05.
- `swa_table.csv`: sliding-window attention per C and (Q, K) chunk: `accuracy` (prefix 0, C, and >= 4K on a
  production-style oversized, garbage-padded cache; PCC gate 0.99 at Gemma's scale 1.0, RMSE recorded) and
  `perf_final` (final 256K chunk, 6 calls, median).
- `run_*.sh`, `analyze.py`: exact commands (paths assume the original box; `source activate` sets TT_METAL_CACHE).
- Raw data (not in git; the large-file hook caps files at 500 KB): JSON-lines with every per-call duration and full
  error text, all pytest logs, the invalid first sliding run (`swa_sweep_v1_bad_reference.*`) and the PR #59484
  perf check log are in `/data/mmanzoor/sdpa_chunk_sweep/bundle_59846.tar.gz` on the NFS share 10.32.13.1:/data
  (also unpacked in that folder). A git bundle of this branch is next to it: `branch_59846.bundle`.

Reproduce (each pytest item opens the full 8x4 mesh):

    GEMMA4_SDPA_SWEEP_OUT=<out>.jsonl pytest -sv \
        "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_gemma4_global_chunk_sweep"
    GEMMA4_SDPA_SWEEP_OUT=<out>.jsonl pytest -sv \
        "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_gemma4_swa_chunk_sweep"
    pytest -sv "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_gemma4_global_chunk_sweep_accuracy"
    python3 analyze.py <out>.jsonl

Global: ~4 min per (C, prefix) item, 25 items. Sliding: 60 items, ~45 min. Accuracy: 18 items, ~35 min.

## Findings

Global attention, final 256K chunk (best tuning / production tuning on main):

| C | c | best tuning | best ms | best util | production tuning | prod ms | prod util |
|---|---|---|---|---|---|---|---|
| 256 | 32 | q32/k512/ksplit4 | 4.87 | 4.6% | q32/k256/ksplit3/seg | 5.51 | 4.1% |
| 512 | 64 | q64/k256/ksplit4 | 4.91 | 9.2% | q32/k256/ksplit3/seg | 5.55 | 8.1% |
| 1024 | 128 | q64/k384/ksplit4 | 5.02 | 18.0% | q32/k256/ksplit3/seg | 5.69 | 15.9% |
| 2048 | 256 | q64/k384/ksplit3 | 5.55 | 32.4% | q64/k256/ksplit3/seg | 5.74 | 31.4% |
| 4096 | 512 | q128/k256/ksplit3/seg | 7.50 | 47.8% | (same) | 7.50 | 47.8% |

- 50% is not reached at any C <= 4096. Warm repeats spread < 1.5% at the final chunk.
- The final-chunk op time is ~5 ms at every C <= 2048: each chunk streams the whole prefix, so utilization scales
  with C. At small C there are too few (head, Q chunk) units: C=256 has 8, so at most 4 bands x 8 = 32 of 110 cores
  work. Main caps K split at 4 bands (kKSplitMaxCount).
- Larger tunings run out of L1 (CBs 1543-2049 KB > 1536 KB): q64/k512, q96/k384, q96/k512, q128/k384, q128/k512.
- Production is 5-13% slower than the best tuning at C <= 1024 (always q32 + 3 bands); within 3% at 2048; best at 4096.
- Prefix length: utilization rises with the prefix and saturates by 128K at every C; the ranking of C does not change.
  Prefix 0 is ~0.3-0.45 ms with up to 30% spread, which says little.
- PR #59484's reference: its 4096 final chunk 50.1% / 7.154 ms did not reproduce here (47.8% / 7.50 ms warm median;
  48.2% / 7.446 ms in a single-shot run of the PR's own check). Not investigated.

Correctness (`global_accuracy.csv`): all 9 last-chunk cases pass (PCC 0.9990-0.9991, RMSE <= 0.036). All 9 first-chunk
cases pass accuracy (PCC 0.9998, RMSE ~0.02) but fail the runner's "output equals the run without K split ... never
applied" check; reproduced identically on a repeat. Cause is the harness: chunk 0 gets a cache sized to the chunk, so
`is_chunked()` (Q rows < local K/V rows) is false, the kernel runs non-chunked causal and the factory disables K split.
Production's full-size cache keeps chunk 0 chunked. Proposed fix: skip the feature check for the first-chunk position.

Sliding-window attention (`swa_table.csv`): main accepts only Q chunk 64/128 with K chunk 128, and both must divide c.
So C=256 and C=512 cannot run sliding attention at all. C=1024/2048/4096 with q64 or q128 (production) and k128 pass:
PCC ~0.9965 at prefix 0, C and >= 4K; final chunk 312-408 us. RMSE ~0.08 comes from scale 1.0 on random N(0, 1)
inputs (0.004 at 1/sqrt(d)). Only C = 1024, 2048, 4096 support both attention types; of those only 4096 is near 50%.

Harness fix in 1b930d75ffb: `run_ring_joint_sdpa_chunked`'s CPU reference ignored `model.scale`, so the Gemma perf
configs (scale 1.0) were checked against 1/sqrt(d) (sliding PCC 0.38). Worth its own PR.

## Not done yet

1. First-chunk K-split feature check fix (above), then rerun the 9 first-chunk accuracy cases.
2. Why 4096 gives 47.8% here vs the PR's 50.1%.
3. Optional: rerun on the `kmabee/gemma4-matmul-dig-base` kernels (up to 8 K-split bands, q128 + bfp8 Q for 2K).
4. Utilization/latency plot, write-up and the issue comment (nothing posted to #59846 yet).

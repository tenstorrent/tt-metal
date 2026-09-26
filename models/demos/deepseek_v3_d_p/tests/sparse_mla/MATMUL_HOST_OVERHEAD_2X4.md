# Matmul host overhead on GLM-5.2 sparse MLA, 2×4 LoudBox

## Workload and method

This compares `origin/main` at `d7618ac9981` with rebased `pjosipovic/matmul-host-ops` at `b8ac11819a1` before this report was added. Both use Release builds and the same GLM-5.2 scaled-FP8 warm sparse-MLA path from `test_sparse_mla_perf.py`. That path uses the ttMLA implementation, GLM configuration, and scaled-FP8 cache helpers covered by `test_sparse_mla.py`; its warm prefix shape differs from the 5,120-token correctness case. On this 8-chip LoudBox it uses a 2×4 mesh with Fabric2D, a 1,280-token chunk at offset 12,800, TP-sharded scaled-FP8 KV and index caches, and the last full indexer layer. Warm-mode cache contents start zeroed; the allocated prefix and operation shapes match the 12,800-token cached-prefix proxy.

For each revision, run 10 untraced cached forwards as warmup, then time 10 more forwards from host call through `ttnn.synchronize_device(mesh_device)`. The **minimum** of those 10 E2E samples is the reported result. A separate 10-forward pass wraps each `ttnn.matmul` and `ttnn.linear` call with `time.perf_counter_ns()` and records each instance's minimum host-call duration. It excludes device completion from each operation timer. The wrappers are absent from E2E timing. These are inclusive host API times and may include internal waits; their independently selected minima should not be added to reconstruct E2E.

After the untraced passes, capture the same forward as three segmented traces. Exclude capture and compilation, replay 10 times as warmup, then time 10 replay-plus-sync intervals and take their minimum. The traced and untraced E2E timers have the same host start/end points around execution and synchronization. Their difference includes dispatch and scheduling effects as well as any change in device overlap; it is not a direct measurement of CPU work alone.

Reproduce on an 8-chip LoudBox from either revision's checkout with its matching Release build:

```bash
TT_METAL_HOME=$PWD PYTHONPATH=$PWD/ttnn:$PWD DS_PERF_UNTRACED_HOST=1 \
  scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py \
  -m perf -k 'glm_5_2 and warm and scaled_fp8' -q -s
```

Use a Python environment with this repository's test dependencies. In an isolated worktree, activate a shared environment or provide a `python_env` link before invoking the safe pytest runner.

The harness writes all samples under `generated/profiler/glm52_untraced_matmul_host/`. The first four untraced runs were ordered branch → main → main → branch to reduce order bias. A further branch → main pair measured the traced gap. All six runs passed.

## E2E result

| Paired run | Main minimum | Branch minimum | Branch change |
| --- | ---: | ---: | ---: |
| 1 | 4.107 ms | 4.077 ms | −30.1 µs (−0.73%) |
| 2 | 4.101 ms | 4.108 ms | +6.9 µs (+0.17%) |
| 3, gap comparison | 4.117 ms | 4.044 ms | −72.2 µs (−1.75%) |

The direction changed between pairs, and individual 10-sample sets spanned roughly 64–108 µs in the first two pairs and 73–95 µs in the third. **This workload does not show a reliable E2E speedup** from the two matmul host fixes at this measurement resolution.

## Current traced versus untraced gap

The third pair used identical 10-warmup/10-measured minima for both execution modes. Trace capture and first replay are excluded.

| Revision | Untraced minimum | Traced minimum | Gap | Gap / untraced |
| --- | ---: | ---: | ---: | ---: |
| `origin/main` | 4.117 ms | 3.023 ms | 1.094 ms | 26.6% |
| `matmul-host-ops` | 4.044 ms | 3.008 ms | 1.037 ms | 25.6% |

The current branch therefore retains about **1.04 ms** of traced/untraced E2E gap for this warm GLM-5.2 FP8 layer on the LoudBox. The traced minima differ by only 15 µs between revisions; that is within the observed run-to-run spread and should not be attributed to the matmul changes.

## Host call duration by operation instance

The forward invokes two `ttnn.matmul` and nine `ttnn.linear` calls. Instance numbers are in call order within each operation type. Values below are the per-instance minima in µs from the separate host-call pass in the first two pairs.

| Instance | Main 1 | Branch 1 | Main 2 | Branch 2 |
| --- | ---: | ---: | ---: | ---: |
| linear 1 | 58.3 | 55.8 | 57.5 | 57.5 |
| linear 2 | 36.3 | 35.4 | 36.4 | 36.6 |
| matmul 1 | 38.7 | 38.5 | 38.7 | 38.9 |
| linear 3 | 37.8 | 38.3 | 37.6 | 37.9 |
| matmul 2 | 39.2 | 42.0 | 39.3 | 38.7 |
| linear 4 | 31.3 | 32.3 | 32.3 | 32.0 |
| linear 5 | 43.0 | 44.4 | 43.3 | 43.9 |
| linear 6 | 34.7 | 35.4 | 36.2 | 35.8 |
| linear 7 | 38.0 | 38.0 | 38.2 | 39.1 |
| linear 8 | 39.7 | 41.7 | 40.7 | 41.2 |
| linear 9 | 35.8 | 36.3 | 36.5 | 36.4 |

No instance shows a consistent reduction across both pairs. The 2D-multicast cached-core-list change only helps launches using that factory; this table does not isolate or establish use of that factory.

## Validation

- `./build_metal.sh --release` passed on the rebased branch.
- `scripts/run_safe_pytest.sh` passed 17 matmul attribute/cache cases, including trace replay, and both 2D-multicast fresh-buffer cache cases.
- The GLM-5.2 scaled-FP8 correctness case in `test_sparse_mla.py` passed on 2×4: output PCC 0.994416, KV cache PCC 0.999529, and PE cache PCC 0.999916.

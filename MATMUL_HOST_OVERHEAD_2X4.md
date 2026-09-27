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

## Where the remaining gap goes

A follow-up diagnostic on the rebased branch (`2e618c20fcd`, 2026-09-27) used the same 2×4 workload and 10-warmup/10-measured protocol, with `DS_PERF_HOST_BREAKDOWN=1`. It split the clean E2E timer at the return from `mla.forward`, then used separate instrumented passes to time TTNN calls and trace replay phases. All figures below are medians; the E2E minimum in this run was 4.130 ms untraced versus 3.004 ms traced.

| Phase | Untraced | Traced |
| --- | ---: | ---: |
| Host forward / trace submissions and manager work | 2.609 ms | ~0.043 ms |
| Final device synchronization | 1.546 ms | 2.913 ms |
| Complete E2E | 4.150 ms | 3.055 ms |

The traced phase numbers come from a separate replay pass: its three `execute_trace` calls took 22 µs combined, other replay work including two manager transitions took 21 µs, and the final synchronization took 2.913 ms. The longer traced wait means that much of eager's host work overlaps device execution. **Inference:** roughly 1.1 ms of eager dispatch pacing remains exposed on the E2E critical path; the full 2.6 ms host-forward interval is not added on top of device time.

An instrumented eager pass counted **67 TTNN calls** per forward. Their inclusive host-call times summed to 2.216 ms median, with another 0.518 ms in Python/model control and other work outside those calls in that instrumented pass. The largest groups were:

| Host call group | Calls | Median combined time |
| --- | ---: | ---: |
| `linear` | 9 | 382 µs |
| `reduce_scatter_minimal_async` | 4 | 311 µs |
| `rotary_embedding_indexed` | 4 | 217 µs |
| `deallocate` | 20 | 180 µs |
| `high_bw_all_gather` | 5 | 170 µs |
| `to_layout` | 4 | 136 µs |
| `all_to_all_async_generic` | 2 | 98 µs |
| `matmul` | 2 | 82 µs |

Matmul plus linear account for about 0.46 ms of host-call time, and other calls plus model control account for more than 2 ms. These call times are measured in a separate wrapper pass and include any waits inside a call; they identify places to investigate, not additive contributions to the 1.1 ms E2E gap. Device realtime-profiler timestamps are per program/core and cannot be combined into a trustworthy cross-core E2E span, so this analysis uses host intervals for the traced/untraced comparison.

## Tracy host and device timeline

On 2026-09-27, a follow-up captured the same 2×4 GLM-5.2 scaled-FP8 warm workload with the Blackhole streaming profiler feeding device kernel zones into Tracy. Tracy also captured the host's C++ zones. Optional `DS_PERF_TRACY_MARKERS=1` messages bracket each of the ten measured untraced and traced iterations. The capture used partial Python profiling (`python -m tracy -p`) to keep Python's per-call profiler out of the timing pass. Its trace is `build/profiler/build_wasm/traces/-s_2026_09_27_06_27_17.tracy` in the branch worktree.

| Aligned streaming capture | Result across ten untraced forwards |
| --- | ---: |
| Untraced E2E minimum / traced E2E minimum | 4.985 / 3.048 ms |
| Worker-kernel-free intervals per untraced forward, median | 597 µs |
| Of those intervals, inside host `EnqueueProgram` zones, median | 93 µs |
| Program enqueues per forward | 47 |
| Last `EnqueueProgram` completion from forward start, median | 3.728 ms |

The worker-kernel-free intervals are gaps in the union of all `BRISC-KERNEL`, `NCRISC-KERNEL`, and `TRISC-KERNEL` zones in Tracy, across all chips and cores. Most of their time lies **between** host `EnqueueProgram` zones. For the first measured iteration, 540 µs of such gaps appeared in a 4.994 ms host interval; 94 µs overlapped `EnqueueProgram` and 446 µs fell outside it. The host's 47 enqueues finished by 3.579 ms, then `FDMeshCommandQueue::finish` spent 1.340 ms waiting for completion. This directly shows host pacing between device programs. These union gaps are a conservative visibility measure: a kernel running on any one core masks idle time on other cores, while uninstrumented device activity is invisible.

A second aligned Tracy capture enabled Python function zones for **only the first measured untraced forward** using `DS_PERF_TRACY_PYTHON_ONE=1`. Its trace is `build/profiler/build_wasm/traces/-s_2026_09_27_06_32_08.tracy`. In that forward, the 1.171 ms of worker-kernel-free time before the final device wait aligned mainly with `ttnn.decorators.FastOperation.__call__` (700 µs), then MLA configuration checks and indexer code such as `_cfg_matches`, `score`, and `write_k`. Only about 4 µs of those gaps overlapped the host's `EnqueueProgram` zones. `FastOperation.__call__` includes its call into C++ and therefore does not separate Python, pybind, and C++ time. The full Python probe lengthened this one forward to 5.978 ms and nearly doubled the observed kernel-free time, so its function labels indicate *where* stalls occur, not their uninstrumented duration.

Profiling perturbs this workload. Even the lighter streaming capture raised the untraced minimum from the earlier clean ~4.13 ms to 4.985 ms, while traced replay stayed near 3.05 ms. The ~597 µs kernel-free median in the lighter capture therefore **cannot be equated** to 597 µs of the clean ~1.1 ms traced/untraced gap. It does establish repeated host-side gaps between programs; the remaining gap also reflects changed host/device overlap. The earlier classic Tracy device profiler (`TT_METAL_DEVICE_PROFILER=1`) raised the untraced minimum to 5.714 ms and its exported device times did not align with host markers, so it was excluded from this timeline analysis.

The streaming captures used `TT_METAL_STREAMING_PROFILER=1 TT_METAL_STREAMING_PROFILER_TRACY=1`, `DS_PERF_UNTRACED_HOST=1`, and the exact pytest node ID for `blackhole-glm_5_2-warm-sparse-kv_scaled_fp8-fabric2d-loudbox_sp2xtp4`. The test passed in both. The repository's `python -m tracy -r` postprocessor subsequently errored because it expects legacy device profiler CSV files; the `.tracy` files were saved and inspected directly with `tracy-csvexport`.

### Which host dispatches precede the gaps

The lighter capture already records each `EnqueueProgram op_id` as a Tracy message. Joining that ID to the TTNN op metadata in the same trace identifies all 47 programs without adding per-op Python zones. Each worker-kernel-free interval was split at enqueue timestamps and assigned to the **next** program launch. Across ten measured forwards, the median gap was 597 µs: about 554 µs occurred before the last launch and about 45 µs after it. Only 93 µs of the gap median overlapped `EnqueueProgram`; approximately 504 µs was outside that C++ enqueue zone. The per-category medians below are independently computed and should not be summed as an exact total.

| Next program launched | Median kernel-free time before its launches |
| --- | ---: |
| Matmul (11 launches) | 120 µs |
| High-bandwidth all-gather (5) | 110 µs |
| Layer norm (3) | 86 µs |
| Reduce-scatter (4) | 68 µs |
| Rotary embedding (4) | 54 µs |
| Fast reduce (1) | 40 µs |
| Mesh partition (1) | 36 µs |
| After the final launch | 45 µs |

The largest *individual* recurring intervals precede the second reduce-scatter (~61 µs median) and the third high-bandwidth all-gather (~62 µs median). A label here means that the host has not yet launched that program; the interval can include cleanup after the preceding op, model Python work, and preparation inside the following TTNN call. It is not the execution time of the named device op.

For host function attribution, a separate capture placed `time.perf_counter_ns()` around each `FastOperation.__call__` in only the first measured forward and used two timestamped Tracy messages to align those samples to the device timeline. It is `build/profiler/build_wasm/traces/-s_2026_09_27_06_43_57.tracy`; the corresponding host times are in `generated/profiler/glm52_untraced_matmul_host/dde52cf3864e_warm.json`. That 3.773 ms forward contained 67 TTNN calls taking 3.198 ms in total, including 0.513 ms inside 47 `EnqueueProgram` zones. The remaining ~0.576 ms lay between those TTNN calls in model/control code. The biggest host-call groups were nine `linear` calls (559 µs), four reduce-scatters (425 µs), four rotary embeddings (322 µs), and five high-bandwidth all-gathers (287 µs). Twenty `deallocate` calls consumed 187 µs without launching programs. The earlier one-forward Python trace placed model-side gap time in config selection and indexer flow; TTNN's `FastOperation.__call__` also covers its C++ binding and native preparation work.

The per-op timestamp wrapper raised its first forward to 5.125 ms versus a 4.886 ms median for the other nine forwards in the same capture. It also changed the device gap distribution, so its host-call totals explain the work on the host but **do not provide an exact decomposition** of the lighter capture's 597 µs gap. The lighter capture establishes the launch boundaries and affected operation sequence; the instrumented captures identify the host functions active around them. A causal speedup claim requires changing one of those host paths and remeasuring clean E2E.

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

# Wan2.2 VAE decode regression on BH galaxy: investigation notes

Issue: https://github.com/tenstorrent/tt-metal/issues/60670

Status: **in progress, root cause not yet confirmed.** These findings come from static analysis
of the commit window and from CI job logs. Nothing was run on hardware (the local galaxy was
unavailable).

## Summary so far

- The regression is **Blackhole-specific**. WH galaxy 4x8 runs the same test on the same untraced,
  no-warmup path, and its VAE time did not move across the boundary.
- A CI run on a branch based on mid-window commit `656f87fcdb4` already shows the regression.
  That narrows the window from 53 commits to the ~24 in `f8c4ce59dd0..656f87fcdb4` and **clears**
  the suspects listed in the issue: the tilize Metal 2.0 port (`93f77385b40`), the llk-helper
  copy/typecast migration (`8cf12808550`), the all_gather scatter-write fix (`e44536782e1`) and
  SFPI 7.77.0 (`e4a93d6371c`).
- In the narrowed window, the only change that is BH-specific and could plausibly affect this
  path is the **UMD bump `0d472904340`** (`tt_metal/third_party/umd` `f70cc57f7f8 -> b2a5f606c87`).
  It is the current lead, but I have not yet found the mechanism.
- The missing 480p number on 09-15 is a side effect. On 09-15, the T2V 480p and I2V 720p sc1
  tests both hit pytest timeouts. That fits `0eb79bc47b4` (runtime binary reload), whose
  `dev_msgs.h` layout change invalidates every cached kernel, so the first run recompiles all of
  them. This is not the VAE regression.

## Evidence

### VAE decode times from nightly logs (`test_performance_wan.py`, `t2v`)

| Job | 09-14 `f8c4ce59dd0` | 09-15 `004157dd545` | 09-16 `df15dfd17d5` |
|---|---|---|---|
| BH sc1 `ring_bh_4x8` 480p | 0.324 s | timeout (recompile) | 0.761 s |
| BH sc1 `ring_bh_4x8` 720p | 0.676 s | 1.148 s | 1.046 s |
| WH galaxy `wh_4x8` 480p | 1.138 s | (setup error) | 1.128 s |
| BH sc4 `bh_4x32` 720p, ranks 0-3 | 0.77 / 0.46 / 0.78 / 0.59 s | 0.80 / 0.77 / 0.81 / 0.89 s | 0.82 / 0.81 / 0.85 / 0.87 s |
| BH quietbox_2 `2x2` 480p | 4.24 s | 2.98 s | 4.27 s (noisy, no clear change) |

The BH 4x8 increase is roughly a constant +0.4 s at both 480p and 720p. BH sc4 (4x32) also moved
up somewhat.

The I2V sc1 job flipped from pass on 09-14 (VAE 0.69 s at 720p) to fail on 09-15 (timeout).

### Mid-window data point

`nkapre/matmul-registry-productized` (head `1b992daddf2`, merge base `656f87fcdb4`, 09-14). Its
diff only adds an opt-in matmul/AGMM config registry plus CI changes:

| Job | 480p VAE |
|---|---|
| 105012202248 | 0.753 s |
| 105013804062 | 0.818 s |

Both are already regressed. So the culprit is in `f8c4ce59dd0..656f87fcdb4`, unless that
branch's own changes are responsible, which is unlikely.

Other branch runs in the following days have merge bases after `004157dd545` and add no
information.

### How the test measures VAE time

- `traced = mesh_shape == (4, 32)` in `test_performance_wan.py`. On 4x8 there is no tracing and
  the test's warmup iteration is skipped. The pipeline constructor already warms up the VAE with
  `uint8` output: the `BuildKernels | compiled ...` log lines for tilize, `eltwise_binary_scalar`,
  clamp, untilize and `typecast_rm_chunked` all appear before "Running warmup".
- No `BuildKernels | compiled` lines appear inside the timed run, in either the good or the bad
  log. So the time is not JIT compile in the timed section.
- The sets of compiled kernels before the timed run are essentially the same in the good and bad
  runs (only small differences in tilize / `tilize_metal2` / `cq_dispatch` variant counts).
- On BH the VAE is not coresident-excluded, so `reload_weights()` inside `decode()` does nothing,
  and no weight upload falls in the timed section.
- The timed section includes `fast_device_to_host` (`float_to_uint8`, then `.cpu()`, then host
  reassembly). Host↔device transfer behaviour is therefore in scope.

### Commits in `f8c4ce59dd0..656f87fcdb4` and why each is ruled out or still open

| Commit | Change | Verdict |
|---|---|---|
| `668c2907575` | Sharded concat rank fix | Only changes factory selection for **sharded** inputs. The VAE's concats take the interleaved path. Unlikely. |
| `5dfe136d485` | msda fixed point | Not used by the VAE. |
| `ce240a3229e` | high_bw_all_gather open path | Not used by the VAE. |
| `294bd9a0436` | gather reader clamp | Not used by the VAE. |
| `9a1b4e565bb` | sparse SDPA attention sink | Only touches the attention-sink path in `compute_streaming.hpp`; the VAE has no sinks. |
| `0d472904340` | **UMD bump** | **Open, lead suspect.** Only BH-specific candidate. See below. |
| `668752a4056` | Comparison mode for mesh tensors (`decorators.py`) | Only touches comparison-mode code; `FastOperation.__call__` is unchanged. Unlikely. |
| `70f840f46ad` | SDPA decode correction in FP32 half-sync | Decode-only and FP32-half-sync-only. Unlikely. |
| `a8fe40d9439`, `fdfad84bffc`, `bd5b7a7a7aa` | LLK sanitizer hooks (pack untilize, unpack AB, WH fast tilize) | `SAN_HOOK` reduces to an empty `do {} while (false)` when the sanitizer is disabled, and WH got the same hooks without regressing. Unlikely, but codegen was not checked. |
| `656f87fcdb4` | Scalar-LHS binary | The tensor-scalar path is unchanged apart from the extra `scalar_is_lhs=false` hash field. Unlikely. |
| others | Models / CI / targets / Quasar only | Not relevant. |

### UMD bump contents (`f70cc57f7f8..b2a5f606c87`)

- tt-umd#3399: adds an `IoOrdering` parameter and flips `LocalChip::write_to_device` /
  `read_from_device`'s reconfigured-window **fallback** from Relaxed to **Strict**. The PR states
  the cost: "11x to 16x on the bulk write path on wormhole_b0, 28x to 30x on blackhole. Reads are
  unaffected". It only applies to targets not covered by a static TLB. Metal configures static
  TLBs for Tensix, ETH and DRAM (`include_dram_tlbs = true` on silicon, `tt_cluster.cpp`), so the
  fast-dispatch hot path should not reach the fallback. **Still need to confirm** which path, if
  any, hits it during VAE decode / D2H on BH.
- tt-umd#3406: routes `SiliconTlbWindow::configure` through `execute_safe` when IoSafety is
  enabled. Matters only if windows are reconfigured often.
- tt-umd#3210, #3380, #3381: telemetry accessors moved to `DeviceFirmware`, ARC APB accessors
  retired, `configure_iatu_region` deleted. Metal's `dispatch_telemetry.cpp` calls
  `tt_device.read_from_device` / `write_to_device`, so check how often that runs.
- tt-umd#3349, #3414, #3416, #3417: cluster_id, test renames, CI. Not relevant.

## Next steps

1. **On hardware:** build `656f87fcdb4` and `0d472904340^` (or check out `f8c4ce59dd0` with only
   the UMD submodule moved to `b2a5f606c87`) and compare
   `test_performance_wan.py -k "ring_bh_4x8_sp1tp0 and t2v and resolution_480p"`.
   This confirms or clears the UMD bump in one A/B run.
2. If the UMD bump is the cause, split it between tt-umd#3399 (Strict fallback) and #3406 /
   telemetry, and find the Metal call site that hits the slow path. Use `perf` or Tracy on the VAE
   decode and `fast_device_to_host`.
3. If the UMD bump is not the cause, bisect the remaining VAE-relevant commits in the narrowed
   window (`668c2907575`, `656f87fcdb4`, the LLK sanitizer commits).
4. Independently: add a warmup or tracing for 4x8 in `test_performance_wan.py` so VAE timing is
   less sensitive to first-run effects.

## Data sources

- Nightly runs: 34799396959 (09-14), 34921216342 (09-15), 35047979653 (09-16).
- BH sc1 T2V jobs: 103840008423, 104231133586, 104643382568.
- WH galaxy jobs: 103840008560, 104643382856. BH sc4 jobs: 103840008385, 104231133439, 104643382682.
- Mid-window branch jobs: 105012202248, 105013804062. Later branch jobs: 104847443485 (VAE 0.81 s).

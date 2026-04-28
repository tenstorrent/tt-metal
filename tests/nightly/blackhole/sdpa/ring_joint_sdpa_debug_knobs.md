# Ring Joint SDPA — debug knobs and CCL/compute breakdown

This file documents two perf-only debug knobs added to the ring joint SDPA op
and a baseline measurement that uses them to attribute wallclock between
SDPA compute and CCL on Blackhole.

The knobs are env-var driven and set kernel `#define`s through the program
factories. They are **perf-only** — outputs are wrong when iters are skipped
or CCL is disabled, so accuracy must not be checked.

## Knobs

### `TT_RING_JOINT_SDPA_RING_ITER_MODE`
Selects which ring iterations execute the SDPA per-iter compute.

| Value | Effect |
|---|---|
| `all` (default) | Runs all `ring_size` iterations (normal behavior) |
| `iter0` | Runs only `ring_iter == 0` (the local-data iteration) |
| `skip_iter0` | Skips `ring_iter == 0`, runs the rest |

Wired in `ring_joint_sdpa_program_factory.cpp`. The skip is applied **after**
`get_next_ring_id_and_sync()` in compute, reader, and writer kernels, so the
fused-op fabric and ring-id state machine still progress in lock-step.

### `TT_RING_JOINT_SDPA_DISABLE_CCL`
Set to `1` to remove CCL from the ring joint SDPA flow.

* The all-gather reader and writer kernels return immediately at the top
  of `kernel_main`.
* The SDPA reader's `get_next_ring_id_and_sync()` skips the
  `noc_semaphore_wait_min` (the AG semaphore is never incremented, so it
  would otherwise hang).
* `gathered_k` / `gathered_v` retain their pre-init values (zeros from the
  test harness), so `ring_iter > 0` reads garbage K/V — perf-only.

Both knobs can be combined.

## What this measures

For `mla_100k` (DeepSeek MLA), `q_chunk=160`, `k_chunk=320`, on a 4-device
single-ring Blackhole setup (Quiet Box):

* `seq_len = 3200` per device, `12800` global (across the ring)
* `nhq = 29`, `nhk = 1`, `d_q = d_k = 576`, `d_v = 128`
* `is_causal = True`, `is_balanced = True` (zigzag balanced)
* Q in `bfloat16`, K/V in `bfloat8_b`
* Compute grid: 87 / 100 cores used; 10 cores for CCL; ring_size = 4

Math util is computed by `compute_math_utilization` in
`tests/nightly/sdpa_perf_utils.py`. When `ring_iter_mode != all`, it scales
useful FLOPs by `iters_executed / ring_size` (uniform per-iter work — exact
for balanced-causal, where each iter contributes equal useful FLOPs).

## Repro

Restrict `mla_100k.k_chunk_sizes` to `[320]` if you only care about the k=320
row, or grep the rank-1 row from the table the test prints.

```bash
source python_env/bin/activate

# Baseline (ring with CCL):
TT_RING_JOINT_SDPA_RING_ITER_MODE=all         scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table -k mla_100k
TT_RING_JOINT_SDPA_RING_ITER_MODE=iter0       scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table -k mla_100k
TT_RING_JOINT_SDPA_RING_ITER_MODE=skip_iter0  scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table -k mla_100k

# Same three with CCL disabled (kernels short-circuit, AG semaphore never fires):
TT_RING_JOINT_SDPA_DISABLE_CCL=1 TT_RING_JOINT_SDPA_RING_ITER_MODE=all         scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table -k mla_100k
TT_RING_JOINT_SDPA_DISABLE_CCL=1 TT_RING_JOINT_SDPA_RING_ITER_MODE=iter0       scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table -k mla_100k
TT_RING_JOINT_SDPA_DISABLE_CCL=1 TT_RING_JOINT_SDPA_RING_ITER_MODE=skip_iter0  scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table -k mla_100k
```

The op-side env vars are read by the SDPA program factory and the ring-attention
all-gather program factory at program-build time, so the env-var setting must be
present in the shell when pytest runs.

## Numbers — recorded 2026-04-28 on Blackhole 4-device single-ring

`q=160`, `k=320`, ring_size=4. Math util is from
`compute_math_utilization` (useful FLOPs / theoretical peak), with the
`ring_iter_mode` scaling applied.

### CCL on (default)

| Mode | Duration (ms) | Math Util |
|---|---|---|
| `all` | 5.20 | 58.1% |
| `iter0` | 1.96 | 38.6% |
| `skip_iter0` | 4.14 | 54.8% |

### CCL off (`TT_RING_JOINT_SDPA_DISABLE_CCL=1`)

| Mode | Duration (ms) | Math Util |
|---|---|---|
| `all` | 5.18 | 58.3% |
| `iter0` | 1.86 | 40.6% |
| `skip_iter0` | 3.34 | 67.9% |

### Δ (CCL off − CCL on)

| Mode | Δ duration (ms) | Δ math util |
|---|---|---|
| `all` | ~0 | ~0 |
| `iter0` | -0.10 | +2.0 pp |
| `skip_iter0` | **-0.80** | **+13.1 pp** |

## Interpretation

* **`all` duration is unchanged when CCL is disabled** → CCL is fully hidden
  under iter 0's compute. Iter 0 runs in parallel with CCL round 1 and is
  effectively "free" wallclock time in the full pipeline.
* **`skip_iter0` drops 0.80 ms with CCL off** → that 0.80 ms is the exposed
  CCL-round-1 wait that the SDPA reader otherwise sits on (because iter 0 is
  no longer there to overlap with it). With CCL out of the picture, removing
  iter 0 (the worst-utilized iter due to causal-mask waste) genuinely
  improves math util — `skip_iter0` (67.9%) beats `all` (58.3%) when CCL is
  off, but loses to `all` (54.8% vs 58.1%) with CCL on.
* **`iter0` standalone is the worst per useful FLOP** in both regimes
  (38.6% / 40.6%). The kernel processes the full local Q×K rectangle and
  applies a causal mask, so the upper-triangle work is wasted.

The takeaway: today the 4-iter pipeline is balanced such that CCL latency ≈
iter 0 compute time, so iter 0 is "free." If CCL ever becomes faster than
iter 0's compute, optimizing iter 0 (e.g. skipping upper-triangle tiles
entirely) would start to pay off in `all` as well.

## Files touched by the knobs

* `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp`
  — reads both env vars, sets defines on SDPA kernels.
* `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp`,
  `kernels/dataflow/ring_joint_reader.cpp`,
  `kernels/dataflow/ring_joint_writer.cpp`
  — `continue` after sync based on `RING_JOINT_SDPA_RING_ITER_MODE`.
* `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_receiver.hpp`
  — skip `noc_semaphore_wait_min` when `RING_JOINT_SDPA_DISABLE_CCL`.
* `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.cpp`
  — reads the disable-CCL env var, attaches define to all 4 AG kernel configs.
* `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp`,
  `kernels/ring_attention_all_gather_writer.cpp`
  — early `return;` at top of `kernel_main`.
* `tests/nightly/sdpa_perf_utils.py`
  — `compute_math_utilization` accepts `ring_iter_mode` for FLOPs scaling.
* `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`
  — perf-table test passes `ring_iter_mode` from env to the util function.

## DRAM-cost breakdown (perf-only experiments, not committed)

Same `mla_100k`, `q=160`, `k=320`, `mode=all`, CCL on. Each variant either
disables a DRAM read or rewires the V/K chain. CB tiles are stale when reads
are skipped, so outputs are wrong — perf-only.

| Variant | Duration (ms) | Math Util | Δ vs baseline |
|---|---|---|---|
| Baseline | 5.205 | 58.1 % | — |
| V row-0 only DRAM, V chain disabled | 5.091 | 59.4 % | −0.114 |
| V chain unicast on, V DRAM off | 5.086 | 59.5 % | −0.119 |
| V chain disabled, V DRAM off | 5.053 | 59.9 % | −0.152 |
| K mcast on, K DRAM off | 5.077 | 59.6 % | −0.128 |
| K + V chains on, both DRAM off | 4.974 | 60.8 % | −0.231 |
| Q DRAM off (K + V on) | 4.778 | 63.3 % | −0.427 |

Reading:

* **Q is the largest exposed DRAM cost (~427 µs)** — 87 cores read Q in
  parallel each q_chunk start, with no chain to share. Q is `bfloat16`
  (2× larger than `bfloat8_b` K/V tiles).
* **K (~128 µs) and V (~119 µs) are comparable** despite K having a single
  mcast injector and V having ~29 unicast injectors. K wins per-reader
  because its chunk is 4.5× larger (`DHt=18` vs `vDHt=4`) *and* K is on
  the critical path of every k_chunk iter (compute waits on K before V).
* The K + V "cost" is roughly additive (231 µs ≈ 119 + 128 − 16) — the
  two readers don't fight for the same NoC bottleneck.

## Q DRAM mid-barrier (committed change)

`fetch_block` / `read_block` in `kernels/dataflow/dataflow_common.hpp`
take an optional `barrier_threshold` parameter (default `0` = old
behavior, single trailing barrier). When non-zero, an intermediate
`noc_async_read_barrier()` fires every `barrier_threshold` tile reads
to throttle in-flight reads and avoid saturating the NoC outstanding-read
budget.

The ring joint reader passes
`get_barrier_read_threshold<q_tile_bytes, num_readers>()` for Q reads —
the same formula `((512 / num_readers) * (1024 + 128)) / tile_bytes`
used by single-chip SDPA. `num_readers` is plumbed through as a new
compile-time arg (`num_cores`) from the program factory.

For `mla_100k` (`num_readers=110`, `q_tile_bytes=2048`) the formula
yields **threshold = 2**. Empirical sweep (3 runs averaged):

| Variant | Duration (ms) | Math Util |
|---|---|---|
| Baseline (no mid-barrier, mean of 3) | 5.209 | 58.1 % |
| Threshold = 2, formula `(1024 + 128)` (mean of 3) | **5.174** | **58.5 %** |
| Threshold = 4, formula `(2048 + 128)` (mean of 3) | 5.200 | 58.2 % |

Net: **−35 µs / +0.4 pp** with the API formula. Tightening to threshold=4
nearly disappears (within noise of baseline) — the budget formula's
`1024` constant is the right tuning here. K and V readers are unchanged
(still no mid-barrier) — only Q saw enough in-flight pressure to benefit.

Files touched:

* `kernels/dataflow/dataflow_common.hpp` — add `barrier_threshold = 0`
  parameter to `fetch_block` and `read_block`.
* `kernels/dataflow/ring_joint_reader.cpp` — read `num_readers`
  compile-time arg, derive `q_barrier_threshold` from
  `get_barrier_read_threshold`, pass into `read_block` for Q.
* `ring_joint_sdpa_program_factory.cpp` — append `num_cores` to reader
  compile-time args.

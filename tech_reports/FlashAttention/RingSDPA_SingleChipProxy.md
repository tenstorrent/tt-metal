# Ring-Joint SDPA: Single-Chip Per-Iter Proxy

Tracks single-chip SDPA configs that reproduce the per-device, per-ring-iter
work of multi-chip `ring_joint_scaled_dot_product_attention`, so per-iter
perf can be tuned without a multi-chip harness.

## Terminology

- **Ring joint SDPA** — multi-device causal+balanced SDPA. Each device runs
  `ring_size` iterations; on iter `R`, every device attends its local Q
  against K/V rotated from device `(self - R) mod ring_size`.
- **Iter 0 (causal)** — every device attends local Q × local K/V with full
  causal masking.
- **Iter > 0** — non-causal on local×local shapes; balanced zigzag splits
  work so each device either does **UP** or **DOWN** work:
  - **UP** — the incoming K/V block sits "above" (earlier in the seq) than
    the local Q, so causality caps each Q's K-loop at half length. All Q
    chunks are assigned; each walks `k_num_chunks/2`.
  - **DOWN** — the heavy Q half is kept, light half is skipped (balanced
    load transfer). Only `q_num_chunks/2` Q slots are assigned; each walks
    the full K length.
- **Single-chip proxy** — regular SDPA on one device with `flatten_work=True`
  and a `ring_proxy_case` flag (`none`/`up`/`down`) that makes the kernel
  skip the matching half-iter work. Test IDs:
  `mla_100k_ring_iter_{0,up,down}` in
  `tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py`.
- **`TT_METAL_RING_ITER_ONLY=R`** — env var on the multi-chip perf-table
  test (`test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table`)
  that runs only ring iter `R`, skips AllGather, and reports per-iter
  (rather than full-ring) math util.

## Current numbers (mla_100k, BH, QB, post causal-chain commit)

### Single-chip proxy (nhq=32, 110 cores, q160/k160)

| Proxy             | Duration (ms) | Math util |
|-------------------|---------------|-----------|
| `ring_iter_0`     | 2.658         | 28.5%     |
| `ring_iter_up`    | 1.423         | 53.3%     |
| `ring_iter_down`  | 1.334         | 56.8%     |

(Iter-0 previously 2.725 ms / 27.8 %; KV chain forwarding on the flat-work
causal path brings modest improvement.  A-narrow chain participants over-read
K past `q_high` — lightweight causal mask zeros the phantom columns — which
costs compute that partly offsets the injector-only DRAM savings.)

### Multi-chip ring-joint, per device × per iter (nhq=29, 80 effective cores, ring_size=4)

Duration (ms):

| iter \ dev | D0    | D1    | D2    | D3    | max (full-ring) |
|------------|-------|-------|-------|-------|------------------|
| **0** CAUSAL | 2.435 | 2.408 | 2.413 | 2.455 | 2.455           |
| **1**       | 1.339 | 1.426 | 1.342 | 1.344 | 1.426           |
| **2**       | 1.444 | 1.428 | 1.351 | 1.428 | 1.444           |
| **3**       | 1.342 | 1.434 | 1.344 | 1.441 | 1.441           |

Math util (%):

| iter \ dev | D0   | D1   | D2   | D3   |
|------------|------|------|------|------|
| **0**      | 31.0 | 31.4 | 31.3 | 30.8 |
| **1**      | 56.5 | 53.0 | 56.3 | 56.3 |
| **2**      | 52.4 | 52.9 | 56.0 | 53.0 |
| **3**      | 56.3 | 52.7 | 56.3 | 52.5 |

Classification (by duration vs proxies):

| iter \ dev | D0     | D1   | D2   | D3   |
|------------|--------|------|------|------|
| **0**      | CAUSAL | CAUSAL | CAUSAL | CAUSAL |
| **1**      | DOWN   | UP   | DOWN | DOWN |
| **2**      | UP     | UP   | DOWN | UP   |
| **3**      | DOWN   | UP   | DOWN | UP   |

## Proxy fidelity

| Case   | Proxy util | Ring util      | Gap (pp) |
|--------|------------|----------------|----------|
| UP     | 53.3%      | 52.4 – 53.0%   | 0 – 1    |
| DOWN   | 56.8%      | 56.0 – 56.5%   | 0        |
| CAUSAL | **28.5%**  | **30.8 – 31.4%** | **~2 – 3** |

UP/DOWN proxies track ring-joint within noise. Iter-0 causal closed ~0.7 pp
after adding KV chain forwarding on the flat-work causal path; a ~2–3 pp gap
remains, likely a mix of parameter mismatch (proxy uses `nhq=32`, 110 cores,
Galaxy-tuned; QB ring-joint runs `nhq=29`, 80 effective cores) and the
A-narrow chain cost — chain participants walk the full `k_num_chunks` and
rely on the lightweight causal mask to zero the over-read columns, which
costs compute that partially offsets the injector-only DRAM savings.

## How to run the single-chip tests

All commands assume `python_env` is activated and are run from the repo root.
`run_safe_pytest.sh` serializes device access, appends `-x`, and resets the
device after every run.

The sprint file exposes four tests, all parametrized over the model configs
(`wan2_2_1xGLX_analog`, `wan2_2_4xGLX_analog`, `mla_100k_ring_iter_{0,up,down}`,
`mla_128k_ring_iter_0`):

| Test                           | Purpose                                   |
|--------------------------------|-------------------------------------------|
| `test_sdpa_sweep_perf_impl`    | Per-config perf sweep (skipped on CI).    |
| `test_sdpa_accuracy`           | Per-config PCC / RMSE check vs torch.     |
| `test_sdpa_determinism`        | Runs each config 10×, asserts bit-exact.  |
| `test_sdpa_create_perf_table`  | Per-model perf table generator (tracy).   |

### Accuracy (single config)

```bash
source python_env/bin/activate
scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy[mla_100k_ring_iter_down-q160-k160]
```

Run all MLA 100k accuracy cases:

```bash
scripts/run_safe_pytest.sh --run-all \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy" \
  -k "mla_100k"
```

### Perf table (one model, with tracy)

```bash
scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[mla_100k_ring_iter_down]
```

Output includes Duration, Core Util, Iters/Core, Pad/Slot Waste, and **Math
Util** per chunk-size combination, with a `Best configuration` line.

Run all three MLA 100k proxies back-to-back:

```bash
scripts/run_safe_pytest.sh --run-all \
  tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[mla_100k_ring_iter_0] \
  tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[mla_100k_ring_iter_up] \
  tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[mla_100k_ring_iter_down]
```

### Determinism

```bash
scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_determinism[mla_100k_ring_iter_0-q160-k160]
```

### Multi-chip ring-joint per-iter measurement (for proxy validation)

Uses `TT_METAL_RING_ITER_ONLY=R` to force only ring iter `R`:

```bash
TT_METAL_RING_ITER_ONLY=0 scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table[mla_100k]
```

Per-device durations live in the tracy CSV:
`generated/profiler/ttnn_ring_joint_sdpa_performance/reports/<ts>/ops_perf_results_<ts>.csv`
(filter by `OP CODE == RingJointSDPADeviceOperation`).

## Recent changes

### KV chain forwarding on the flat-work causal path (A-narrow)

Extends the existing chain-forwarding optimization (previously non-causal
only) to `flatten_work=true && is_causal=true` configs such as
`mla_100k_ring_iter_0`.  Implementation sketch:

- Host gate widened from `!is_causal && !is_chunked` to
  `!is_chunked && (!is_causal || (flatten_work && lightweight_causal))` —
  hierarchical causal keeps its tuned per-Q truncated-K loop; flat-work
  causal + lightweight causal mask gets chains.
- A new `SDPA_KV_CHAIN_ENABLED` compile-time define replaces the previous
  `!is_causal` gate in the reader / compute kernels, decoupling chain
  logic from causality.
- Chain participants loop the **full** `k_num_chunks` (A-narrow): reader,
  writer, compute stay in sync; the lightweight causal mask (enabled by
  default under `lightweight_causal`) zeroes softmax contributions past
  each Q's true `q_high`.  Alone cores (no chain on their head) keep the
  per-Q truncated K loop.
- `is_chain_participant` is plumbed through `sdpa_standard` and
  `sdpa_inner_loop` as a runtime bool so compute's K-loop end bound
  matches the reader's.

Result on mla_100k_ring_iter_0: 2.725 ms / 27.8 % → 2.658 ms / 28.5 %
(≈1 pp).  UP / DOWN / WAN / hierarchical causal configs unchanged.
Full accuracy suite passes; determinism stays bit-exact.

### Skip mid-barrier for K/V reads in single-chip SDPA reader

Commit `Skip mid-barrier for K/V reads in single-chip SDPA reader` —
adds `enable_mid_barrier` template bool (default `true`) to
`read_chunk_with_padding`. `reader_interleaved.cpp` opts out at K/V
DRAM-read call sites. Matches `ring_joint_reader`'s single-end-barrier
pattern. Lifted mla_100k DOWN proxy from 44.2% → 56.6% util
(1.72 → 1.34 ms). Windowed SDPA and Q reads keep the default.

## Full-ring bottleneck

Each iter > 0 has at least one device doing UP work (~1.43 ms). Since
ring joint waits on the slowest device per iter, full-ring time is
dominated by UP regardless of how many DOWNs run in parallel. Over
iters 1–3 the totals are 6×UP and 6×DOWN (balanced), but per iter
asymmetric (3D/1U, 1D/3U, 2D/2U). Shaving UP's K-loop
launch/end-barrier overhead would compress the whole ring more than
speeding up DOWN.

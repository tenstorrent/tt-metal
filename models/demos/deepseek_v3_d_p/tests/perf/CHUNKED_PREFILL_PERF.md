# Chunked-Prefill Device-Perf vs E2E-Perf (Kimi K2.6)

## What this is

A single combined perf driver that quantifies the **host/dispatch tax** of DeepSeek/Kimi chunked
prefill — the gap between the device's pure compute time and the actual end-to-end wall-clock per
chunk — and breaks it down **per layer, per op**. It complements the analysis in
[`../../tt/runners/H2D_DISPATCH_TAX.md`](../../tt/runners/H2D_DISPATCH_TAX.md).

For one prefill chunk it reports:

- **Device perf** — merged multi-device `DEVICE KERNEL DURATION` (pure on-device compute).
- **E2E perf** — averaged wall-clock over 10 warm standalone loops.
- **Loss** — `e2e − device` (host/dispatch/sync overhead), in ms and as % of e2e.
- **Per-layer / per-op breakdown** — for the worst (critical-path) device, every op's device-kernel
  time and the op-to-op gap (host-dispatch latency) before it, grouped by transformer layer.

## Files

| File | Role |
|---|---|
| `tests/perf/test_prefill_chunked_perf.py` | Driver test (this dir). Runs both phases, prints summary + per-layer op2op. |
| `tests/test_prefill_transformer_chunked.py` | Worker: `test_kimi_prefill_transformer_chunked_no_pcc` + `run_chunked_transformer_no_pcc` (build once, loop the forward, no PCC). |
| `utils/perlayer_op2op.py` | Per-layer/per-op device + op2op parser. Standalone CLI and used by the driver. |
| `utils/perf_utils.py` | `measure_device_perf_ns` (run tracy + merge, return ns), `run_e2e_wall_clock` (plain subprocess + JSON). |

## How it works

The driver takes **no device fixture** (like the other drivers here); each phase opens the mesh in its
own subprocess, so the driver holds no chip locks.

**Phase 1 — device perf + op2op (under tracy):** runs the worker with `TT_PREFILL_PROFILE_WARMUP=1`,
which runs **one compile/warmup chunk** (JITs all kernels), flushes the on-device profiler buffer
(`ttnn.ReadDeviceProfiler`), then runs **one warm measured chunk** bracketed by the
`PROFILE_MEASURE_START` / `PROFILE_MEASURE_END` signposts. Only the warm chunk is summed/analyzed; the
compile chunk is excluded. Warm is deliberate: op-to-op gaps on the cold/JIT pass are meaningless, and
flushing after warmup keeps tracy's DRAM marker buffer from overflowing.

**Phase 2 — e2e wall-clock (no tracy):** runs the worker (`num_iters=10`) as a plain subprocess; the
worker writes per-iter timings to `TT_PREFILL_PERF_JSON`. Phase 2 runs after Phase 1 and uses JSON
(no profiler dir), so it never clobbers the Phase 1 CSV.

Per-layer attribution uses the `forward_layer_{i}_start/_end` signposts emitted inside
`TtPrefillTransformer.forward`. The parser locks to the **worst device** (max kernel + op2op), buckets
pre-layer embedding and post-layer norm/lm_head, and clips the spurious region-start idle gap.

> Notes:
> - `TT_PREFILL_PROFILE_WARMUP` is passed via env, never prefixed into the pytest command — tracy's
>   `-m` mis-parses leading `KEY=VAL` tokens.
> - The worker always runs a single chunk and is parametrized `num_layers ∈ [1,5,10,61]`, `num_iters ∈
>   [1,2,10,20]`; the driver selects `L5 / iters1` (device) and `L5 / ten_iters` (e2e). `-k` is
>   substring-matched, so pick the unambiguous layer id (`L5`, not `L1` which also matches `L10`).

## Running

Requires an 8x4 Blackhole mesh + the Kimi TTNN weight cache:

```bash
export KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized
export TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill
export TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden

pytest models/demos/deepseek_v3_d_p/tests/perf/test_prefill_chunked_perf.py -s
```

Input tokens come from the longbook K2.6 golden trace
(`/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320`, resolved automatically);
the worker falls back to a synthetic in-vocab pattern if the trace is absent (perf is unaffected).

Standalone per-layer breakdown on any tracy ops CSV:

```bash
python -m models.demos.deepseek_v3_d_p.utils.perlayer_op2op <ops_perf_results.csv>
```

## Reference results — 5 layers, 1 chunk (0 KV cache, 5120 tokens), warm

| Metric | Value |
|---|---|
| Device perf (kernel, merged 32 chips) | 115.0 ms |
| E2E perf (wall-clock, 10 warm loops) | 135.3 ms |
| **Loss (host/dispatch)** | **20.3 ms (15.0% of e2e)** |

Per-layer (worst/critical-path device, per-chunk µs):

| Layer | device µs | op2op µs | ops |
|---|---|---|---|
| embed/pre | 25.8 | 0.0 | 1 |
| L0 (dense) | 4,601 | 12,638 | 40 |
| L1 (MoE) | 23,568 | 17,894 | 100 |
| L2 (MoE) | 23,427 | 11,829 | 100 |
| L3 (MoE) | 23,978 | 8,142 | 100 |
| L4 (MoE) | 22,103 | 6,811 | 100 |
| norm/lm_head/post | 1,626 | 2.7 | 5 |
| **Grand total** | **99,330** | **57,317** | — |

**Grand total: device 99.3 ms / op2op 57.3 ms / total 156.6 ms → op2op = 36.6%** on the worst device
(higher than the 15% cross-device-merged loss, because the merged figure takes per-op max-kernel across
chips and hides per-op dispatch bubbles; the single-device view is the honest critical-path picture).

### Findings

- **The op2op tax concentrates in the MoE attention + gate/routing region** (many small ops —
  `MoeGroupedTopk`, `MaskedBincount`, `OffsetCumsum`, `Untilize`, `Slice`, `Typecast`, `Dispatch` —
  each doing trivial device work but eating 150 µs–3.5 ms of dispatch gap). This is the dispatch-bound
  small-op pattern from the H2D dispatch-tax analysis.
- **The expert-FFN + combine region is ~0 op2op** — big back-to-back ops (`UnifiedRoutedExpertFfn`
  ~770 µs ×16, `Tilize`/`Unary`/`Combine`/`ReduceScatter` 1.3–2.6 ms) pipeline perfectly.
- **Biggest device ops:** `ReduceScatter` (combine) ~2.6 ms, `RingJointSDPA` ~2.15 ms, `Tilize` ~2.1
  ms, `UnaryDevice` ~1.48 ms, `Combine` 1.35–1.87 ms, `Dispatch` 1.1–1.8 ms.
- op2op trends down across L1→L4 (17.9→6.8 ms) as the pipeline settles; L0's 12.6 ms is partly
  first-measured-layer startup residue.

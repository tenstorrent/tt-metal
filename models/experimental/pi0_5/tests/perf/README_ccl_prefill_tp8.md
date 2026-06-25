# CCL TP=8 Prefill Latency

Device-kernel duration for `reduce_scatter` + `all_gather` at pi0.5 prefill shapes (seq=768, TP=8 Ring).

## Run the pytest

Requires 8 BH chips (24-31) + fabric.

```bash
python -m tracy -p -r -n ccl_prefill_tp8 -o /tmp/tracy_ccl_tp8 \
  $(which pytest) models/experimental/pi0_5/tests/perf/test_ccl_prefill_tp8_perf.py -s
```

Parse `ops_perf_results.csv` → filter `ReduceScatterDeviceOperation` / `AllGatherDeviceOperation` → read `DEVICE KERNEL DURATION [ns]`.

## Latency numbers (chips 8-15, sane devices 1/5/6)

Each VLM layer has **2 all-reduces** (MLP down_proj + attention o_proj). RS and AG operate on
different shapes — RS takes the full hidden partial sum, AG takes the scattered slice:

| Op             | Shape          | DK duration |
|----------------|----------------|-------------|
| reduce_scatter | [1,1,768,2048] | ~71 µs      |
| all_gather     | [1,1,768,256]  | ~54 µs      |

Gate/up projections are column-parallel and stay local — no CCL between gate/up and down_proj.

Per layer: 2 × (71 + 54) = **~250 µs CCL** out of ~472 µs total DK = **~53% of prefill**.

## 1-layer production prefill (chips 8-15)

Run `_bench_prefill_tp8_ccl.py` under tracy — mirrors exact env flags of `test_perf_tt_bh_glx_1x8.py`:

```bash
source _bench_runs/pi05_production.env
python -m tracy -p -r -n prefill_tp8_ccl -o /tmp/tracy_prefill_ccl \
  python_env/bin/python \
  models/experimental/pi0_5/tests/perf/_bench_prefill_tp8_ccl.py
```

18-layer prefill DK ≈ 8.5 ms → per layer ≈ 472 µs → CCL ≈ **53% of prefill**.

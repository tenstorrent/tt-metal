# CCL TP=8 Prefill Latency

Device-kernel duration for `reduce_scatter` + `all_gather` at pi0.5 prefill shapes (seq=768, TP=8 Ring).

## Run the pytest

Requires 8 BH chips (24-31) + fabric.

```bash
python -m tracy -p -r -n ccl_prefill_tp8 -o /tmp/tracy_ccl_tp8 \
  $(which pytest) models/experimental/pi0_5/tests/perf/test_ccl_prefill_tp8_perf.py -s
```

Parse `ops_perf_results.csv` → filter `ReduceScatterDeviceOperation` / `AllGatherDeviceOperation` → read `DEVICE KERNEL DURATION [ns]`.

## Latency numbers (chips 24-31, sane devices 2/5/6/7)

Each VLM layer has **2 all-reduces** at hidden shape `[1,1,768,1024]`:

| Location        | Op             | Shape          | DK duration |
|-----------------|----------------|----------------|-------------|
| MLP down_proj   | reduce_scatter | [1,1,768,1024] | ~55 µs      |
| MLP down_proj   | all_gather     | [1,1,768,1024] | ~39 µs      |
| Attention o_proj| reduce_scatter | [1,1,768,1024] | ~55 µs      |
| Attention o_proj| all_gather     | [1,1,768,1024] | ~39 µs      |

Gate/up projections (mlp_mid [1,1,768,4096]) are column-parallel and stay local — no CCL at that shape.

## 1-layer production prefill (chips 8-15)

Run `_bench_prefill_tp8_ccl.py` under tracy — mirrors exact env flags of `test_perf_tt_bh_glx_1x8.py`:

```bash
source _bench_runs/pi05_production.env
python -m tracy -p -r -n prefill_tp8_ccl -o /tmp/tracy_prefill_ccl \
  python_env/bin/python \
  models/experimental/pi0_5/tests/perf/_bench_prefill_tp8_ccl.py
```

Result (sane devices 1/5/6 for chips 8-15):

| Op             | DK duration |
|----------------|-------------|
| reduce_scatter | ~55 µs      |
| all_gather     | ~38 µs      |
| **CCL/layer**  | **~186 µs** (2 all-reduces × 93 µs) |

18-layer prefill DK ≈ 8.5 ms → per layer ≈ 472 µs → CCL ≈ **39% of prefill**.

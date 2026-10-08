# Completed single-step GDN candidate sweep

[Input/output/TTFT graphs](index.html) · [CSV](sweep.csv) · [Raw JSON](sweep.json) ·
[Matched native comparison](../gdn-matched-comparison-v4/README.md)

One physical TP4 replica, all 64 layers, fixed 128-token generation, one warmup
and three measured repeats. Weights, KV and activation precision match the
native control; only decode recurrence changes to the in-place FP32 candidate
with fused Q/K normalization. This is not an online-evaluation or full-Galaxy
qualification result.

The sweep finished with 24 measured cells, one B32/32K allocator OOM and ten
explicitly untested capacity/implementation guards. `attempt-01` completed
the long-context cells, hit the known transient prefill allocation limit and
closed devices cleanly. `attempt-02` preserved those measurements and the OOM,
loaded a fresh process, and completed the remaining cells. The aggregate
receipt is `sweep.json`; raw attempts and compressed hardware JUnit receipts
are included. No failed or guarded point is plotted as a throughput result.

| Input tokens | Users per TP4 | Prefill input tok/s | Decode output tok/s | Decode tok/s/user |
|---:|---:|---:|---:|---:|
| 32,768 | 16 | 6,567.87 | 259.22 | 16.20 |
| 131,072 | 8 | 4,983.04 | 123.90 | 15.49 |
| 262,016 | 4 | 3,678.79 | 64.74 | 16.18 |

For exact timings, use the CSV/JSON. Input throughput is measured during
prefill; decode throughput excludes prefill. Request throughput and TTFT are
reported separately. Near-256K reserves space for the output budget. The
candidate remains opt-in until full-model reference evaluations pass.

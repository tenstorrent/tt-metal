Measured on 2026-10-21: full 60-layer Gemma4 31B, CP=8 / TP=4, fixed 8192-token batch capacity, five warmed samples per cell. All three paths share one model and cache allocation. The same 120 hybrid traces are reused across every request layout and prefix. See [timings](summary.md) and [individual samples](measurements.json).

Hybrid adds 9.42–11.62 ms (1.6–9.2%) over fully traced packed prefill in these cases.

| Case        | Prefix   | Extra device ms   | Extra %   | Extra program ms   | Extra gap ms   | Extra programs   |
|-------------|----------|-------------------|-----------|--------------------|----------------|------------------|
| single_full | 0        | 10.44             | 8.1%      | 5.99               | 4.52           | 363              |
| single_1600 | 0        | 9.42              | 9.2%      | 4.30               | 5.14           | 303              |
| four_1600   | 0        | 11.08             | 8.4%      | 4.34               | 6.65           | 303              |
| mixed_six   | 0        | 10.36             | 7.5%      | 4.55               | 6.01           | 303              |
| single_full | 253,952  | 10.19             | 3.5%      | 5.81               | 4.42           | 363              |
| single_1600 | 253,952  | 9.97              | 4.1%      | 4.39               | 5.57           | 303              |
| four_1600   | 253,952  | 11.58             | 1.6%      | 4.69               | 6.81           | 303              |
| mixed_six   | 253,952  | 11.62             | 2.5%      | 5.03               | 6.56           | 303              |

Program time is the union of native program intervals on the chip with the longest span; it includes communication and waits inside kernels. Gap time has no active program on that chip. These are independently computed medians, so their differences need not sum exactly to the difference of median spans. They separate added device work from time between programs, not compute from communication.

The hybrid implementation adds persistent Q/K/V copies, output clears, and residual copies/layout conversions around trace boundaries. Both paths use the same striped SDPA tuning. Keeping the residual in its existing sharded layout and writing projections directly into persistent buffers are candidates for reducing the added program time; this run does not isolate each copy's cost.

Hybrid host timings with the lightweight profiler callback differ from the separate callback-free timings by -0.34–0.52 ms. Python record parsing runs after measurement.

The real-time profiler narrows runtime IDs to 16 bits and filters ID zero. A wrapped eager ID can therefore remove one record per chip, and eager IDs can overwrite kernel-source metadata for captured traces. The sampler rejects incomplete data and retries once only for the uniform missing-record/ID-wrap signature. It validates fixed trace IDs and eager kernel-source order separately. The final dataset contains only complete, validated captures; `collection_attempts` records retries. The underlying native profiler is unchanged.

Loading, compilation, capture, input/metadata staging, and warmups are excluded. Legacy timing sums one full padded chunk per request. KV history is synthetic; this benchmark does not check PCC. Numerical correctness is covered separately.

Validation: the full-model comparison passed all 24 cells in 370.55 seconds. Eleven sampler unit tests and a separate device test combining two traces with an eager operation also passed.

# Native projection-to-collective delivery

October 12, 2026 UTC (October 11 Pacific). **Measured model performance remains
20.884562 TSU at B16/32K/TP4**. This experiment establishes a small complete
projection-boundary improvement; it has not run a full model or GPQA.
The separately GPQA-qualified compact configuration remains 177/198 (89.39%).

The guarded, opt-in `projection_delivery="matmul_output"` route removes the
standalone redistribution between local projection and all-reduce. It retains
the existing activation senders, two readers per DRAM bank, K blocking, BFP8
weights, BF16 outputs, FP32 accumulation, collective output layout and reduction
arithmetic. Matmul output storage uses 40 shards of width 128 instead of eight
shards of width 640. Native all-reduce accepts the producer's different grid.
The producer uses the native row-wise grid; the collective output retains the
existing rectangular grid. Equal shard widths do not mean identical grids.

`projection_delivery="native_input"` is a second comparison: it feeds the
eight-shard output directly to all-reduce without changing matmul storage.
The default remains `"reshard"`. The new route is restricted to compact
`[1,1,B,K]`, B16/B32, `keep_sharded=True`, replicated residuals and the measured
GDN output / MLP down projections. Public rows, full-attention output, prefill,
B1/B8 and other configurations retain their existing route. Public-row fallback
was checked natively at B16/B32; this is not a new lower-batch or prefill sweep.

## Results

Final v4 complete-boundary timings, including input conversion, local matmul
and TP all-reduce, in microseconds:

| Batch / projection | Before-after control | Native input | Matmul output | Best latency change |
|---|---:|---:|---:|---:|
| B16 GDN output | 52.998 | 50.516 | 49.368 | -6.85% |
| B16 MLP down | 94.215 | 91.700 | 90.564 | -3.88% |
| B32 GDN output | 53.019 | 50.518 | 49.345 | -6.93% |
| B32 MLP down | 94.230 | 91.699 | 90.561 | -3.89% |

Compact control drift is below 0.074%. Public-layout fallback is unchanged
within timing noise (at most 0.015%). Each arm uses five timing samples of
100 replays, with 16 projection calls captured per replay. This amortizes host
submission overhead; compare within these matched brackets rather than with
earlier one-call trace measurements. Each timing arm has eight warmup replays.

Across 48 GDN outputs and 64 MLP down projections, v4 projects a **0.407892-ms
step saving**, approximately **+0.1794 TSU / 21.064 projected TSU** from the
47.882258-ms model baseline. This extrapolation excludes the 16 full-attention
output projections and assumes the isolated savings survive model integration.
The earlier v2 experiment independently projected 0.407878 ms. These are not
measured model results, are not additive to another experiment that removes
the same conversions, and do not close the 14.55-ms gap to 30 TSU.

The component gain is below 1 TSU, so no standalone full-model, G0/API or full
GPQA cycle was launched. Retain it for combined validation. Full GPQA remains
gated on measured B16/32K performance of at least 25 TSU.

## Correctness and failed attempts

All attempts ran on physical TP4 `[0,4,12,8]` inside one allocated Galaxy, using
real layer-0 checkpoint weights and synthetic, seeded, rank/user-distinct
inputs. Existing quantized-weight dense reference checks passed: PCC >=0.999
and relative RMS <=0.015. Each successful arm additionally requires exact
baseline output on every rank, changes the captured input A/B/A at the same
address, checks the original result is restored, and checks source inputs and
weight addresses remain unchanged. B16/B32 compact cases cover both projections.
Final v4 adds B16/B32 public-row output fallback, for 24 cases total.

`TT_METAL_TRACE_ALLOC_TRACKING=1` was active in every attempt; v2-v4 also enable
allocation tracebacks. No corruptible-allocation suppression was used.

| Attempt | Result | Meaning |
|---|---|---|
| v1 | 13 CPU tests passed; first hardware baseline stopped before timing | Tracker found two live program-cache allocations after capture, from copy and sharded-to-interleaved reference readback. No candidate result. |
| v2 | 13 CPU tests and 16 native cases passed | Warmed copy/readback programs before capture, moved independent local-reference checks before capture and explicitly freed their device tensors. Allocation tracking remained enabled. |
| v3 | 13 CPU tests passed; hardware setup failed | New fallback guard sliced native `Shape`, whose pinned API only supports integer indexing. No timing result. |
| v4 | 13 CPU tests and 24 native cases passed | Converted `Shape` to a tuple before slicing; checked both compact candidates and unchanged public-layout fallback. |

The combined setup/lifetime change resolved v1's tracking failure. This does
not isolate the contribution of each setup change or prove every older model
trace warning is fixed. v3 was a Python/native API compatibility error. Neither
failure was a device hang. All attempts closed the devices; no reset occurred.
The successful services were already garbage-collected at final collection.
Their controller, JUnit and native PASS receipts establish completion; the
default Result field of a missing unit is not used as proof.

## Reproduction and provenance

From the repository root:

```sh
python3 models/demos/qwen38_27b_qb2/galaxy-evidence/projection-delivery-results-v1/summarize.py
```

This verifies all 60 raw hashes, final tested implementation hashes, JUnit,
case coverage, exact-output checks, before/after drift and timing arithmetic,
then regenerates [summary.json](summary.json). Raw Python, XML and logs are
compressed before formatting so their sealed contents remain unchanged.
The source manifests cover the frozen model tree, including precision config.

Native Metal pin: `a08819ddbe23077f8037d3802303939064868ff6`.
Checkpoint: `checkpoint-pinned-1d4bf0f2` on host `10.228.203.98`.
Final frozen source:
`/home/ttuser/qwen38-artifacts-20261007/projection-delivery-source-v4`.
Final service: `qwen38-projection-delivery-v4-20261011.service`.
Its invocation ID is `1a01b4147dbe4217878bac6daa88980a`.

Exact environment and launch arguments are in
[launch.json](raw/projection-delivery-control-v4/launch.json).
The persistent controller owns `/tmp/tt-device.lock` through the safe pytest
wrapper, with a 900-second pytest timeout, 1200-second process-group bound,
1800-second service limit and 64-GiB memory cap. It survives SSH disconnection;
it does not resume after reboot. Hardware pytest finished in 35.56 seconds.
No serving defaults, precision, firmware or installed native library changed.

Follow-up: combine this candidate with a larger independent gain, then measure
the actual model. The next larger opportunity remains GDN operand delivery or
SDPA scheduling; neither has a new measured gain from this experiment.

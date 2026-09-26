# Same-run timing reconciliation

The fused baselines now have four complementary timing quantities from each
profiled invocation. All times below are microseconds per decode. Workload:
batch/concurrency 1, prefill 4096, 128 successive positions 4096..4223.

| Layer | Estimated bytes / 512 GB/s | Complete device window, mean | Refreshed 128-step host loop, mean | Fixed-final-position host replay, median |
| --- | ---: | ---: | ---: | ---: |
| Sliding, layer 0 | 1194.195 | 5146.563 | 5164.833 | 5154.617 |
| Full, layer 5 | 1381.001 | 5586.547 | 5605.655 | 5594.646 |

The theoretical terms use 611,427,644 and 707,072,292 estimated DRAM bytes per
decode respectively, divided by the stated decimal 512 GB/s peak. They are
traffic-only roofline estimates, not memory-controller measurements or
complete-layer runtime predictions. Complete-device-window time includes every
layer operation and the gaps between its first and last firmware timestamps.

The exact Tracy host signpost pairs are:

| Layer | `PERF_DECODE`, ns | `PERF_DECODE_END`, ns | Total host span, ns |
| --- | ---: | ---: | ---: |
| 0 | 19,205,816,150 | 19,866,914,773 | 661,098,623 |
| 5 | 20,922,849,420 | 21,640,373,229 | 717,523,809 |

The CSV `HOST START TS` values exactly match the `TT_SIGNPOST` message
`total_ns` fields in each raw `tracy_ops_data.csv`. Dividing each elapsed host
span by 128 gives the same-loop host means above. The runner synchronizes before
the opening signpost and before the closing signpost. The measured span includes
input/position refresh, host-to-device copies, trace submissions, inter-replay
intervals, final synchronization and signpost boundary costs; it excludes HF
reference preparation and output readback. No harness change or new benchmark
was needed to recover this duration.

Same-loop host minus device is **18.270 us** sliding and **19.108 us** full.
Fixed-position host minus successive-position device is **8.053/8.099 us**.
The former compares the full refreshed loop against its individual device
windows. The latter compares different execution regimes within the same
process: after profiling successive inputs, the runner measures five batches of
30 replays at the final position, without refreshing inputs. Neither difference
isolates Python, dispatch, transfer, synchronization or contention cost. Host
and device work can overlap. No particular contention explanation is supported
by these differences alone.

The device-minus-theoretical differences are **3952.369/4205.546 us**. The pure
DRAM estimate omits a model for compute/SFPU work and program gaps; the complete
device windows include that work. The differences cannot partition those
components or establish a single bottleneck.

Older unprofiled host medians, **5067.964/5519.467 us**, come from
`actual_text_fused_layer{0,5}_4096.json`. They are kept as separate observations
and are not subtracted from device timings belonging to
`profile_actual_fused_layer{0,5}.json`. Mixing those invocations created the
apparent negative host/device gap. Instrumentation and execution conditions
differ, so that subtraction would not measure host overhead.

`tests/reconcile_perf.py` generates the durable artifacts:

- `tracy/actual_fused_layer0/timing_reconciliation.json`
- `tracy/actual_fused_layer5/timing_reconciliation.json`
- `timing_reconciliation_commands.json` records the exact CPU commands.

The helper checks matching workload geometry and replay/native-operation
counts. It requires the profile command journal, matches the supplied runner
JSON to exactly one successful `--profile --timing` invocation, and verifies
that the summary CSV is byte-identical to that command's raw Tracy CSV. Every
source has a SHA256. Older unprofiled results are recorded separately when
supplied. For final profiles, run from the repository root:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/tests/reconcile_perf.py \
  /path/to/whole_layer.json --runner-json /path/to/profile_result.json \
  --profile-command-journal /path/to/profile_commands.json \
  --output /path/to/timing_reconciliation.json
```

Clock assumption: compare elapsed durations only. Never subtract absolute host
timestamps from device cycles. Device cycles use the conversion recorded by
`summarize_perf.py`; signpost spans use Tracy host nanoseconds; fixed-replay
durations use the runner's `perf_counter_ns`. Negative arithmetic differences
are retained without clamping.

CPU verification matched both raw signpost pairs, rejected an older unprofiled
JSON as the same-run result, rejected a missing closing signpost, and verified
negative differences remain visible. Black and Python compilation passed. No
TTNN import, hardware execution, runtime change or extra benchmark occurred.

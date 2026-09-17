# Early-exit cost attribution against PR #56632

## Finding

**Preparation's fixed work assignment is the main source of the recurrence
slowdown.** Skipping padded chunks leaves the busiest cores with the same work
as a full-capacity call. The PR, running just the valid tokens, distributes that
smaller workload evenly.

At 4096 valid tokens, preparation takes **844.0 µs versus the PR's 668.9 µs**:
+175.1 µs. The total warm trace recurrence gap is about +150.4 µs (+10.65%);
other operations offset part of the preparation cost. At 4896 tokens,
preparation adds 49.6 µs, again partly offset by other stages.

This comparison uses **the actual published PR head**, not an omitted-end call
on the prototype branch:

- PR: `84ae832f1787fc1b8495e9a1b40f8090601a658d`, isolated checkout
  `/localdev/mvasilijevic/tt-metal.worktrees/kda_pad_pr_baseline`.
- Prototype: `57e03d621b9970b5f2c870e67e9df9d9c4675728`, checkout
  `/localdev/mvasilijevic/tt-metal.worktrees/kda_pad_runtime`.
- Synthetic Kimi-K3, Blackhole SP1×TP8, 12 local heads, K=V=128, 110 worker
  cores. Prototype capacity stays 5120; PR physical length equals valid length.

No operation implementation was changed for this investigation. The benchmark
source copied into the baseline checkout was byte-identical to the prototype
harness (SHA-256 recorded in the [raw results](KDA_EARLY_EXIT_ATTRIBUTION.json)).

## Direct PR comparison

Wall milliseconds per trace replay. Each case warms 400 replays, then records
16 samples of 16 replays plus synchronization. The PR is measured before and
after the prototype in separate processes. Percentages compare the prototype
median with the median of the pooled PR A/B samples; they are not paired
same-process ratios.

| Stage | Valid tokens | PR A ms | PR B ms | Prototype ms | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| recurrence | 4096 | 1.4125 | 1.4123 | 1.5629 | +10.65% |
| recurrence | 4896 | 1.6436 | 1.6409 | 1.6659 | +1.43% |
| recurrence | 5120 | 1.7078 | 1.7062 | 1.6894 | -1.01% |
| layer | 4096 | 10.0804 | 10.1308 | 11.8358 | +16.90% |
| layer | 4896 | 11.7432 | 11.8113 | 12.0297 | +1.91% |
| layer | 5120 | 12.0193 | 12.0226 | 12.0873 | +0.56% |

The recurrence baseline repeats agree within 0.17%. At full length, recurrence
is about 1% faster, so this is chiefly a loss of padding efficiency, not a
uniform penalty from adding runtime bounds.

Layer measurements need more care: AICLK sampling during sustained runs observes
clocks falling from 1350 MHz into approximately 1020–1200 MHz. Idle samples can
be 800 MHz. Both PR repeats remain close, but the small +0.56% full-layer wall
difference is not established as an added-kernel cost: its summed device-kernel
time is slightly lower on the prototype. Clock samples and full timing sequences
are retained; no thermal cause or fixed clock is assumed.

## Where the device time goes

These are per-operation **device kernel** durations, separate from synchronized
wall time. Both revisions use the same eager profiling protocol: two warmups,
then three marked invocations on eight devices. Categories are summed within an
invocation/device, then medians are taken over 24 observations. Exported device
nanoseconds use the profiler's nominal 1350 MHz conversion. These values must not
be presented as an exact additive decomposition of warm trace wall latency.

### Recurrence, 4096 valid tokens

| Operation | PR µs | Prototype µs | Difference µs |
| --- | ---: | ---: | ---: |
| Preparation | 668.9 | 844.0 | **+175.1** |
| Beta reshape/materialization | 56.8 | 70.9 | +14.1 |
| Group summary | 244.9 | 252.5 | +7.6 |
| Exclusive scan | 69.1 | 65.0 | −4.1 |
| Final recurrent scan | 313.7 | 291.5 | −22.2 |
| Summary typecasts, combined | 20.0 | 3.9 | −16.1 |

The other transpose/slice differences are below 1 µs. The sum of all device
kernel durations increases by 156.6 µs, close to the separately measured
150.4 µs wall gap.

### Why preparation stalls at almost full cost

The [factory](../../../../../ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/prepare_chunk_recurrence_program_factory.cpp)
assigns contiguous ranges of `num_heads * physical_num_chunks` to cores through
`distribute_prep`. The [reader](../../../../../ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/dataflow/reader_prepare_chunk_recurrence.cpp)
and compute/writer kernels then skip indices whose physical chunk is invalid.
They retain the original ranges; useful work is not redistributed.

| Valid tokens | PR total valid head/chunks | PR busiest core | Prototype busiest core | Preparation PR → prototype |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 1536 | 14 chunks | 18 chunks | 668.9 → 844.0 µs |
| 4896 | 1836 | 17 chunks | 18 chunks | 798.9 → 848.5 µs |
| 5120 | 1920 | 18 chunks | 18 chunks | 844.5 → 846.6 µs |

For 4096 tokens, ten prototype preparation cores have **zero** valid chunks,
while 37 still have 18. The PR gives 106 cores 14 chunks and four cores 13.
Preparation duration tracks the busiest-core count at roughly 47 µs per chunk.
The full-length preparation difference is only 2.1 µs; the +175.1 µs short-case
difference is overwhelmingly the retained work assignment.

This is observed timing plus source-level work-count evidence, not an ablation
of a replacement scheduler. An implementation fix was deliberately not included
in this diagnostic task.

### Other recurrence effects

- Beta layout preparation still operates on all 5120 physical rows, explaining
  its larger materialization workload. The 4096-token device difference is
  +14.1 µs; at 4896 it is +2.3 µs.
- Grouping is physically fixed: eight groups of 20 chunks. The PR chooses eight
  groups of 16 at 4096 and nine groups of 17 at 4896. This changes parallelism,
  but it is **not the dominant measured regression**. In particular, final scan
  is faster in the 4096 prototype. Its speedup's precise cause was not isolated;
  a simple “20 versus 16 iterations” latency prediction would be wrong.
- Runtime chronology selects the PR's existing compact BF16 summary transport.
  The ordinary local PR path writes FP32 summaries before typecasting to BF16.
  The prototype reduces the combined typecast cost by about 16 µs at all three
  lengths. This offsets some losses and largely accounts for the full-length
  recurrence improvement. The dtype distinction is explicit in
  `recurrent_chunk_scan/device/recurrent_chunk_scan_device_operation.cpp`,
  under `compact_summary`.

At 4896 tokens, preparation is +49.6 µs; final scan +10.0 µs; summary +3.6 µs;
beta materialization +2.3 µs. Exclusive scan saves 11.8 µs and typecasts save
16.3 µs. The summed device-kernel gap is 37.4 µs versus a warm wall gap of
approximately 23.5 µs. The timing domains are different; that residual is not
assigned an invented cause.

## Whole-layer cost

The layer still projects, convolves, normalizes, and communicates **5120 rows**.
Early exit was implemented only for recurrence plus valid-end state selection.
For 4096 valid tokens, the largest extra device-kernel costs versus the PR are:

| Category | Additional µs |
| --- | ---: |
| Matmuls, combined | +422.3 |
| QKV convolution | +218.3 |
| Output reduce-scatter | +186.9 |
| Recurrence preparation | +174.4 |
| Slices | +67.1 |
| Untilize | +45.3 |
| Gated RMS normalization | +31.8 |

The summed device-kernel difference is approximately +1.150 ms. The sustained
wall difference is larger, approximately +1.71 ms, and includes the clock/timing
domain effects described above. The new chronology and history-selection
kernels themselves add only about 5.5 µs of device time.

## Implication for a fix

First rebalance preparation over **valid logical head/chunks** on device, then
map each logical chunk back to its existing physical tensor address. All three
cooperating kernels must derive the same worker range from the runtime bound.
The launch shape and tensor placement can stay fixed. More early-exit checks
alone cannot move work away from the busiest cores.

Beta materialization is the next smaller recurrence target. Revisit summary/scan
grouping only with measurements: the current group-size difference alone does
not explain the observed timings. Matching a physically trimmed whole layer
would also require avoiding padded work in the surrounding layer operations.

## Validation, provenance, and reproduction

- Isolated PR native build passed with
  `./build_metal.sh --enable-ccache --build-ttnn-tests`, exit 0, 7m34.908s.
  Release, clang/clang++ 20, Tracy enabled, matching the prototype configuration.
  PR checkout HEAD and upstream both matched the published SHA; imports were
  contained in that checkout. Only the standalone benchmark was added there.
- PR A wall run: 6 passed, 130.41s. Prototype wall: 6 passed, 81.05s.
  PR B wall: 6 passed, 80.46s. All safe-wrapper PASS, no hangs or resets.
- Prototype profile: 6 passed, 132.24s. PR profile: 6 passed, 147.62s.
  Both safe-wrapper PASS with explicit pytest success, not just Tracy exit 0.
- 2664 prototype and 2520 PR marked device-program records were extracted from
  native profiler output and matched exactly against the generated canonical
  operation CSVs. The additional records are the two extra layer selection
  operations, across three lengths × three repeats × eight devices.
- The profiler builds kernels on real devices. It reported C++/SFPU deprecation
  and profiler instrumentation messages; Python SWIG/Pydantic deprecations and
  mixed-column pandas warnings were observed. Build warnings included CMake
  dependency deprecations/version-policy warnings, Tracy `fread` unused-result,
  and a Tracy WASM constant-conversion warning. Existing layout/allocation/L1
  semaphore/fabric-packet warnings are preserved in complete logs.

Use the [portable harness](perf/test_padding_cost_attribution.py) identically in
both checkouts. From each checkout, activate its own environment and set
`TT_METAL_HOME` and `PYTHONPATH` to that checkout. Set `KDA_COST_VARIANT=pr` in
the PR checkout and `early` in the prototype:

```bash
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/perf/test_padding_cost_attribution.py -sv
KDA_COST_PROFILE=1 scripts/run_safe_pytest.sh --profile models/demos/deepseek_v3_d_p/tests/kda/perf/test_padding_cost_attribution.py -sv
```

All aggregate and raw wall samples are in
[KDA_EARLY_EXIT_ATTRIBUTION.json](KDA_EARLY_EXIT_ATTRIBUTION.json). Complete logs,
Tracy captures, native records, canonical CSVs, extraction scripts, and clock
samples are retained in `generated/kda_cost/` in the prototype checkout. The
canonical profile CSV timestamps are `2026_09_17_20_09_51` (prototype) and
`2026_09_17_20_23_35` (PR).

Scope: these findings explain the reported SP1×TP8 cases. They do not establish
SP>1 performance or prove the performance of a proposed replacement scheduler.

# KDA prefill offsets: contract and measured performance

Historical performance measurements: Blackhole, 2026-09-15.
Current cleanup and validation: `tt-metal_tracker-mf3`.
Tracking: `tt-metal_tracker-ea6` and children `.5`, `.6`, `.7`, `.13`

## Cleanup validation

`tt-metal_tracker-mf3` passed 952 final test executions on eight Blackhole devices
at implementation revision `693a0ecc8541b400eb530d9b0ea943e8b8972ac5`, with zero
failures or skips. The matching isolated host build/install passed. All device
runs used `scripts/run_safe_pytest.sh` and reconciled JUnit outcomes.

| Suite | Passed | Pytest seconds |
| --- | ---: | ---: |
| Topology oracle and layer contracts | 506 | 123.426 |
| Public operations and numerical-policy utilities | 329 | 100.083 |
| Local recurrence and stateful layer | 24 | 16.579 |
| Multi-device components and all offset tests | 39 | 112.275 |
| Synthetic/real-weight acceptance and checkpoint loading | 9 | 169.488 |
| Isolated cold / warm specialization | 18 / 18 | 18.890 / 4.085 |
| Performance and policy | 8 + 1 | 70.819 + 0.247 |

The offset suite includes all 160 aligned starts at C2560/G4, both SP axes,
nonzero-carry continuation, and trace replay. Production acceptance retained the
PCC ≥0.9995 gate; the lowest reported PCC was 0.999758. The cold run had 0/48
persistent JIT cache hits, and the identical warm selection had 48/48.

Real-weight median trace wall times at T5120 were 9.629093 ms (SP1xTP8),
9.564964 ms (SP2xTP4), and 9.988273 ms (SP4xTP2). Synthetic SP2xTP4 measured
9.585874 ms. All existing two-sided ±3% performance gates passed unchanged.
Actual operation dispatch retained one summary, one affine scan, and one
recurrent scan, with G4 at T5120 and G1 at T1280 for baseline and split.

The final paired offset run showed approximately 19–21% T5120 wall-time spread;
its sub-percent paired deltas do not establish a precise cost. T1280 split
cost was approximately 2.9%, consistent with the earlier measurements below.
Named device-program measurements remain separate from trace wall time.

SP8xTP4 Galaxy requires 32 devices and was explicitly excluded locally. Its CI
selection and host performance-policy test are covered; no local Galaxy
hardware result or completed upstream CI run is claimed.

## Historical performance verdict

The split-specific single-group fallback is gone. At production `C=2560`,
baseline, rotation, and every split now retain four 20-chunk recurrence groups
and execute exactly one summary, one reduction, one SP prefix, one affine group
prefix, and one recurrent scan.

This removes the former 5k performance failure. The original G=1 split path was
17.4-20.5% slower in a drift-heavy run. With G=4 restored, the rotated
round-robin measurement gives median paired deltas of **+0.17% to +0.82%** for
the three splits. The run itself drifted by about 17%, so those sub-percent
wall-time differences are below measurement resolution; the defensible claim
is that the old large penalty is absent, not that a precise sub-percent cost has
been established.

At `C=640`, where baseline and split both use one group, the result is stable:
all split sizes cost **about +3.0%** with less than 1.0% run spread. Named device
programs pin this delta on boundary summary/seed handling and returned-state
plumbing, not on the main matmul or QKV convolution.

## Terminology and cases

`C` is the number of sequence rows resident on each SP device:
`C = sequence_length / SP_size`. Thus the 5k/SP2 case has `C=2560`, while the
1280/SP2 control has `C=640`. KDA chunks contain 32 rows.

- Baseline: offset 0; physical and causal order already agree.
- Rotation: offset equals `C`; causal order starts on another device but no
  device contains a wrap.
- Small split: offset 32; the boundary device has a 32-row head/tail fragment.
- Medium split: offset `C/2`; the boundary lies halfway through the device.
- Large split: offset `C-32`; the opposite 32-row extreme.

The table reports five rotated round-robin samples, each averaging ten warm
trace replays. “Paired delta” compares each offset with the baseline sample from
the same round and takes the median; it is the useful statistic under monotonic
machine drift.

| C / sequence | Case (offset) | Median trace wall | Median paired delta | Run spread |
| ---: | --- | ---: | ---: | ---: |
| 2560 / 5120 | baseline (0) | 11.1650 ms | 0.00% | 17.01% |
| 2560 / 5120 | rotation (2560) | 11.2857 ms | -0.03% | 16.69% |
| 2560 / 5120 | small split (32) | 10.7536 ms | +0.82% | 17.31% |
| 2560 / 5120 | medium split (1280) | 10.9507 ms | +0.17% | 17.01% |
| 2560 / 5120 | large split (2528) | 11.1194 ms | +0.58% | 16.09% |
| 640 / 1280 | baseline (0) | 3.2452 ms | 0.00% | 0.55% |
| 640 / 1280 | rotation (640) | 3.2489 ms | +0.04% | 0.60% |
| 640 / 1280 | small split (32) | 3.3431 ms | +3.02% | 0.59% |
| 640 / 1280 | medium split (320) | 3.3449 ms | +3.01% | 0.99% |
| 640 / 1280 | large split (608) | 3.3428 ms | +3.01% | 0.61% |

## Implemented dataflow

The boundary device emits fixed-shape head and tail affine summaries for every
physical group in one pass. Only heads enter the unchanged reduction and SP
prefix. The SP prefix's final carry is the tail seed. One device-local affine
exclusive scan injects a constant reset transform immediately before the first
tail-only group; this is algebraically the requested two-seed prefix without a
second scan. The straddling group keeps its head-derived entry and reloads the
tail seed once at its local wrap. All groups then run in parallel in one
`recurrent_chunk_scan`.

Production K/V dimensions initially exceeded L1 when four groups were enabled.
The summary now streams the straddling head A/B snapshots one K row at a time,
reducing those two buffers from a full K×V payload to one K-row payload each.
The runtime wrap selector is one FP32 tile rather than K tiles. The final scan
state selector reuses existing scratch/ring buffers instead of materializing
three extra K×V tensors.

The convolution path remains device-local: one small pack publishes the
predecessor's three carry rows and the boundary's physical final state through
one SP all-gather. The recurrent final state is still gathered and sliced so
the public `KdaState` remains replicated across SP.

Key implementation locations:

- `tt/kda/recurrence.py`: G selection, head-only
  reduction, one affine prefix, one recurrent scan, and final-state replication.
- `ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/`:
  one-pass segmented summary, streamed snapshot, and one in-group state reload.
- `ttnn/cpp/ttnn/operations/experimental/kda/affine_exclusive_scan/device/`:
  constant-reset transform and head/tail input selection in one scan program.
- `tt/kda/convolution.py` and
  `qkv_causal_conv1d_silu/device/`: one-tile carry packing and wrap-aware
  convolution state.

## Device-profile attribution

Profiles cover all eight chips and report no incomplete program sequences.
Durations below are per-program maxima across chips; programs can overlap and
must not be summed into a wall-time prediction.

For `C=2560`, baseline versus medium split:

- summary: 298.6 us vs 299.5 us (**+0.8 us**);
- reduce: 41.1 us vs 41.1 us (**+0.1 us**);
- affine group prefix: 70.4 us vs 78.0 us (**+7.6 us**);
- recurrent scan: 344.9 us vs 344.5 us (**-0.4 us**);
- QKV convolution: 1109.9 us vs 1113.9 us (**+4.0 us**);
- main projection matmul: 3751.1 us vs 3750.4 us (**-0.7 us**);
- split-only convolution pack: 9.1 us.

For `C=640`, baseline versus medium split:

- summary: 229.5 us vs 255.2 us (**+25.8 us**);
- affine group prefix: 19.0 us vs 28.0 us (**+8.9 us**);
- recurrent scan: 266.0 us vs 274.0 us (**+7.9 us**);
- QKV convolution: 312.2 us vs 308.9 us (**-3.3 us**);
- main projection matmul: 1045.0 us vs 1045.9 us (**+0.9 us**);
- split-only convolution pack: 10.0 us.

Both profiles contain one program each for segmented summary, affine prefix,
and recurrent scan. There is no second scan and no offset-specific affine
matmul/select graph. The remaining split-only topology is the local convolution
pack plus one additional all-gather and two small data-movement programs for
replicating the final recurrent state. At `C=640`, the 25.8 us summary delta and
8.9 us affine-prefix delta are the largest named changes; the main matmul and
convolution are flat. At `C=2560`, restoring four groups makes segmented-summary
overhead negligible and leaves only a small affine reset cost in the recurrence
core.

## Correctness and validation evidence

- All 160 32-aligned offsets at `T=5120`, SP2×TP4, `C=2560`, G=4 passed the
  natural-order output and both-state gates in 69.97 s. Representative values:
  output PCC 0.999940, recurrent PCC 0.999904, convolution PCC 0.999997.
- Seven `T=10240`, SP4×TP2, `C=2560`, G=4 placements passed in 14.12 s,
  including group boundaries, straddles, extremes, and a second boundary rank:
  output PCC 0.999942, recurrent PCC 0.999897, convolution PCC 0.999997.
- Segmented summary covers G=1,2,4 and first/before/on/after/final group-wrap
  positions against an independent host affine oracle. Cache-hit, ordinary
  bit-identity, and trace replay tests pass.
- Segmented affine prefix covers G=1,2,4, exact boundaries and straddles, with
  head/tail seeds separated by five orders of magnitude. Cache rebinding and
  trace replay tests pass.
- Focused recurrent tests cover early/late/in-group reloads and untouched
  ordinary devices; output PCC is 0.999990-0.999998 and final-state PCC is
  0.999997-1.0.
- The complete summary, affine-prefix, and recurrent-scan device-op suites pass:
  143 tests in 92.82 s.
- Complete split-layer trace replay is bit-identical for output, recurrent
  state, and convolution state (1 passed in 0.39 s device-test time).
- `cmake --build build --target ttnn/install -j24` passed.
- Performance command:
  `scripts/run_safe_pytest.sh --run-all -q -s models/demos/deepseek_v3_d_p/tests/kda/perf/test_offset_perf.py --tt-arch blackhole`
  — 2 passed in 25.29 s. The test gates the one-summary/one-prefix/one-scan
  topology and prints raw samples plus named device-program timings.

## Residual risk

The 5k host-wall samples are dominated by machine drift. The named device
profiles and restored G=4 topology prove removal of the old serialization, but
a tighter performance percentage requires a thermally stable lane or longer
randomized sampling. No claim stronger than “the prior 17-20% penalty is gone”
is supported by this run.


## Supported offset contract

The caller supplies the MLA block-cyclic activation for one complete prefill
chunk, a nonnegative 32-aligned absolute start, and immutable recurrent and
three-row convolution carries replicated on every SP rank within each TP line.
Stream continuity is the caller's responsibility. Start zero is the default.
A nonzero rank boundary rotates the chronological chip order; an in-chip split
keeps its physical head and tail and retains the ordinary group geometry.

The layer owns construction-time selector tensors and chooses the boundary rank
once per forward. Convolution and recurrence consume the same selector. The
private SP routes receive a normalized topology and axis. Public experimental
TTNN operations remain independently validated trust entries.

Summary modes are ordinary full-group `(A, B)` and segmented
`(head_A, head_B, tail_A, tail_B)`. Segmented mode requires an indicator and a
strictly interior wrap boundary. Ordinary mode rejects wrap controls. Public
summary ranges and head-only mode are removed. Recurrent scans retain
unconditional wrap with a tail state and no indicator.

The local suite validates SP2xTP4 and SP4xTP2 production offsets on eight
Blackhole devices. The separately selected SP8xTP4 Galaxy CI case requires
32 devices; local eight-device evidence does not establish that topology.

Historical prototype comparisons are preserved in Git history at
`4d2c3bb3d57c7628c1cf671005a61143bac8ef18`; they describe superseded implementations.

# AutoFix Report: fused mutable trace inputs

## Starting Evidence

- Source: `doc/fused_decoder/AUTODEBUG.md`, Finding 1.
- Original gate: `doc/fused_decoder/stage_review.md`, P1 mutable-input finding.
- Hypothesis: the fused suite does not execute the existing changed-token,
  uint32 RoPE-position, and int32 cache-index trace regression through
  `FusedDecoder`.
- Static verification: `test_changed_trace_buffers_random_and_boundaries` had
  only a `FunctionalDecoder` construction site, while the fused suite wrapped
  other functional regressions but not this one.

## Hypothesis Experiment

- Experiment: add the same thin monkeypatch wrapper used by neighboring fused
  inherited tests and run its complete 2 layer kinds x 2 position cases.
- Ordinary command:

  ```bash
  LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH} MPLCONFIGDIR=/tmp/matplotlib \
    pytest -sv models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py::test_fused_changed_trace_buffers_random_and_boundaries
  ```

- Ordinary result: 4 passed in 24.13 seconds. The reference-B PCC values were
  0.998794, 0.999429, 0.999599, and 0.999385. Correct/stale RMSE pairs were
  0.020830/0.570989, 0.010361/0.383111, 0.021021/0.510991, and
  0.012503/0.443649. All correct RMSE values beat the stale-A negative control,
  and all repeated B replays were bitwise equal.
- Watcher command:

  ```bash
  LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH} MPLCONFIGDIR=/tmp/matplotlib \
    TT_METAL_WATCHER=10 \
    TT_METAL_LOGS_PATH=$PWD/models/autoports/google_gemma_4_31b/doc/fused_decoder/mutable_trace_watcher \
    pytest -sv models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py::test_fused_changed_trace_buffers_random_and_boundaries
  ```

- Watcher result: 4 passed in 26.06 seconds with the same PCC/RMSE results.
  Watcher attached, checked the device during execution, detached, and the
  device closed cleanly. Its device log contains no fatal, assert, NoC/L1
  error, overflow, sanitizer, or error signature.
- Verdict: **verified coverage defect; fused runtime semantics pass**. The
  implementation does consume the mutated captured allocations, so no
  implementation repair is warranted.

## Fix And Exact Diff

Only `tests/test_fused_decoder.py` changed. The durable regression is:

```diff
+@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
+@pytest.mark.parametrize("position_case", ["random", "window_wrap"])
+def test_fused_changed_trace_buffers_random_and_boundaries(
+    monkeypatch, hf_config, mesh_device, layer_idx, layer_kind, position_case
+):
+    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)
+    functional_tests.test_changed_trace_buffers_random_and_boundaries(
+        hf_config, mesh_device, layer_idx, layer_kind, position_case
+    )
```

`git diff --check` passes. Final test SHA-256 is
`a5ebbccda7465f9eaee423b04a350ec54842e7c6777e85b713eb7ab3ca2195e8`.

## Evidence Artifacts

- `mutable_trace_focused.log`: ordinary four-case run.
- `mutable_trace_watcher.log`: watcher four-case console log.
- `mutable_trace_watcher/generated/watcher/watcher.log`: watcher device log.

## Final Status

Fixed. The fused suite now durably proves mutated token, RoPE-position, and
cache-index replay for sliding/full attention at random non-block-aligned and
1023-to-1024 boundary positions. Remaining uncertainty is limited to cases not
represented by this inherited four-case matrix; no broader suite or other
Stage 02 hypothesis was in scope for this AutoFix.

---

# AutoFix Experiment: long-prefill GELU fusion

## Starting Evidence

- Source: `doc/fused_decoder/AUTODEBUG.md`, Finding 2, and
  `doc/fused_decoder/stage_review.md`, long-prefill GELU P1.
- Hypothesis: the local `m_tiles <= 4` guard is not a TTNN legality rule, and
  long `M=4096, K=5376, N=21504` may legally and profitably fold approximate
  GELU into an explicit 1D gate matmul.
- Prediction: F4 admits, matches B0 at PCC >= 0.99, removes the standalone GELU
  row, and reduces warmed gate latency.

## Focused Experiment

The env-gated component node
`test_long_gelu_real_weight_gate_candidate` loads the real layer-0 BF16 shared
MLP weights, creates one deterministic BF16 `M=4096` input, and compares B0
against exactly one selected candidate. It retains both outputs for PCC, then
collects six warmed synchronized samples per arm in ABBA order. The initial
pre-device harness attempt selected the wrong checkpoint prefix
(`shared_mlp.` instead of `mlp.`), produced a `None` weight, and was corrected
before any candidate admission result was recorded.

Exact ordinary command, with `F4` replaced by `F2` and `F1` for the two
requested adaptations:

```bash
LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-} \
MPLCONFIGDIR=/tmp/matplotlib GEMMA4_LONG_GELU=F4 \
pytest -sv models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py::test_long_gelu_real_weight_gate_candidate
```

All candidates use grid 11x10, `in0_block_w=2`, subblock 1x7, block width 7,
`per_core_M=128`, `per_core_N=7`, `mcast_in0=True`, and config-level
approximate GELU. Only `out_block_h` changes:

| Candidate | Block height | PCC vs B0 | B0 median (us) | Candidate median (us) | Result |
|---|---:|---:|---:|---:|---|
| F4 | 4 | 0.9996926043 | 11195.8745 | 18636.2880 | admitted, +66.5% slower |
| F2 | 2 | 0.9996926043 | 11178.6845 | 36824.3625 | admitted, +229.4% slower |
| F1 | 1 | 0.9996926043 | 11189.5390 | 72886.7330 | admitted, +551.4% slower |

The repeated samples are in `candidates/long_gelu/{F4,F2,F1}/gate.log`.

## Op Topology

Exact profiler command:

```bash
LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-} \
MPLCONFIGDIR=/tmp/matplotlib GEMMA4_LONG_GELU=F4 \
GEMMA4_LONG_GELU_PROFILE=1 python -m tracy -r -p -v \
--output-folder models/autoports/google_gemma_4_31b/doc/fused_decoder/candidates/long_gelu/F4/profile \
-m pytest models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py::test_long_gelu_real_weight_gate_candidate -s
```

`tt-perf-report` was run directly on that raw CSV between `LONG_GELU_B0` /
`LONG_GELU_B0_END` and `LONG_GELU_F4` / `LONG_GELU_F4_END`. B0 contains a
10,113.030 us matmul plus a 914.847 us `UnaryDeviceOperation`. F4 contains only
one 18,480.290 us `MatmulDeviceOperation`; there is no standalone unary/GELU
row. Thus fusion is real, but the explicit 96-core 1D matmul is much slower
than the 110-core auto-selected B0 matmul even after removing GELU.

Artifacts are under `candidates/long_gelu/F4/profile/`, including the raw Tracy
report, signpost-filtered `B0_ops.csv` and `F4_ops.csv`, and console log.

## Verdict And Fix

The required adapted chunk ladder was then run one hardware command at a time.
Each row compares the fused candidate against same-M B0 with identical real
weights and deterministic data. Normalized totals multiply the warmed median
by the exact number of calls needed to cover 4096 rows, so dispatch/chunk count
is included:

| Candidate | M | `per_core_M` | Calls / 4096 | PCC | Same-M B0 median (us) | Fused median (us) | B0 normalized (us) | Fused normalized (us) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C2048 | 2048 | 64 | 2 | 0.9997687300 | 3116.9550 | 9362.2885 | 6233.9100 | 18724.5770 |
| C1024 | 1024 | 32 | 4 | 0.9997684004 | 1616.0025 | 4748.2970 | 6464.0100 | 18993.1880 |
| C128 | 128 | 4 | 32 | 0.9997683854 | 687.9870 | 644.3270 | 22015.5840 | 20618.4640 |

Exact command template:

```bash
LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-} \
MPLCONFIGDIR=/tmp/matplotlib GEMMA4_LONG_GELU=C2048 \
pytest -sv models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py::test_long_gelu_real_weight_gate_candidate
```

`C2048` was replaced by `C1024` and `C128` for the subsequent serialized
commands. Full samples and clean device close records are under
`candidates/long_gelu/{C2048,C1024,C128}/gate.log`. One C1024 B0 sample was a
16.489 ms outlier; the predeclared median remains 1.616 ms and is consistent
with the other five 1.570--1.715 ms samples.

Verdict: **legality verified; profitability refuted for the complete required
matrix**. C128 verifies the expected small-M fused control and improves its
same-M gate by 6.3%, but its 32-call normalized total is 20.618 ms. C2048 and
C1024 fused totals are 18.725 ms and 18.993 ms. All are substantially slower
than the directly measured current B0 M4096 total of 11.196 ms, before adding
the extra slice/concat work that a real chunked MLP needs. Therefore no fused
chunk family warrants a complete-MLP or decoder-prefill candidate.

No runtime change was kept: `_FusedSharedMLP` retains B0 for `m_tiles > 4`, and
the existing mutable-trace wrapper is unchanged. The only code addition is the
env-gated durable component experiment. Remaining uncertainty: untested
program families outside the prescribed 1D F geometry could behave
differently, but no evidence from this minimal matrix supports expanding the
search.

---

# AutoFix Experiment: fragile decode-candidate ordering

## Starting Evidence And Hypothesis

- Source: `AUTODEBUG.md` Finding 4 and `stage_review.md` Other Concerns.
- The former selected pre-projection batch slice measured 2556.330 us and the
  post-projection slice candidate measured 2557.140 us in separate one-sample
  runs. The claimed 0.810 us / 0.032% ordering was predicted to be noise.
- The focused experiment used one process, one sliding layer-0 decoder, the
  same real weights, deterministic token and position-32 input, the same paged
  cache/page table, separate captured trace IDs, three warmed replays per arm,
  and six ABBA cycles (12 measured replays per arm).

## Experiment And Results

Exact ordinary command:

```bash
LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-} MPLCONFIGDIR=/tmp/matplotlib \
GEMMA4_POST_PROJECTION_SLICE_AB=1 pytest -sv \
models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py::test_post_projection_slice_repeated_ab
```

The synchronized-host result passed with candidate-versus-selected PCC 1.0 and
bitwise determinism for both arms. Pre-projection median/min/max/spread was
2627.957/2603.097/2639.857/36.760 us; post-projection was
2626.502/2604.117/2630.897/26.780 us. The -1.455 us median difference favored
post-projection, but ABBA-cycle differences crossed zero, so this run refuted
the old ordering without selecting a winner.

The first profiler attempt used the command below with output folder `profile`.
It filled device-profiler buffers after eight complete pairs; its incomplete
tail was explicitly discarded. The complete rerun added one profiler-only
`ttnn.ReadDeviceProfiler(mesh_device)` after trace warmup and used output folder
`profile_complete`:

```bash
LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-} MPLCONFIGDIR=/tmp/matplotlib \
GEMMA4_POST_PROJECTION_SLICE_AB=1 GEMMA4_POST_PROJECTION_SLICE_AB_PROFILE=1 \
python -m tracy -r -p -v \
--output-folder models/autoports/google_gemma_4_31b/doc/fused_decoder/candidates/post_projection_slice_repeated/profile_complete \
-m pytest models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py::test_post_projection_slice_repeated_ab -s
```

The complete profiler run retained 12 complete 40-op intervals per arm and no
buffer-full warning. Device-kernel-sum results were:

| Variant | n | median us | min us | max us | spread us |
|---|---:|---:|---:|---:|---:|
| pre-projection | 12 | 2558.574 | 2555.949 | 2561.319 | 5.370 |
| post-projection | 12 | 2556.062 | 2552.758 | 2558.353 | 5.595 |

The candidate-minus-selected median difference was -2.512 us. All six paired
ABBA-cycle mean differences favored post-projection:
`[-2.475, -3.856, -2.015, -2.451, -2.767, -2.742]` us. The synchronized host
measurements in the same profiler run also favored post-projection in every
ABBA cycle, with a -3.785 us median difference. Verdict: the original
pre-projection lead is refuted, and the repeated device-paired evidence verifies
a small post-projection win.

## Fix And Verification

Temporary instrumentation consisted of an instance slice-placement flag, an
env-gated two-trace ABBA node, optional per-replay signposts, and a profiler-only
buffer flush. All instrumentation and the runtime knob were removed. The only
final implementation change moves the existing batch crop after `o_proj`; the
mutable trace wrapper and long-GELU candidate runner remain unchanged.

Final correctness command:

```bash
LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-} MPLCONFIGDIR=/tmp/matplotlib \
pytest -sv 'models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py::test_fused_real_weight_paged_decode_trace_pcc[0-sliding_attention]'
```

Result: passed; HF PCC 0.9996291558392226 and eight bitwise-identical trace
replays. Final performance command:

```bash
LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-} MPLCONFIGDIR=/tmp/matplotlib \
GEMMA4_FUSED_PERF=1 python -m tracy -r -p -v \
--output-folder models/autoports/google_gemma_4_31b/doc/fused_decoder/candidates/post_projection_slice_repeated/final_profile \
-m pytest 'models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py::test_fused_warmed_performance[decode-0-sliding_attention]' -s
```

Result: passed; 2557.519 us across 40 device ops and zero host ops in the final
signposted interval. Both commands closed the device cleanly. `git diff
--check` passes. Final SHA-256 values are `941dd1d16b64246111e8402d875cbb7fc1cc6bbf6b33465ff0ba97103df90af6`
for `tt/fused_decoder.py` and
`bef5374cd5ec09948fca3690d0ddbade9694cc4be748370acf41d49f79712b9d`
for `tests/test_fused_decoder.py`.

## Artifacts And Remaining Uncertainty

- `candidates/post_projection_slice_repeated/repeated_ab.log`
- `candidates/post_projection_slice_repeated/profile_complete_console.log`
- `candidates/post_projection_slice_repeated/profile_complete/`
- `candidates/post_projection_slice_repeated/final_pcc_trace.log`
- `candidates/post_projection_slice_repeated/final_profile_console.log`
- `candidates/post_projection_slice_repeated/final_profile/`
- `candidates/post_projection_slice_repeated/summary.md`

The literal gain is about 0.1%, and unpaired host ranges overlap. The verdict is
based on complete device-timed ABBA cycles, not the original noise-scale
single-sample ordering. If a larger margin is required, the smallest next graph
improvement is to fuse/eliminate the remaining crop at the
sharded-to-interleaved/head-concat boundary; no current local TTNN contract was
found that safely materializes only the logical batch rows, so no speculative
change was retained.

---

# AutoFix Experiment: advertised-context distinct-token fused decode

## Starting Evidence And Hypothesis

- Source: `stage_review_rereview.md` P1.
- The fused suite inherited exact-capacity same-token prefill/decode parity and
  short mutable trace coverage, but did not invoke the functional stage's
  genuine 262144-context distinct-token HF gate through `FusedDecoder`.
- Hypothesis: this was a test-coverage omission; the fused full-attention
  `paged_fused_update_cache` and sliding modulo-update paths would preserve the
  distinct final token, absolute position 262143, and deterministic trace
  replay at the advertised limit.
- Prediction: both meaningful layer kinds would exceed PCC 0.995 against the
  periodic one-query HF oracle, be closer to the correct-position oracle than
  the wrong-position negative control, and produce bitwise-identical repeated
  replay output.

## Focused Test

Added the thin env-gated
`test_fused_exact_context_distinct_traced_decode` wrapper. It monkeypatches the
inherited test's decoder class to `FusedDecoder`, spies on construction, and
asserts exactly one object of exact type `FusedDecoder` whose shared MLP is
exactly `_FusedSharedMLP`. The inherited test then prefills 262143 periodic but
nonconstant history tokens, captures decode with a distinct sentinel, copies a
different final token into the stable traced allocation, replays twice at
position 262143, and compares with correct- and wrong-position HF oracles.

Preflight was serialized on the single P150: `timeout 60 tt-smi -ls --local`
reported one Blackhole p150b and a 1x1 `ttnn.open_mesh_device` smoke opened and
closed successfully. No pytest, Tracy capture, profiler, serving, or EngineCore
process owned the device. Exact hash-bound command:

```bash
LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-} \
MPLCONFIGDIR=/tmp/matplotlib GEMMA4_LONG_DECODE=262144 \
pytest -sv \
models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py::test_fused_exact_context_distinct_traced_decode
```

## Result And Verdict

| Layer | PCC | Correct RMSE | Wrong-position RMSE | Pytest call time |
|---|---:|---:|---:|---:|
| sliding layer 0 | 0.9993801877423455 | 0.0214797705411911 | 0.08728896081447601 | 87.19 s |
| full layer 5 | 0.9989366009706286 | 0.020169483497738838 | 0.022472156211733818 | 65.59 s |

Result: **2 passed in 155.96 s**. Both tests exceeded PCC 0.995, both
correct-position RMSE values beat their wrong-position controls, and both
second trace replays were bitwise equal to their first replay. The log records
logical/physical final-block addressing (`15/14` sliding, `4095/4094` full),
the hash header, and clean TT device close/destruction. No implementation bug
was exposed and no `fused_decoder.py` change was made.

Verdict: **verified**. The P1 was a missing fused-only wrapper, now closed by a
durable regression. Evidence:
`exact_context_distinct_262144_final.log`.

Hash-bound tested state:

- `tt/fused_decoder.py`:
  `941dd1d16b64246111e8402d875cbb7fc1cc6bbf6b33465ff0ba97103df90af6`
- `tests/test_fused_decoder.py`:
  `29b99b8c51d8bc3f5052da35f37477267e66215627ece3e100e012071b3eee06`

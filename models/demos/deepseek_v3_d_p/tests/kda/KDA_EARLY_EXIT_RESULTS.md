# KDA runtime early-exit prototype: validation

Implemented on `kda_pad_early_exit_runtime`, based on published PR #56632 head
`84ae832f1787fc1b8495e9a1b40f8090601a658d`, in
`/localdev/mvasilijevic/tt-metal.worktrees/kda_pad_runtime`.
Validation date: 2026-09-17. See the approved [contract](KDA_EARLY_EXIT_DESIGN.md)
and [implementation plan](KDA_EARLY_EXIT_DEV.md).

## Result

Runtime bounds work within one captured trace. Preparation, summary, group
reduction, exclusive scan, and final scan skip invalid aligned chunks/groups.
Physical addressing and collective participation remain fixed; empty ranks
publish identity transitions. Final convolution and recurrent carries stop at
the valid end. No input-state mutation or padded-value dependence was observed.

The prototype does **not** establish trimmed-run performance parity. In the
measured production-shaped SP1 recurrence, 4096 valid tokens in capacity 5120
were 10.59% slower than trimming; 4896 were 1.35% slower. Both saved time compared
with processing all physical tokens. Whole-layer savings were smaller.

## Timing

Eight Blackhole devices, SP1×TP8, synthetic Kimi-K3 weights/inputs, 96 global
heads, K=V=128, physical capacity 5120. Production program configuration,
including grouped local recurrence, was retained. Every graph was initialized
before the first capture. Eight untimed rotating rounds preceded eight measured
rotating rounds; each sample timed ten replays plus synchronization, divided by
ten. These are **wall latency** measurements, not device-profiler durations.

The ordinary physical, runtime-bound, and trimmed graphs were captured in the
same process. The runtime graph changes end from full to partial without
recapture. Controls use this branch's omitted-end path; these numbers do not
independently benchmark an untouched checkout of the PR.

Milliseconds are per-scenario medians. Percentages are medians of paired
same-round ratios; negative means early exit is faster. Full-bound overhead is
visible in the separate early-full column. Raw samples: [JSON](KDA_EARLY_EXIT_TIMINGS.json).

| Stage | Valid tokens | Physical ms | Early full ms | Early tail ms | Trimmed ms | Tail vs physical | Tail vs trimmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recurrence | 4096 | 1.7084 | 1.6899 | 1.5643 | 1.4139 | -8.47% | +10.59% |
| recurrence | 4896 | 1.7069 | 1.6876 | 1.6656 | 1.6433 | -2.43% | +1.35% |
| layer | 4096 | 11.7416 | 11.6919 | 11.5586 | 10.1273 | -1.46% | +14.19% |
| layer | 4896 | 11.9169 | 11.8832 | 11.8561 | 11.6412 | -0.41% | +1.96% |

Observed implementation distinction: physical grouping remains eight groups of
20 chunks; trimming to 4096 selects eight groups of 16, and trimming to 4896
selects nine groups of 17. Early exit removes invalid work without redistributing
valid chunks. This can leave a different critical path; its contribution to the
measured gap has not been isolated. Projection, convolution, and surrounding
ordinary tensor operations still process physical shapes.

## Correctness and compatibility evidence

- **3 layer tests passed, 41.95 s**: SP1×TP8, SP2×TP4, SP4×TP2; changing start/end
  in one capture, one valid chunk, partial groups, empty ranks, boundary tails,
  high absolute starts, full→short→full transitions, nonzero carries, repeated
  replay, random/zero padding invariance, and input-carry immutability.
- **10 native/compatibility tests passed, 62.08 s**: native preparation, direct
  scan, and summaries match trimmed execution; cached programs rebind fresh
  scalar addresses; known affine compositions check reduction and exclusive
  scan, including empty ranks with K=32 and K=128. Existing topology replay on
  SP1/2/4/8 and existing layer offset/continuation replay on SP1/2/4 pass.
- **4 performance test items passed, 55.70 s**: all final timing scenarios
  completed. The exact pytest duration is also retained in the final log.
- Host topology harness passed explicit rank-count examples and exhaustive
  aligned conservation/chunk-count checks for P=1/2/4/8, C=128.
- Native C++ build passed. Black and clang-format checks and `git diff --check`
  passed. Python/native imports resolved inside this worktree.

The new layer test retains the existing CPU output/convolution gates and
recurrent PCC/RMSE gates. Some short SP1 and full-offset no-padding baselines
exceed the inherited recurrent peak-error gate: for example, SP1 length 96 has
relative L-infinity 0.6786 against 0.6; SP2 full offset has 0.8220. Those cases
instead require **bit-identical recurrent and convolution carries against the
corresponding unpadded device baseline**, alongside CPU PCC/RMSE. Other padded
SP2/SP4 cases keep the original 0.6 peak-error limit. Existing regression test
thresholds were not relaxed.

## Reproduction and retained evidence

Run from the checkout above, with its local environment:

```bash
source python_env/bin/activate
export TT_METAL_HOME="$PWD"
export PYTHONPATH="$PWD"
./build_metal.sh --enable-ccache --build-ttnn-tests
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/layer/test_padding_early_exit.py -sv -x
scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_padding_recurrence.py tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_padding_prefix.py models/demos/deepseek_v3_d_p/tests/kda/components/test_device_chronology.py models/demos/deepseek_v3_d_p/tests/kda/layer/test_dynamic_trace.py -sv
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/perf/test_padding_early_exit_perf.py -sv
clang++-20 -std=c++20 -I. generated/kda_early_exit_validation/topology_check.cpp -o /tmp/kda-pad-topology
/tmp/kda-pad-topology
```

Complete local logs, including failures, are archived under
`generated/kda_early_exit_validation/` in this worktree. They are runtime
artifacts, not versioned source. Key files:

| Log | Verdict |
| --- | --- |
| `kda-pad-runtime-build.log` | Initial full native build, exit 0 |
| `kda-pad-runtime-build-control.log` | Reader-only metadata flag fix built, exit 0 |
| `kda-pad-runtime-build-docs.log` | Final binding documentation built, exit 0 |
| `kda-pad-runtime-build-verified.log` | Exit 0, 7.035 s wall time |
| `kda-pad-runtime-layer-7.log` | Safe wrapper PASS, 3 items |
| `kda-pad-runtime-regression.log` | Safe wrapper PASS, 10 items |
| `kda-pad-runtime-perf-final.log` | Safe wrapper PASS, 4 items; final warm measurements |
| `kda-pad-runtime-format.log` | Black/clang-format/diff whitespace checks passed |

## Failures, warnings, and limits

Initial kernel JIT exposed an extra reader-only compile argument on compute and
writer kernels; corrected and rebuilt. An initial summary oracle used a
different FP32/BF16 transport path; matching the PR's dynamic BF16 boundary
restored exact comparison. The CPU peak-error discrepancies above reproduce in
unpadded baselines and are explicitly covered by differential checks.

Three exploratory runs reported HANG: interleaved eager references in the layer
test, and two performance runs replaying the trimmed graph after other graphs
had been captured. Safe pytest collected triage and reset the devices each
time. Computing layer references before capture and warming **all** benchmark
paths before **any** capture eliminated these failures in subsequent runs.
Allocation warnings make resource aliasing a plausible explanation, but no
specific offending buffer was identified; the precise cause remains unresolved.
No manual device reset was used.

Initial layer timings drifted during warm-up. They are retained in
`kda-pad-runtime-perf-2.log` but superseded by the final run's additional untimed
rounds. Do not combine their samples with the final measurements.

Observed warnings: CMake dependency minimum-version deprecations (protobuf, googletest,
md4c), nanobind dependency version mismatch, md4c policy/developer warnings,
Tracy's unused `fread` result on the initial build, deprecated SFPU
`copy_dest_value` during earlier typecast JIT, Python SWIG/Pydantic deprecations,
redundant `to_layout` memory configuration, L1 semaphore fragmentation,
suboptimal fabric packet sizes, and allocations while traces exist. Complete
messages and JIT counts are in the logs. Final performance JIT was 423/423 cache
hits; final correctness runs included real JIT compilation.

Not validated: globally empty intervals (excluded by contract), unaligned bounds,
out-of-range values (caller preconditions), arbitrary per-rank metadata,
performance on SP>1, real checkpoint accuracy, or device-cycle attribution.
No projection/convolution early exit or physical regrouping is implemented.

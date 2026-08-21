# Metalium 2.0 convolution-operation migration learnings

This document records reusable technical findings and validation evidence for
the migration program tracked by Beads epic `tt-metal-rf8`. Beads is the source
of truth for task status, ordering, ownership, dependencies, and blockers; this
file is not a task tracker.

## Non-negotiable execution contract

- Target validation system: the local T3K Wormhole system.
- Build after each operation with `./build_metal.sh --release`. Convert-to-HWC
  and Conv2D also require `./build_metal.sh --release --build-tests` before
  their C++ gtest gates so `unit_tests_ttnn` cannot be stale.
- Run every Python test through `./scripts/run_safe_pytest.sh`; use
  `--run-all` in default mode for the final operation-level correctness suite.
  Do not use `--dev` as a correctness gate because watcher-enabled tests have
  additional skips. Use it only as a diagnostic rerun after a failure/hang,
  record watcher-induced skips, and revalidate correctness in default mode.
  Use `--profile` only for timing: the Tracy wrapper masks pytest's exit code,
  so a separate default-mode `--run-all` pass is always required. Before a
  profile run, verify the build has Tracy enabled and the Python Tracy
  dependencies are importable.
- The safe pytest runner serializes hardware access, enforces dispatch
  timeouts, resets dirty devices, and automatically invokes
  `./tools/tt-triage.py` on a detected hang. For additional manual diagnosis,
  run `./tools/tt-triage.py` (the bare `tt-triage` command is not currently on
  `PATH`). Preserve the pytest, watcher, and triage outputs as evidence.
- Freeze correctness and performance baselines before editing an operation.
  Do not relax accuracy thresholds, remove cases, add skips, or accept a
  performance regression to finish a migration.
- If current infrastructure cannot express a correct, equal-or-faster design,
  keep the operation task open, create and link an infrastructure blocker in
  Beads, and record the evidence here. Do not skip the operation or retain an
  unreviewed legacy shadow path as a nominal migration.
- At every baseline, record the actual board identity, architecture, firmware,
  device count, and compute/storage grid. Inventory every pre-existing
  architecture/grid/watcher skip in the selected test files. If the local T3K
  is harvested such that a required 8x8 Wormhole specialization is skipped,
  equivalent correctness and performance coverage must be added for the
  available topology or the operation remains blocked; a skip is not evidence.
- Eleven operations have no dedicated in-tree performance gate. For each one,
  record reproducible device-profiler and synchronized wall-time baselines for
  a small representative set of important shapes/configurations before
  editing. Profiling every parametrized correctness case is intentionally not
  required: the full operation-level inventory runs normally, while Tracy is
  reserved for focused before/after performance evidence. Conv2D additionally
  uses its existing per-shape Wormhole targets in
  `test_conv2d_device_perf.py` as a hard gate.

Use an isolated cold-JIT cache for every meaningful specialization:

```bash
migration_cache_dir=$(mktemp -d /tmp/metalium2-jit-XXXXXX)
TT_METAL_CACHE="$migration_cache_dir" ./scripts/run_safe_pytest.sh --run-all <test-path-and-selection>
```

After each host rebuild/install, verify both the Python extension and the
actually mapped `_ttnncpp.so` loaded by the test environment before trusting
runtime behavior. Checking only `ttnn._ttnn.__file__` is insufficient because
`LD_LIBRARY_PATH` can override the extension's RUNPATH:

```bash
python -c 'import pathlib, ttnn._ttnn as module; print(module.__file__); print(*sorted({line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines() if "_ttnncpp.so" in line}), sep="\n")'
```

Both printed paths must resolve to the current worktree's installed release
artifacts. If the import environment is not configured or points at stale
`build_Release/ttnn`, `build_Release/lib`, or another checkout, fix/install the
worktree build before testing.

## Incremental Claude Opus review protocol

Treat each operation as one migration stage and review one operation at a
time. After the operation's implementation and validation are complete, the
review prompt must limit itself to that operation's incremental diff, evidence,
and shared code directly changed for that operation. Invoke the reviewer
non-interactively with:

```bash
claude --dangerously-skip-permissions --model opus --effort high -p "<single-operation review prompt>"
```

The prompt must review semantic Metalium 2.0 completeness, correctness, cache
rebinding, cold-JIT specialization coverage, DFB ownership and queue semantics,
test coverage, and before/after performance evidence. The operation's review
gate is complete only when Opus explicitly returns `APPROVED`. Otherwise,
address every finding and run a fresh review focused on that same operation.
Record prompts, findings, fixes, and the final approval under the operation's
section below.

While a review is running, overlap only non-conflicting preparation for later
operations: boundary inventories, baseline runs, test-matrix design, and
performance capture. Do not start later shared-kernel edits or declare a later
operation implementation-ready before the current operation is approved.

## Existing operation-level test inventory

Every listed Python path is run through `./scripts/run_safe_pytest.sh
--run-all`. Existing architecture/topology skips must be inventoried before the
migration; no new migration-owned skip or xfail is acceptable. Tests that are
not applicable to the T3K Wormhole must be identified by their existing reason,
not silently omitted.

| Order | Operation | Existing directly applicable operation-level tests |
|---:|---|---|
| 1 | Fold | `tests/ttnn/unit_tests/operations/conv/data_movement/test_fold_op.py`; `tests/ttnn/nightly/unit_tests/operations/conv/data_movement/test_fold_op.py`; `tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_fold.py`; consumer regression `tests/ttnn/unit_tests/operations/conv/test_prepare_conv_weights.py::test_prepare_conv_weights_with_fold` |
| 2 | Convert-to-CHW | `tests/ttnn/unit_tests/operations/conv/data_movement/test_convert_to_chw.py` |
| 3 | Rotate | `tests/ttnn/unit_tests/operations/pool/test_rotate.py`; `tests/ttnn/nightly/unit_tests/operations/pool/test_rotate.py` |
| 4 | Grid Sample | `tests/ttnn/unit_tests/operations/pool/test_grid_sample.py`; `tests/ttnn/nightly/unit_tests/operations/pool/test_grid_sample.py` |
| 5 | Convert-to-HWC | `tests/ttnn/unit_tests/operations/data_movement/test_convert_to_hwc.py`; CPU gather gtests in `tests/ttnn/unit_tests/gtests/test_convert_to_hwc_gather.cpp` |
| 6 | Padded Slice | Padded-slice cases in `tests/ttnn/nightly/unit_tests/operations/data_movement/test_slice_for_conv.py` |
| 7 | Slice Write | `tests/ttnn/unit_tests/operations/data_movement/test_slice_write.py`; `tests/ttnn/unit_tests/operations/data_movement/test_slice_write_overlap.py`; the `test_slice_write_*` cases in `tests/ttnn/unit_tests/operations/data_movement/test_slice.py`; slice-write cases in `tests/ttnn/nightly/unit_tests/operations/data_movement/test_slice_for_conv.py` |
| 8 | Upsample | `tests/ttnn/unit_tests/operations/pool/test_upsample.py`; `tests/ttnn/nightly/unit_tests/operations/pool/test_upsample.py` |
| 9 | Halo | Halo is an internal component of composite operations rather than an independently exposed public op. Validate it through `tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::test_halo_reshard_conv`, the full applicable Conv2D suite, and Pool2D coverage including `test_run_max_pool_height_shard`, `test_run_max_pool_width_shard`, and `test_run_max_pool_block_shard` in nightly Pool2D. No standalone Halo suite is required. |
| 10 | Pool2D | `tests/ttnn/unit_tests/operations/pool/test_maxpool2d.py`; `test_avgpool2d.py`; `test_mpwi.py`; `test_adaptive_pool2d.py`; `test_global_avg_pool2d.py`; and the corresponding nightly files `test_maxpool2d.py`, `test_max_pool2d_sweeps.py`, `test_max_pool2d_sweeps_mpwi.py`, `test_avgpool2d.py`, `test_avg_pool2d_sweeps.py`, and `test_adaptive_pool2d.py` |
| 11 | Conv3D | `tests/ttnn/unit_tests/operations/conv/test_conv3d.py`; `tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py` |
| 12 | Conv2D | `tests/ttnn/unit_tests/operations/conv/test_conv2d.py`; nightly `test_conv2d.py`, `test_conv2d_sweeps.py`, and `test_conv2d_ulp.py`; regular and nightly `test_conv1d.py`; nightly `test_conv1d_sweeps.py`; regular and nightly `test_conv_transpose2d.py`; nightly `test_conv_transpose2d_sweeps.py` and `test_conv_transpose2d_ulp.py`; `tests/ttnn/unit_tests/operations/conv/test_prepare_conv_weights.py`; C++ `Conv2DFixture` gtests |

The Convert-to-HWC CPU gather tests run from the release test binary as
`./build_Release/test/ttnn/unit_tests_ttnn
--gtest_filter='GatherTransferTest.*'`. The Conv2D C++ tests use
`--gtest_filter='*Conv2DFixture*'`. These are supplemental to, not substitutes
for, the safe pytest suites.

Conv2D's existing hard performance gate is run without nesting the safe
runner's `--profile` wrapper:

```bash
./scripts/run_safe_pytest.sh --run-all tests/ttnn/perf_tests/operations/conv/test_conv2d_device_perf.py::test_conv2d_device_perf
```

That test launches the device profiler itself and checks the per-shape
Wormhole targets, including activation reuse. Its result and its profiler
artifacts are both required.

Operation-focused cache, cold-JIT, trace, determinism, and performance tests
required by the migration definition of done must be added when the existing
suite does not prove those contracts. At final validation, rerun the complete
applicable operation-level inventory, including all newly added cases.

## Per-operation evidence format

Add a dated section for each operation containing:

1. Baseline commit, hardware/firmware identity, exact build and test commands,
   existing skips, correctness metrics, and both device-program and synchronized
   wall-time samples.
2. Host/kernel boundary inventory, specialization matrix, semantic argument
   table, and DFB/semaphore topology.
3. Implementation decisions, including shared-kernel consumer impact and any
   rejected alternatives.
4. Release-build result; focused, full, cache, trace, determinism, cold-JIT,
   and profiler results; generated-artifact specialization proof; normalized
   warning inventory.
5. Hang evidence when applicable: safe-runner output, watcher log, triage
   output, root cause, and verified correction.
6. Incremental Opus review iterations and final explicit approval.

## Cross-operation learnings

- Hardware inventory on 2026-08-19 found four Tenstorrent PCIe endpoints, all
  device ID `0x401e`, subsystem device `0x0014` (n300/T3K Wormhole). Each
  operation baseline still records runtime-visible logical device count,
  firmware, and compute/storage grid because harvesting can reduce applicable
  coverage without changing these PCI identities.
- The repository entry point for hang diagnosis is `./tools/tt-triage.py`; the
  safe pytest wrapper already integrates it on dispatch timeout.
- Rotate bilinear shares kernels with Grid Sample and Pool2D, so its focused
  migration must compile and regress those consumers without broadening the
  Opus review beyond the shared changes caused by Rotate.
- Padded Slice and Slice Write share the nightly
  `test_slice_for_conv.py` validation surface; both tasks rerun the whole file
  at final validation to expose interaction regressions.
- Halo is intentionally validated through its Conv2D and Pool2D composite
  consumers. Its migration evidence must show the relevant Halo topology,
  cache, correctness, and performance specializations were exercised within
  those composite suites.
- Conv2D activation-reuse and split-reader FIFO cursor mutation remains a
  conditional Wormhole/Quasar infrastructure risk. Equal-or-better measured
  operation-local rewrites are preferred; otherwise a linked infrastructure
  blocker is required and Conv2D remains open.

## Fold (`tt-metal-rf8.1`)

### Baseline and boundary inventory (2026-08-19)

The corrected branch base is `origin/main` at
`03312d2a2bcc0f7685a33f4fe89f7d8a904ac90f`. The unrelated commit
`32832e4084a7d3ec6b3508ebe65fcf3cd44707ba` is not an ancestor of the branch.
The migration assessment and Beads setup are the only two commits above main.
The rebase changed none of Fold, the kernel-signature compiler, or the safe
pytest runner, so the frozen baseline remains source-identical to the corrected
base.

The local T3K exposes eight Wormhole B0 chips (local IDs 0--3 and remote IDs
4--7), firmware `18.12.1`, KMD `2.9.0`, and an applicable 8x8 worker grid. The
release extension check resolved to this worktree's
`ttnn/ttnn/_ttnn.so` and `build_Release/lib/_ttnncpp.so`.

Baseline command results before source edits:

- `./build_metal.sh --release`: passed.
- Regular Fold matrix: 198 passed, 252 skipped in 219.41 seconds. The skips
  were 72 unsupported bfloat8_b/row-major combinations, 144 unsupported tile
  front-padding combinations, 24 invalid padded-dimension/stride combinations,
  and 12 FP32/L1-capacity combinations. No architecture, harvested-grid,
  watcher, xfail, or migration-owned skip occurred.
- Nightly Fold matrix: 20 passed in 24.63 seconds, no skips.
- Universal input/output Fold matrix: 84 passed in 43.10 seconds, no skips.
- `test_prepare_conv_weights_with_fold`: 5 passed in 8.31 seconds, no skips.

Corrected-base Tracy runs used 21 repetitions and excluded the first/cold
sample. For RM DRAM interleaved (`rm_DRAM_to_DRAM`), the 20 warm medians were
host 5,578 ns, firmware 2,757 ns, and device kernel 1,901.5 ns (kernel p95
1,998 ns). For the 8-core height-sharded `-Os` specialization
(`32x32x16_2x2_8c`), the medians were host 6,000 ns, firmware 1,797.5 ns, and
device kernel 964 ns (kernel p95 986 ns). The profiler CSVs are
`generated/profiler/reports/2026_08_19_22_07_35/ops_perf_results_2026_08_19_22_07_35.csv`
and
`generated/profiler/reports/2026_08_19_22_08_00/ops_perf_results_2026_08_19_22_08_00.csv`.

Fold has three semantic program topologies and no semaphores:

- Height-sharded row-major: two instances of `writer_dfb2s_row_major.cpp`
  split columns using the `is_reader` CTA. Both bind borrowed `src0` and
  `dst0` DFBs; there are 12 CTAs and no runtime args. The host preserves the
  tuned `-Os` optimization level.
- Row-major interleaved: a reader produces `in0` and a writer consumes it. The
  reader binds input tensor `src`; the writer binds output tensor `dst` and,
  for sub-L1-aligned sticks only, a self-loop `in1` scratch DFB under the same
  `FOLD_RM_NOT_L1_ALIGNED` condition. Both have five CTAs. Reader runtime args
  are `work_per_core`, `src_index`, and `curr_src_row_index`; writer runtime
  args are `work_per_core` and `dst_index`.
- Tile interleaved: reader -> `src0` -> untilize compute -> `src1` -> writer,
  with separate main/cliff compute specs. Reader has two CTAs and two runtime
  args; writer has eight CTAs and four runtime args; compute has two CTAs and
  no runtime args. The public composite currently untilizes tile input before
  primitive dispatch because the pre-existing native tile-index math is only
  correct for `(N,32,32,C<=32)`; this migration does not change that routing or
  claim the unrelated native-TILE follow-up is complete.

### Implementation and focused validation

All five owned kernels plus shared `untilize_metal2.cpp` now use a single
`TT_KERNEL` entry with named template parameters for every registered CTA and
named function parameters for every registered runtime argument. The operation
bodies, TensorAccessor/DFB/Noc calls, queue lifecycle, conditional scratch DFB,
work split, compiler defines, and `-Os`/`-O3` choices are unchanged.

On the corrected base, `./build_metal.sh --release` passed. Forced-cold JIT
with isolated `TT_METAL_CACHE` directories passed for RM DRAM interleaved and
height-sharded routes with zero cache hits. Generated `kernel_includes.hpp`
artifacts contain the expected auto-generated `kernel_main` shims for the RM
reader/writer and both `is_reader` instances of the sharded kernel. A focused
same-key cache probe created fresh host/input/output allocations twice and
proved both RM DRAM and height-sharded program-cache hits preserve correctness
without increasing the cache-entry count. It was used only as a disposable
validation probe and removed; the migration adds no permanent Fold test.

Final default-mode correctness results through `scripts/run_safe_pytest.sh
--run-all` were 198 passed/252 unchanged pre-existing skips for the regular
matrix, 20 passed for the nightly matrix, 86 passed for the migration-stage
universal matrix (84 pre-existing cases plus two temporary fresh-allocation
cache routes), and 5 passed for the weight-preparation consumer. The temporary
routes were subsequently removed to keep only migration code; the permanent
universal matrix remains the original 84 cases. No threshold, skip, or xfail
was added or relaxed.

Candidate Tracy runs used the same 21-repetition protocol. RM DRAM warm medians
were host 5,886.5 ns, firmware 2,779.5 ns, and device kernel 1,921 ns (kernel
p95 2,076 ns). The device-kernel distributions do not show a statistically
significant slowdown versus the corrected base (one-sided Mann--Whitney
`p=0.258`): NCRISC median improved from 853.5 ns to 833.5 ns while BRISC moved
from 1,884.5 ns to 1,905 ns. Height-sharded warm medians were host 6,476 ns,
firmware 1,799.5 ns, and device kernel 963.5 ns (kernel p95 988 ns); stripped
BRISC and NCRISC binaries are byte-identical to the baseline. The large host
median movement on the byte-identical sharded program demonstrates that host
timing at this scale is system noise rather than a migration regression. The
candidate profiler CSVs are
`generated/profiler/reports/2026_08_19_22_08_34/ops_perf_results_2026_08_19_22_08_34.csv`
and
`generated/profiler/reports/2026_08_19_22_09_04/ops_perf_results_2026_08_19_22_09_04.csv`.

The shared `untilize_metal2.cpp` compute kernel also has a live tiled-input
Upsample consumer. Its focused interleaved suite passed 94 cases, including
live TILE-layout cases that JIT-compiled the new compute shim; 20 pre-existing
cases skipped because TILE layout does not yet support different logical and
padded shapes. The disposable cache probe also asserted that the first
invocation actually created at least one program-cache entry before checking
that the second fresh-allocation invocation left the count unchanged.

The native tiled Fold reader/writer pair remains unreachable by design: both
public composite branches untilize TILE input before `prim::fold`, and the
primitive has no Python binding. Consequently `reader_dram2dfb_tiled.cpp` and
`writer_dfb2dram_for_tiled_input.cpp` receive no executed parser, JIT, or
device coverage in this stage. Their exact CTA/RTA name sets were statically
checked against the host schemas during review, their bodies were not changed,
and this limitation is accepted only because the pre-existing path is dead;
the migration does not claim to repair or reactivate its known-narrow tile
index math.

### Incremental Opus review

Review iteration 1 returned `CHANGES_REQUIRED`. It found the migrated kernel
bodies, host/signature name sets, DFB lifecycles, cache routes, and performance
evidence correct, then requested: regression of the shared Upsample consumer,
an explicit record of the dead native-tiled pair's zero executed coverage, and
a nonzero first-entry assertion in the fresh-allocation cache probe. The
consumer and cache reruns above passed and the coverage boundary is now stated
explicitly. The probe was removed after review so only migration code remains.
A second Fold-only review follows these fixes.

Review iteration 2 independently reran the Upsample consumer, rechecked both
dead-path host/signature name sets and reachability, reran the cache cases, and
verified formatting/diff hygiene. It returned explicit `APPROVED`.

## Convert-to-CHW (`tt-metal-rf8.2`)

### Baseline and boundary inventory (2026-08-19)

The baseline is the corrected `origin/main` base at
`03312d2a2bcc0f7685a33f4fe89f7d8a904ac90f` on the same local T3K Wormhole
system described for Fold: eight visible Wormhole B0 chips, firmware
`18.12.1`, KMD `2.9.0`, and an applicable 8x8 worker grid. An isolated
cold-JIT run of the complete existing
`test_convert_to_chw.py` inventory passed 111 cases in 29.76 seconds with zero
skips or xfails and 0/23 JIT cache hits. The existing suite already contains
`test_convert_to_chw_with_program_cache`, which repeatedly creates fresh
input/output allocations across BF16 and BF8 configurations and asserts the
expected cache-entry count; no new Convert-to-CHW test was needed.

The legacy program has one cached factory and no semaphores. Its physical CB
order is input `c_0`, output `c_1`, and owned transpose scratch `c_2`:

- The reader produces the borrowed input CB. The tensor address is carried by
  that globally allocated CB; its only compile-time argument is `cb_in`, and
  it receives `total_tiles_per_core` at run time.
- The compute kernel consumes input and produces transpose scratch, has
  `cb_in` and `cb_transpose` compile-time arguments, and receives
  `total_tiles_per_core` at run time. It uses `-O3`, HiFi4 math fidelity, and
  precise SFPU behavior.
- The writer consumes transpose scratch and writes through the borrowed output
  CB. Its compile-time arguments are `cb_transpose`, `cb_out`, and channel
  count; its runtime argument is `total_tiles_per_core`. The writer's existing
  stateful NOC helper is retained.

The representative Tracy specialization is the applicable 8x8,
`HW=131072`, `C=32`, BF16, non-padded case. It used 21 repetitions with the
cold first sample excluded. The 20 warm baseline medians were host 6,970 ns,
firmware 38,730.5 ns, and device kernel 38,071.5 ns, with device p95 38,094
ns. The baseline CSV is
`generated/profiler/reports/2026_08_19_22_18_56/ops_perf_results_2026_08_19_22_18_56.csv`.
The other 110 correctness cases were deliberately run normally rather than
under Tracy.

### Implementation and focused validation

The cached legacy factory is replaced by one `ProgramArtifacts`/`ProgramSpec`
and per-invocation `ProgramRunArgs`. The input and output DFBs are tensor-backed
borrowed buffers and the transpose DFB is owned. The DFB vector deliberately
preserves the legacy physical order `input`, `output`, `transpose`; this keeps
the performance-critical compute and writer binaries stable while bindings
remain semantic. The writer registers output as both producer and consumer,
preserving its legacy self-managed borrowed-output queue behavior. Tensor
addresses now rebind semantically on every invocation, eliminating the manual
cache-hit override.

The reader, compute, and writer use `TT_KERNEL` entries with named runtime and
compile-time arguments and named DFB bindings. Kernel bodies, stateful NOC
operations, queue ordering, data formats, channel specialization, math
configuration, and optimization levels are otherwise unchanged. The initial
cold-JIT probe exposed a missing include for the writer's pre-existing
`experimental::set_read_state`/`read_with_state` helpers; restoring
`experimental_device_api.hpp` fixed the compile without changing behavior.

Final validation evidence:

- `./build_metal.sh --release` passed.
- The complete existing operation-level suite passed 111/111 in 30.03 seconds
  from an isolated fresh cache, with zero skips/xfails and 0/23 JIT hits.
- The loaded release artifacts resolved to this worktree's
  `ttnn/ttnn/_ttnn.so` and `build_Release/lib/_ttnncpp.so`.
- Generated artifacts contain reader and writer shims that pass the named
  `total_tiles` runtime argument, the writer specializes the named `channels`
  argument, and every generated compute phase calls
  `convert_to_chw(get_arg(args::total_tiles))`. Generated compute bindings map
  input to physical slot 0 and transpose to slot 2, matching the preserved
  legacy layout.
- `clang-format --dry-run --Werror` on all five migrated files and
  `git diff --check` passed.

The final representative Tracy candidate used the same 21-repetition protocol.
Its 20 warm medians were host 7,683.5 ns, firmware 38,750.5 ns, and device
kernel 38,083.5 ns, with device p95 38,106 ns. Component medians were BRISC
37,852 ns, NCRISC 229 ns, TRISC0 20,722.5 ns, and TRISC2 29,822.5 ns, versus
baseline BRISC 37,834 ns, NCRISC 237.5 ns, TRISC0 20,714.5 ns, and TRISC2
29,805 ns. The candidate CSV is
`generated/profiler/reports/2026_08_19_22_35_26/ops_perf_results_2026_08_19_22_35_26.csv`.

The 12 ns device-median movement is run-to-run measurement noise rather than a
code-generation regression: after stripping, candidate BRISC and all three
TRISC binaries are byte-identical to baseline. Only the short reader NCRISC
binary differs due to the generated named-argument shim, and its measured
median improves by 8.5 ns. This provides stronger evidence than the aggregate
timing alone that the migration does not compromise device performance.

### Incremental Opus review

Review iteration 1 independently compared the five Convert-to-CHW files with
`origin/main` and returned explicit `APPROVED`. It verified that no legacy host
construction or cache override remains; DFB author order deterministically
preserves physical slots `c_0/c_1/c_2`; the output producer/consumer self-loop
is the sanctioned form for a borrowed output with no downstream drainer; DFB
sizes and queue operations match legacy; tensor specifications hash all values
that determine baked runtime arguments; the existing cache test exercises
fresh allocation churn; TT_KERNEL schemas and generated shims match; and the
byte-identical critical binaries make the performance conclusion sound. The
review requested no code or test change. Its only documentation observation,
clarifying that the legacy reader tensor address rode the globally allocated
CB rather than a kernel argument, is incorporated above.

## Rotate (`tt-metal-rf8.3`)

### Baseline and boundary inventory (2026-08-19)

The baseline is `origin/main` at
`03312d2a2bcc0f7685a33f4fe89f7d8a904ac90f` on the same local T3K: eight
visible Wormhole B0 chips, firmware `18.12.1`, KMD `2.9.0`, and an applicable
8x8 worker grid. The complete existing operation-level inventory is exactly
the regular and nightly `test_rotate.py` files. After a board-initialization
stall on the first attempt, the safe runner invoked triage; the native stack
located the stall in UMD remote-RISC reset/TLB setup inside Metal context
initialization, before any Rotate program ran. The process was terminated, the
safe runner reset the dirty T3K successfully, an eight-case custom-fill probe
passed, and an isolated-cache retry passed all 376 existing cases in 85.65
seconds with zero skips or xfails and zero JIT hits. This was infrastructure
recovery, not an op hang.

Rotate has two mode-specific factories and no semaphores:

- Nearest uses a reader and writer. An owned fill DFB is read locally by the
  reader; the output DFB is owned for interleaved output or borrows the sharded
  output tensor. The input and output tensors use semantic tensor bindings.
- Bilinear owns fill, input-tile, and scalar DFBs. Its output DFB is owned for
  interleaved output or borrows the sharded output tensor. The reader feeds a
  shared Pool2D compute implementation; interleaved output drains through the
  writer shared with Grid Sample. Up to two disjoint work units preserve the
  legacy work split and compute specialization.

Only two representative performance cases were profiled: the same
`1x64x64x128`, angle 45, BF16 row-major configuration in nearest and bilinear
mode. The other 374 correctness cases were deliberately run normally rather
than with Tracy. Legacy isolated-call warm medians/p95s were 14,235/14,538 ns
for nearest and 96,501/96,880 ns for bilinear. Their CSVs are
`generated/profiler/reports/2026_08_19_22_50_12/ops_perf_results_2026_08_19_22_50_12.csv`
and
`generated/profiler/reports/2026_08_19_22_50_59/ops_perf_results_2026_08_19_22_50_59.csv`.

### Implementation and focused validation

Both cached factories now return `ProgramArtifacts` with a semantic
`ProgramSpec` and per-invocation `ProgramRunArgs`; all three Rotate-owned
dataflow kernels use `TT_KERNEL` with named compile-time/runtime arguments,
DFBs, and tensor accessors. The work split, fixed-point coordinate math,
center/fill conversion, burst sizing, sharding behavior, DFB capacity and
ownership, queue ordering, and compiler/hardware configuration are preserved.

The bilinear compute and writer bodies were not duplicated. The original
`compute_pool_2d.cpp` remains a thin legacy entry shell for its not-yet-
migrated Pool2D consumer. Shared force-inlined implementations live in
`pool_2d_compute_impl.hpp` and `pool_stick_writer_impl.hpp`, and Rotate uses
small Metalium 2.0 `TT_KERNEL` shells. Grid Sample subsequently migrated to
the shared Metalium 2.0 writer shell and removed its legacy entry shell. This
keeps each migration incremental without maintaining two algorithms.
Metalium 2.0 assigns the four bilinear DFBs densely; no fake DFBs were
introduced to mimic legacy numeric CB gaps.

The first final-review iteration caught an install-only boundary that local
JIT could not expose: Pool's kernel `FILE_SET` glob installed generic `.cpp`
and Grid Sample `.hpp` files, but not the new generic compute `.hpp` or Grid
Sample Metalium 2.0 writer `.cpp`. The Pool CMake glob now includes both
`generic/device/kernels/*.hpp` and `grid_sample/device/kernels/*.cpp`.
`./build_metal.sh --release` passed after the change, installed both new files
at their source-relative paths, and also repaired the pre-existing omission of
Grid Sample dataflow `.cpp` files from packages.

Validation evidence:

- `./build_metal.sh --release` passed.
- A fresh-cache focused specialization matrix passed 10/10 and forced all
  nearest/bilinear kernel variants through JIT. The cold build first exposed
  that phase-one template parameters must be `uint32_t` and that `ALWI` was
  unavailable in the extracted writer helper; representing boolean
  specializations as `uint32_t` and spelling the helper `always_inline` fixed
  those build-only issues without changing the algorithms.
- A second isolated-cache run of both complete Rotate files passed 376/376 in
  86.20 seconds with zero skips/xfails and 0/103 JIT hits. No Rotate test,
  threshold, skip, or xfail was added or changed.
- Clean focused shared-consumer regressions passed for bilinear Grid Sample
  identity, interleaved AvgPool2D 3x3, and height-sharded MaxPool2D. An earlier
  accidentally broad selector was manually interrupted and is not counted as
  validation evidence.
- `clang-format` on the changed C++ files and `git diff --check` passed.

The initial candidate profile incorrectly queued 21 calls back-to-back,
whereas the frozen baseline's calls were isolated by roughly 0.3 seconds of
pytest setup/teardown. That mismatch accumulated prior-launch completion skew
across cores and made bilinear appear about 1% slower even though its per-core
BRISC, NCRISC, and TRISC kernel bodies were already slightly faster. With an
explicit device synchronization between candidate samples, restoring the
baseline's isolated-op condition, bilinear improves to 96,276 ns median and
96,580 ns p95; cross-core kernel-start spread improves from 647.5 ns to 491.5
ns. The candidate CSV is
`generated/profiler/reports/2026_08_19_23_22_55/ops_perf_results_2026_08_19_23_22_55.csv`.

Nearest isolated candidate repetitions measured 14,278.5/14,491.1 ns and
14,260.5/14,578.8 ns median/p95, versus 14,235/14,538 ns baseline. The roughly
0.2--0.3% median movement is smaller than the observed run-to-run distribution
change; the two candidate p95s straddle the baseline by -47 ns and +41 ns.
BRISC and NCRISC medians move by only tens of nanoseconds. The candidate CSVs are
`generated/profiler/reports/2026_08_19_23_23_49/ops_perf_results_2026_08_19_23_23_49.csv`
and
`generated/profiler/reports/2026_08_19_23_24_17/ops_perf_results_2026_08_19_23_24_17.csv`.
The temporary profiling harness was removed and is not part of the migration.

### Incremental Opus review

Before implementation, a Rotate-only Opus design review recommended extracting
one shared force-inlined compute body and one shared writer body, retaining
thin legacy shells for not-yet-migrated consumers, and adding Metalium 2.0
shells for Rotate. It explicitly rejected broadening this stage to migrate all
shared consumers and rejected a hybrid positional-argument path. A later
Rotate-only diagnostic review was run concurrently with the performance
investigation and terminated after its performance premise was disproved.

The first final Rotate-only Opus iteration returned `CHANGES_REQUIRED` only
for the missing installed-kernel glob coverage described above. After the
two-line CMake fix and a fresh exact release build/install, the incremental
re-review verified all 28 Pool kernel files are covered, compared the 13-file
Rotate transitive installed closure byte-for-byte with the source tree, found
no new correctness, performance, scope, or packaging issue, and returned the
required explicit `APPROVED`.

## Grid Sample (`tt-metal-rf8.4`)

### Baseline and boundary inventory (2026-08-19)

The baseline is the same `origin/main` commit and local T3K Wormhole described
above. The complete existing op-level inventory is the regular and nightly
`test_grid_sample.py` files. The pre-edit safe-runner baseline collected 468
cases and passed 349, with 119 pre-existing skips and 21 warnings in 282.05
seconds. The skips are 26 nearest sharded-output cases that exceed capacity,
92 precomputed-grid cases that require BF16, and one explicit OOM case. No
skip, xfail, threshold, or permanent test was added or changed.

Grid Sample had two cached `ProgramDescriptor` factories and no semaphores.
Nearest is dataflow-only and can use a second DM kernel for its split-reader
path. Bilinear uses one or two shared Pool2D compute specializations. The
specialization contract includes mode, sharding, standard versus precomputed
grid, grid and IO dtype, padding, `align_corners`, grid batching,
`batch_output_channels`, compute configuration, channel chunking, partial last
chunks, core-group work split, and the Wormhole wide-reduction split-reader
decision.

Only two representative configurations were profiled, not the full test
inventory. Both use NHWC `1x64x64x128` input, `56x56` output grid, and 21
calls/20 warm samples. Baseline bilinear device-kernel median/p95 was
167,990.5/168,275.4 ns with 1,409.5 ns median first-to-last kernel start spread;
the CSV is
`generated/profiler/reports/2026_08_19_23_41_32/ops_perf_results_2026_08_19_23_41_32.csv`.
Baseline nearest with a valid precomputed grid measured 12,795/12,994 ns and
210 ns start spread; its CSV is
`generated/profiler/reports/2026_08_19_23_42_32/ops_perf_results_2026_08_19_23_42_32.csv`.
An earlier nearest profile with an invalid non-precomputed input is excluded.

### Implementation and validation

Both factories now return `ProgramArtifacts` containing semantic
`ProgramSpec` and per-call `ProgramRunArgs`. Tensor addresses, DFBs, kernels,
work units, and named runtime/compile-time arguments are expressed through the
Metalium 2.0 artifact model. Grid Sample-owned readers and nearest writers use
`TT_KERNEL`; the bilinear factories use the shared Metalium 2.0 Pool2D compute
and stick-writer shells extracted during Rotate. The old Grid Sample writer
entry shell is deleted, leaving no legacy factory or kernel-boundary API in
the operation. Work splitting, queue capacities and ownership, math and
compile configuration, tensor accessor specialization, core selection, and
algorithm bodies are preserved.

Validation evidence:

- `./build_metal.sh --release` passed after the final legacy-shell removal.
  The install manifest contains the new split bilinear compute and Metalium
  2.0 stick writer and excludes the deleted legacy writer. Changed C++ files
  pass `clang-format --dry-run --Werror`; `git diff --check` passes; and a
  boundary scan finds no `ProgramDescriptor`, `CreateKernel`,
  `CreateCircularBuffer`, positional runtime-argument override, or
  `kernel_main` path in Grid Sample.
- Fresh-cache focused JIT coverage passed interleaved bilinear, interleaved
  nearest, sharded nearest, and sharded split bilinear. A separate targeted
  Wormhole probe forced the sharded precomputed-bilinear non-split path with a
  DRAM input; two distinct seeds and fresh allocations matched the golden
  result while the program-cache entry count remained stable. The temporary
  probe was removed.
- A clean nightly-only rerun passed 241 with 119 expected skips in 177.32
  seconds. The required combined regular+nightly rerun then passed 349 with
  the same 119 expected skips and 21 warnings in 79.38 seconds, with 813/813
  JIT cache hits.
- A temporary bilinear trace probe compiled, captured once, and replayed three
  times. All replays were bit-identical and matched the golden result; it
  passed in 1.17 seconds and was removed.
- Because Grid Sample uses the shared Pool2D compute/writer implementation, a
  complete regression of both Rotate files plus the existing AvgPool2D and
  MaxPool2D files passed 756 cases with 157 pre-existing skips in 527.04
  seconds.

One earlier combined-suite attempt hit a host fetch-queue timeout after many
successful cases. The safe runner invoked `tt-triage`; its analyzer consumed a
CPU for more than five minutes without emitting a report, so only the analyzer
was terminated, the runtime timeout trace was retained, and the runner reset
all four PCI devices successfully. The first apparent selector passed alone,
and both subsequent full-suite runs completed cleanly. This is recorded as a
transient infrastructure/triage-tool incident, not accepted as correctness
evidence and not hidden by a skip.

Candidate Tracy used the identical representative shapes and protocol.
Bilinear measured 168,376 ns median and 168,673.25 ns p95 with 1,410.5 ns start
spread, a +0.229477% median movement. Nearest measured 12,695 ns median and
12,927.75 ns p95 with 209.5 ns start spread, a -0.781555% median movement. The
CSVs are
`generated/profiler/reports/2026_08_20_00_16_42/ops_perf_results_2026_08_20_00_16_42.csv`
and
`generated/profiler/reports/2026_08_20_00_16_57/ops_perf_results_2026_08_20_00_16_57.csv`.
The sub-quarter-percent bilinear movement is measurement noise, while nearest
improves; start spread is unchanged. Candidate artifacts identify
`pool_2d_bilinear_metal2.cpp`,
`writer_pool_stick_interleaved_metal2.cpp`, and
`writer_grid_sample_nearest_interleaved.cpp`, confirming that the measured
paths are the migrated kernels. The temporary profiling harness was removed.

### Incremental Opus review

The first final Grid-Sample-only Opus high-effort review compared the scoped
changes with `origin/main` and returned explicit `APPROVED`. It independently
verified exact named CTA/RTA schemas, semantic tensor-only cache rebinding and
trace safety, DFB sizes/formats/tile geometry/ownership, compute and DM
hardware configuration, every specialization and work split, one-producer/
one-consumer role consistency, the absence of a legacy shadow path, complete
install-manifest coverage, and the four cited performance CSVs. It recomputed
the documented +0.229477% bilinear and -0.781555% nearest deltas exactly and
requested no migration change.

Two observations were explicitly non-blocking for the T3K Wormhole scope.
First, an untested FLOAT32-input plus FP32-destination combination that was
already numerically invalid on main now fails a Metalium 2.0 invariant instead
of silently producing garbage. Second, Quasar Gen2 rejects the Gen1-legal DM
self-loop used by interleaved Grid Sample, but this operation claims no Quasar
support and the target architecture here is Wormhole. Neither is a working
mainline path regressed by this migration.

## Convert-to-HWC (`tt-metal-rf8.5`)

### Baseline and boundary inventory (2026-08-20)

The complete existing operation-level inventory is
`tests/ttnn/unit_tests/operations/data_movement/test_convert_to_hwc.py`.
Before editing, all 261 cases passed in 69.21 seconds with zero skips/xfails.
The gather-table helper suite also passed all 21 `GatherTransferTest.*`
gtests after a test-enabled release build.

The legacy implementation had one cached factory, two instances of the same
data-movement source (reader RISC and writer RISC), one compute kernel, no
semaphores, and six numeric CBs c0--c5. L1 input and output CBs borrowed tensor
storage. The reader-side kernel gathered source fragments into a staging CB;
the two DM kernels consumed alternating transpose tiles and wrote disjoint
output halves. Per-output-core gather tables were positional runtime arguments
copied to both DM kernels even though the second writer never read them.

Only two representative configurations were selected for Tracy rather than
profiling the full 261-case suite. Both use 21 synchronized calls and compare
20 warm samples. The L1 case is B=1, C=4, HW=8192 on an 8x8 worker grid; its
baseline device median/p95 was 3,323/3,355.65 ns. The DRAM case is the
UNet-shallow B=1, C=4, HW=168960 shape on 12 DRAM and 63 worker cores; its
baseline was 42,072.5/42,365.9 ns. Baseline CSVs are
`generated/profiler/reports/2026_08_20_00_37_54/ops_perf_results_2026_08_20_00_37_54.csv`
and
`generated/profiler/reports/2026_08_20_00_38_14/ops_perf_results_2026_08_20_00_38_14.csv`.

### Implementation and validation

The factory now returns `ProgramArtifacts` with semantic tensor parameters,
six named DFBs in the original c0--c5 declaration order, three kernel specs,
one work unit, and per-call tensor/vararg bindings. Scalar compile-time
parameters and all tensor addresses are named. The genuinely variable-length
per-core gather table remains a Metalium 2.0 runtime vararg, uniformly padded
to the longest table; the kernel's serialized group counts ensure padding is
never read. Only the reader-side DM kernel receives that table.

The original two-RISC split is preserved with a dedicated secondary-writer
`TT_KERNEL`. Both writers call a shared force-inlined output loop in
`convert_to_hwc_writer_impl.hpp`, so there is one algorithm body. The compute
kernel is a named-argument `TT_KERNEL`. Producer/consumer roles are assigned
truthfully without the unsafe multi-binding escape hatch: the two raw output
touchers form its producer/consumer pair, while input and tiled scratch use
same-kernel self-loops. The CMake kernel file set includes and installs the new
shared header and secondary source.

The first cold JIT exposed the phase-one restriction that TT_KERNEL template
parameters must be `uint32_t`, not `bool`; changing only the template type
preserved the compile-time specialization. The first single-core uneven case
then exposed Metalium 2.0's borrowed-DFB validator comparing padded physical
shard capacity with TensorSpec's smaller logical packed size. Input and output
DFBs are raw-address aliases rather than FIFOs, so only their advertised extent
is clamped for this case; the borrowed address, physical allocation, offsets,
and reads/writes remain unchanged. The isolated case and the full matrix then
passed exactly.

Validation evidence:

- The final exact `./build_metal.sh --release` passed, as did the preceding
  test-enabled release build. Installed artifacts contain all four
  Convert-to-HWC kernel source/header files. A boundary scan finds no legacy
  program/kernel/CB construction, cache override, positional legacy accessor,
  or `kernel_main`; `git diff --check` passes.
- The complete existing Python suite passed 261/261 in 75.58 seconds with zero
  skips/xfails. No permanent test, threshold, skip, or xfail was added.
- A fresh `/tmp` `TT_METAL_CACHE` large-L1 run passed with 0/8 JIT cache hits,
  proving a true cold build. Runtime provenance resolves `ttnn._ttnn` to this
  worktree's `ttnn/ttnn/_ttnn.so` and `_ttnncpp.so` to
  `build_Release/lib/_ttnncpp.so`.
- A temporary probe held the first tensors alive, allocated distinct same-key
  input/output buffers, and verified exact output with an unchanged program
  cache entry count. It also proved deterministic normal execution and three
  bit-identical trace replays against the golden result. The initial probe
  mistakenly read the captured output before any replay; after correcting the
  harness to validate after execution, it passed. The probe was removed.
- After `./build_metal.sh --release --build-tests`, all 21
  `GatherTransferTest.*` gtests passed.

The two representative candidate profiles both passed exact golden checks.
L1 repetitions measured 3,355.5/3,377.4 ns and 3,349/3,374.7 ns median/p95,
or +0.98% and +0.78% median movement versus baseline; their host medians
improved from 1,667.5 ns to 1,574 and 1,617.5 ns. DRAM repetitions measured
42,236.5/43,035.75 ns and 42,115/43,275.6 ns, or +0.39% and +0.10% median
movement; host medians improved from 1,756.5 ns to 1,433 and 1,510.5 ns. The
second-run first-to-last start spreads are 314.5 ns (identical to L1 baseline)
and 373 ns (1 ns above DRAM baseline). DRAM p95 variation is concentrated in
the two slowest samples and varies between repetitions; medians and start
spread show no material regression. Candidate CSVs are
`generated/profiler/reports/2026_08_20_00_59_47/ops_perf_results_2026_08_20_00_59_47.csv`,
`generated/profiler/reports/2026_08_20_01_00_54/ops_perf_results_2026_08_20_01_00_54.csv`,
`generated/profiler/reports/2026_08_20_01_00_07/ops_perf_results_2026_08_20_01_00_07.csv`,
and
`generated/profiler/reports/2026_08_20_01_01_11/ops_perf_results_2026_08_20_01_01_11.csv`.
The temporary profiling harness was removed.

### Incremental Opus review

The first Convert-to-HWC-only high-effort Opus review returned
`CHANGES_REQUIRED` for one performance defect: legacy `ComputeConfig` defaults
to O3, whereas an otherwise-equivalent Metalium 2.0 `KernelSpec` defaults to
O2. The review correlated that missed setting with the repeatable 48 ns shift
in the candidate L1 distribution floor. The compute spec now explicitly sets
`KernelBuildOptLevel::O3`, matching the legacy factory and the neighboring
Convert-to-CHW migration. No DM compiler option needed changing because both
legacy and Metalium 2.0 DM defaults are O2.

The exact release build passed after the fix. A first post-fix profile using
the existing cache was rejected as evidence: it reported 16/16 JIT hits and
retained the O2 image's 2,248-byte compute kernel. This exposed an important
validation rule: changing a kernel compiler option requires an isolated fresh
`TT_METAL_CACHE`; the displayed kernel hash remained unchanged even though the
compiler option changed, so the ordinary cache can preserve a stale image.

With fresh cache `/tmp/tt-metal-hwc-o3.96Mjl0`, the compiler command explicitly
used `-O3`, JIT telemetry reported 0/16 hits, and the compute image changed to
2,580 bytes. The same representative L1 workload passed exact golden checks
and measured 3,203.5/3,218.35 ns warm median/p95, versus the 3,323/3,355.65 ns
baseline. A second run from that isolated O3 cache measured
3,208/3,232.4 ns. Thus the required setting removes the regression and improves
the device median by 3.60% and 3.46% in the two repetitions. The CSVs are
`generated/profiler/reports/2026_08_20_01_19_05/ops_perf_results_2026_08_20_01_19_05.csv`
and
`generated/profiler/reports/2026_08_20_01_19_54/ops_perf_results_2026_08_20_01_19_54.csv`.
The temporary profiling harness was removed after each run. Finally, the
complete 261-case operation-level matrix was rerun with the isolated cache so
every non-profiled specialization compiled under the final O3 setting. It
passed 261/261 in 125.97 seconds with zero skips/xfails and 0/246 JIT cache
hits. Final approval is pending the required incremental re-review.

The second review verified the O3 correction and its fresh-cache evidence but
returned `CHANGES_REQUIRED` for documentation preservation only. It identified
four expert-intent comments lost during the mechanical rewrite: the DRAM
bank-address folding proof, the split-writer output stride, the padded-channel
minimum, and the semantic purpose of the input/batch/tiled/output CBs. Those
comments are restored at their migrated equivalents without changing code.
Strict clang-format and diff checks pass, and a fresh exact release build and
install passed after the restoration. Approval remains pending the next
incremental re-review.

The third incremental Convert-to-HWC-only review verified every requested
comment at its accurate migrated equivalent, confirmed the delta was
comment-only, found no remaining correctness, performance, packaging, or
migration-boundary issue, and returned the required explicit `APPROVED`.

## Padded Slice (`tt-metal-rf8.6`)

### Baseline and boundary inventory (2026-08-20)

Padded Slice has no standalone post-commit test file. Its complete existing
operation-level coverage is the Padded Slice half of
`tests/ttnn/nightly/unit_tests/operations/data_movement/test_slice_for_conv.py`;
the whole file is the required gate because the other half exercises the
closely related Slice Write operation. Before editing, the full safe-runner
baseline passed 254 cases and retained four pre-existing skips in 85.89
seconds. The skips are the existing block-sharding guard for
`[2, 1024, 1024, 256]`, where the requested 11-core channel grid is not fully
used. No test, skip, xfail, or correctness threshold was added or changed.

The row-major path has one cached factory, an interleaved-input reader, and one
of two writers. Its c0 DFB is borrowed from the sharded L1 output. The padding
writer optionally uses c1 as a zero-row staging buffer; the reader's
non-aligned TRID path uses c1 when padding is absent or c2 when both modes are
active. The non-padding writer is the shared sharded-output handshake kernel.
The tiled path has c0 tiled input, c1 untilized rows, c2 borrowed output, and
c3 zero padding; it combines an interleaved reader, the shared Halo
`pack_untilize` compute kernel, and an owned padded-output writer. Neither
factory uses semaphores. Specialization axes include row-major/tiled input,
height/block/width sharding, slice dimension 1/2 and nonzero starts, BFLOAT8_B,
BFLOAT16, and FLOAT32, pad granularity 8/16/32, aligned and non-aligned rows,
channel-tail padding, core orientation/mapping, and fresh preallocated output
addresses on program-cache hits.

Only two representative workloads were profiled, not the 258 collected test
cases. Both use BFLOAT16 input `[2, 64, 64, 256]`, slice dimension 1, slice
size 32, a 4x4 block-sharded output, 22 calls with 20 warm samples, and exact
golden checks. Row-major input measured 129,080/130,523.2 ns warm device
median/p95, 139/157.3 ns first-to-last start spread, and 6,806/7,244.25 ns
host median/p95. Its CSV is
`generated/profiler/reports/2026_08_20_01_10_01/ops_perf_results_2026_08_20_01_10_01.csv`.
Tiled input measured 23,901.5/24,398.95 ns device median/p95,
289.5/301.6 ns start spread, and 6,308/7,534.35 ns host median/p95. Its CSV is
`generated/profiler/reports/2026_08_20_01_10_38/ops_perf_results_2026_08_20_01_10_38.csv`.
The temporary profiling harness was removed.

The migration design will keep the two legacy shared sources unchanged for
their not-yet-migrated consumers and add Padded-Slice-local Metalium 2.0 entry
shells that preserve the shared sharded-writer handshake and Halo untilize
algorithm exactly. Owned reader/writer kernels will move to named `TT_KERNEL`
interfaces, private staging buffers will become scratchpads, and only true
producer/consumer queues will remain DFBs. This avoids widening this operation
stage into a standalone Halo migration; Halo correctness remains covered
through Padded Slice and later Pool2D/Conv composite gates as requested.

### Migration and validation (2026-08-20)

The two factories now return `ProgramArtifacts` built from mode-specific
`ProgramSpec` and `ProgramRunArgs` objects. The row-major spec has a borrowed
output DFB, an optional padding-row scratchpad, and a TRID scratchpad only for
the non-aligned reader specialization. The tiled spec has input and untilized
row DFBs, a borrowed output DFB, and padding scratch. Tensor bindings carry
fresh input/output addresses; named fixed schemas plus rank-sized runtime
varargs carry slice geometry. The tiled compute kernel explicitly retains the
legacy O3 optimization level.

The legacy shared sources remain unchanged for their other consumers. Local
`TT_KERNEL` shells preserve the shared sharded handshake writer and Halo
pack-untilize algorithm, while the owned readers and padding writers use named
bindings. A row-major shard-tail clamp limits the final core to the logical
output stick count, preventing a source over-read when physical shard capacity
exceeds the logical slice. The operation CMake kernel file set includes the new
sources and shared implementation header.

Validation evidence:

- The exact `./build_metal.sh --release` passed before review and again after
  the review fixes. A fresh-cache focused cold JIT compiled and passed both
  the row-major block-sharded path and a small non-aligned, padded tiled path
  with 0/10 JIT cache hits.
- The complete existing shared operation-level file passed 254 cases with the
  same four pre-existing block-sharding skips in 72.57 seconds. No permanent
  test, skip, xfail, or threshold was added or changed.
- The first full-suite attempt encountered a dispatch timeout in a legacy
  Slice Write case before Padded Slice execution. Automatic `tt-triage` was
  invoked but itself stalled without producing a report. After a standard
  board reset, the timed-out Slice Write case and the first cascaded Padded
  Slice case passed together; the clean full-suite rerun above then passed.
  This distinguishes transient device state from a reproducible migration
  failure without suppressing or weakening any case.
- A disposable probe kept the first tensors alive, allocated distinct
  same-key input/output buffers, and observed unchanged program-cache entry
  counts for both row-major and tiled paths. Normal execution was bit-exact
  and deterministic, and three trace replays per path matched the golden
  output bit-for-bit. The probe was removed.
- `git diff --check` passes, the Padded Slice tree contains no remaining
  legacy program/kernel/CB construction or `kernel_main`, and no temporary
  validation or profiling file remains.

Only the two baseline workloads were profiled after migration. Row-major warm
device median/p95 improved from 129,080/130,523.2 ns to
91,084.5/91,223.8 ns (29.44% median improvement); first-to-last start was
144/162 ns and host median/p95 improved from 6,806/7,244.25 ns to
2,888.5/3,834.65 ns. Its candidate CSV is
`generated/profiler/reports/2026_08_20_01_58_49/ops_perf_results_2026_08_20_01_58_49.csv`.
Tiled warm device median/p95 improved from 23,901.5/24,398.95 ns to
23,128/23,701.25 ns (3.24% median improvement); first-to-last start was
280/310.15 ns and host median/p95 improved from 6,308/7,534.35 ns to
2,938/3,607.3 ns. Its candidate CSV is
`generated/profiler/reports/2026_08_20_01_59_07/ops_perf_results_2026_08_20_01_59_07.csv`.
The first row-major profiling attempt retained every sharded output and
correctly failed with L1 exhaustion after ten calls; it was rejected as
evidence. The corrected disposable harness validated and released each output
per iteration, passed all 22 exact-golden calls for both layouts, and was
removed. The full test matrix was not run under Tracy.

### Incremental Opus review

The first Padded-Slice-only high-effort Opus review returned
`CHANGES_REQUIRED`. Its blocking finding was that the new row-major
non-aligned reader was public but absent from the existing operation matrix:
every permanent row-major case has a 64-byte-multiple channel shard. The
review confirmed that no JIT cache contained the specialization. In keeping
with the request not to add a permanent test, a disposable cold-cache probe
covered two exact-golden public cases: a 24-element/48-byte block shard and an
aligned-width shard with a nonzero, misaligned last-dimension start.

The first cold attempt proved the review finding by exposing a compile error:
the helper accepted a numeric scratchpad id, but Metalium 2.0 supplies a
`ScratchpadBindingToken`. The helper now accepts that token directly. The
review also found that an aligned output row with a misaligned
last-dimension start would bypass staging and silently select a different
wrong source offset than legacy. Factory selection now routes either an
unaligned row stride or an unaligned source start through the scratch/TRID
reader, making that accepted public case correct instead of merely guarding
or preserving the legacy error.

After the exact release rebuild, a new isolated cache
`/tmp/tt-metal-padded-rm-align-fixed.mpHr3E` compiled and passed both cases
with 0/6 JIT hits. The temporary probe was removed. The review-requested
expert comments were restored at the migrated alignment, scratch sizing,
padding, coordinate-carry, TRID-state, and untilize init/uninit sites; two
unused copied Halo constants were removed. The row-major logical-tail clamp
is now explicitly documented as an intentional safety deviation: physical
shard rows beyond the TensorSpec logical extent remain unspecified rather
than being populated with legacy out-of-slice input rows, and the reader does
not exceed the input TensorAccessor page range.

The complete shared operation-level file was rerun after these code changes
and passed 254 cases with the same four skips in 73.92 seconds. The isolated
cache contains the generated
`padded_slice_reader_rm_interleaved_start_id_non_aligned` specialization, so
the proof is not inferred from host selection alone. Normalized warnings are
limited to the pre-existing CMake dependency/policy messages during the
release build and the recurring subset-MMIO runtime warning; there is no
Padded-Slice-owned compile, link, JIT, or runtime warning.

The second Padded-Slice-only Opus review rechecked every first-review finding,
the final release build and complete shared-suite evidence, the cold-JIT
specialization proof, formatting/diff hygiene, and the absence of temporary
tests. It returned explicit `APPROVED`.

## Slice Write (`tt-metal-rf8.7`)

### Baseline and boundary inventory (2026-08-20)

Read-only preparation ran while the Padded Slice review was pending; no Slice
Write implementation source was edited. Its complete operation-level gate is
the 16 rank/layout cases in
`tests/ttnn/unit_tests/operations/data_movement/test_slice_write.py`, the four
overlap cases in `test_slice_write_overlap.py`, the four `four_dim` and three
`copy` cases in `test_slice.py`, and the shared 258-case convolution slice
file. The first group passed 27/27 in 40.56 seconds. The shared file passed
254 cases with four pre-existing block-sharding skips in 72.57 seconds. There
were no new skips, xfails, thresholds, or tests.

The operation has three legacy factories and no compute kernel or semaphore.
The row-major interleaved path owns an input c0 producer/consumer queue and,
only for last-dimension striding, c1 staging for read-modify-write output rows.
Its local reader and strided writer specialize on input/output alignment,
last-dimension offset, element size, and contiguous versus strided writes. The
row-major sharded-input path borrows input c0, pairs the shared sharded
handshake entry with the local writer, and dynamically rebinds the borrowed
input address. The tiled sharded-input path likewise borrows tiled c0 and uses
its shared sharded reader specialization plus the local tiled writer. All
three writers use output TensorAccessor metadata plus runtime page sizes and
rank-sized positional geometry today. The migration must preserve rank 1-8,
nontrivial starts/ends, per-dimension and last-dimension strides, untouched
output regions, alignment-offset overlap semantics, core orientation and
work splitting, fresh input/output addresses, and borrowed-DFB rebinding.

The frozen Metalium 2.0 boundary design is mode-specific rather than a single
overdeclared spec. RM interleaved has input/output tensor parameters, an owned
input DFB produced by the reader and consumed by the writer, and a conditional
writer-private scratchpad for last-dimension read-modify-write staging. Its
reader has seven fixed runtime values; its writer has nine fixed values plus
four rank-sized arrays (input extents, output gaps, starting coordinates, and
reverse strides). RM sharded has input/output tensor parameters, a borrowed
input DFB rebound from the input tensor, a local readiness-handshake reader,
and a writer with nine fixed values plus three rank-sized arrays. Tiled sharded
uses the same borrowed-input handshake topology and adds the conditional
unpadded-channel-tail value to its writer geometry. The shared legacy unary
reader source must stay intact for unmigrated consumers; Slice Write will own
a local `TT_KERNEL` shell for the same handshake. The output is an in-place
destination/return value, so every mode must bind its current address while
leaving every unwritten byte unchanged.

Only three controlled exact-golden factory representatives were profiled, each
with 22 calls and 20 warm samples. Row-major interleaved wrote a
`[2,32,64,256]` source into rows 16:48 of a `[2,64,64,256]` DRAM output and
measured 16,622/17,007.05 ns device median/p95, 186.5/211.95 ns first-to-last
start, and 1,248.5/1,756.1 ns host median/p95; CSV
`generated/profiler/reports/2026_08_20_02_04_59/ops_perf_results_2026_08_20_02_04_59.csv`.
The row-major block-sharded input representative wrote rows 0:32 and measured
154,786.5/159,489.35 ns device, 1,182.5/1,239.1 ns start, and
1,284.5/1,438.55 ns host; CSV
`generated/profiler/reports/2026_08_20_02_05_15/ops_perf_results_2026_08_20_02_05_15.csv`.
The equivalent tiled block-sharded input measured 37,657.5/38,214.5 ns
device, 1,228.5/1,253.45 ns start, and 1,792.5/2,634.2 ns host; CSV
`generated/profiler/reports/2026_08_20_02_05_31/ops_perf_results_2026_08_20_02_05_31.csv`.
The disposable harness passed normally before profiling and was removed; no
full suite was run under Tracy.

### Migration, correctness, and performance (2026-08-20)

The three factories now return mode-specific `ProgramArtifacts`. Row-major
interleaved owns an input DFB between its reader and writer and uses a private
scratchpad only for last-dimension read-modify-write output-row staging. The
row-major and tiled sharded modes borrow their input DFB from the current input
tensor and use a Slice-Write-local readiness-handshake reader, leaving the
shared legacy reader unchanged for unmigrated consumers. All local kernels use
`TT_KERNEL`, named runtime arguments, tensor bindings, `DataflowBuffer`, `Noc`,
and `TensorAccessor`; rank-dependent geometry remains runtime varargs until the
typed-array replacement is available. The row-major sharded output accessor
adds the per-core channel byte offset to the bound output base, which fixed the
only correctness defect exposed by the first complete shared-suite run.

The exact `./build_metal.sh --release` passed. A disposable driver proved
bit-exact normal execution, same-key fresh input/output address rebinding with
an unchanged program-cache entry count, deterministic repeated results, and
three successful trace replays for all three modes. It also provided the three
representative profiling cases and was then removed; no permanent test, skip,
xfail, or threshold changed. Final existing operation-level results were:

- The complete shared `test_slice_for_conv.py` file passed 254 cases with its
  four pre-existing block-sharding skips in 63.88 seconds.
- `test_slice_write.py` plus `test_slice_write_overlap.py` passed 20/20 in
  15.09 seconds.
- `test_slice.py -k=slice_write` passed 7/7 with 476 unrelated cases
  deselected in 3.68 seconds.
- `git diff --check` passes, the Slice Write tree contains no legacy host
  program/kernel/CB construction or `kernel_main`, and no disposable test or
  profiling source remains.

Only one exact-golden representative per factory was profiled, never the full
test matrix. With 22 calls and the first two Slice Write samples excluded, the
clean `DataflowBuffer` implementation preserves or improves device time:

- Row-major interleaved device median/p95 is 16,357.5/17,131.85 ns versus
  16,622/17,007.05 ns at baseline, with 185.5/200.05 ns first-to-last start.
  The final candidate CSV is
  `generated/profiler/reports/2026_08_20_03_15_42/ops_perf_results_2026_08_20_03_15_42.csv`.
- Row-major block-sharded device median/p95 is 105,679/107,106.3 ns versus
  154,786.5/159,489.35 ns, with 152.5/181.25 ns first-to-last start. Its CSV is
  `generated/profiler/reports/2026_08_20_03_06_22/ops_perf_results_2026_08_20_03_06_22.csv`.
- Tiled block-sharded device median/p95 is 37,188.5/37,828.2 ns versus
  37,657.5/38,214.5 ns, with 149/181.1 ns first-to-last start. Its CSV is
  `generated/profiler/reports/2026_08_20_03_06_41/ops_perf_results_2026_08_20_03_06_41.csv`.

Host duration is not yet at parity: median increased from 1,248.5 to 6,750 ns
for row-major interleaved, 1,284.5 to 2,989 ns for row-major block-sharded, and
1,792.5 to 3,089.5 ns for tiled block-sharded. Moving invariant named values
and geometry arrays into common runtime arguments was measured rather than
assumed; it worsened the row-major interleaved host median to 8,305.5 ns and
was rejected and fully reverted. The incremental Opus review was launched
with the host regression disclosed as a potential performance blocker. Per the
no-compromise rule, the Bead remains open unless that gate is resolved; the
device improvements are not being used to conceal the host regression.

The first Opus pass returned `CHANGES_REQUIRED` for two concrete blockers and
two low-risk hardening items. The follow-up patch restores the legacy empty-work
guard before dividing by the merged read count, value-initializes the fixed-rank
kernel geometry arrays so rank-1 execution cannot observe an indeterminate
second slot, restores the rank-arithmetic TODO/edge-case comments and direct
container `log_trace` diagnostics, and resolves predeclared runtime-argument
tables through non-inserting `Table::get()` references. The latter retains the
name-first host construction optimization without allowing a typo to insert,
reallocate the vector-backed table, and dangle earlier references.

After those review fixes, the exact release build passed again. The complete
existing final gate passed again on T3K Wormhole: the shared nightly suite was
254 passed plus four pre-existing skips in 63.06 seconds; the dedicated rank
suite was 16/16 in 18.59 seconds; the overlap suite was 4/4 in 2.92 seconds;
and the Slice integration selection was 7/7 with 476 deselected in 5.37
seconds. No test source changed. The framework host-dispatch regression is now
tracked explicitly as blocker `tt-metal-rf8.13`, which blocks this operation;
its read-only inventory found fresh vector/Table construction in
`ProgramSpecMeshWorkloadFactoryAdapter::apply_descriptor` followed by another
unordered-map construction in `UpdateTensorArgs`. The op task remains open
until that no-regression gate is resolved even if the repeat op review approves
the migration itself.

The repeat Opus pass found no remaining code defect but asked for durable
disposition of two evidence gaps before approval. A disposable safe-runner
probe then exercised an identical-key second call with both fresh input and
fresh output allocations for every factory. All three second calls were
bit-exact and kept the cache count at one: RM interleaved addresses changed
from input/output `(2968640, 2979648)` to `(3001536, 3012544)`, RM sharded from
`(1466368, 2968640)` to `(1433600, 2990528)`, and tiled sharded from
`(1466368, 2968640)` to `(1433600, 2991168)`; each cache-count sequence was
`[(0, 1), (1, 1)]`. The probe passed 3/3 and was immediately removed. The
Quasar suite registered at
`models/demos/vision/classification/resnet50/quasar/tests/ops/test_slice_write.py`
cannot run on this user-requested local T3K Wormhole target, so it is explicitly
unvalidated here; related follow-up `tt-metal-rf8.14` owns Quasar hardware and
the Gen2-only DFB implicit-sync validation rather than silently claiming it.
The next incremental Opus pass verified both evidence dispositions, found no
remaining blocker to the Slice Write code, and ended with explicit `APPROVED`.
Slice Write nevertheless remains open because `tt-metal-rf8.13` is a linked
hard performance blocker under the no-compromise acceptance rule.

### ProgramSpec adapter host-dispatch resolution (`tt-metal-rf8.13`)

The regression came from rebuilding the combined io-plus-op-owned tensor
enumeration separately for every cached coordinate range. The minimal fix
keeps the common io enumeration in `SmallVector` inline storage and resolves
each cached binding directly against either that enumeration or the parked
op-owned tensor vector. It deliberately leaves `UpdateTensorArgs` unchanged:
the smaller change restored parity, so replacing its validation-aware lookup
machinery would add risk without measured benefit. Its existing eight mock
Wormhole `ProgramRunArgsTestGen1.UpdateTensorArgs*` cases all pass.

Only the same three exact-golden Slice Write representatives were profiled,
with 22 calls and the first two samples excluded; the disposable source was
removed immediately. The post-fix host median/p95 results are 1,260/1,475 ns
for row-major interleaved, 1,296.5/1,459 ns for row-major block-sharded, and
1,365/1,581.5 ns for tiled block-sharded. The corresponding origin/main
baselines are 1,248.5/1,756.1 ns, 1,284.5/1,438.55 ns, and
1,792.5/2,634.2 ns. Thus the first two medians are within one percent of
baseline with comparable or better tails, while tiled is faster. Device
median/p95 remains improved or comparable at 16,199/17,113.7 ns,
106,526/110,269 ns, and 38,089.5/38,491.75 ns. The CSVs are respectively:

- `generated/profiler/reports/2026_08_20_04_01_29/ops_perf_results_2026_08_20_04_01_29.csv`
- `generated/profiler/reports/2026_08_20_04_01_46/ops_perf_results_2026_08_20_04_01_46.csv`
- `generated/profiler/reports/2026_08_20_04_01_57/ops_perf_results_2026_08_20_04_01_57.csv`

After the adapter change, the complete existing T3K Wormhole Slice Write gate
passed again: 254 nightly cases plus four pre-existing skips, 16 dedicated
rank cases, four overlap cases, and seven Slice integration cases with 476
unrelated cases deselected. No operation test source was added or changed.
A final disposable post-adapter probe also passed all three factories with
fresh live input and output addresses and exact output while the same-key cache
count remained `[1, 1]`: RM interleaved used `(1466368, 2968640)` then
`(1433600, 3318336)`, RM block used `(1368064, 2968640)` then
`(1236992, 3318336)`, and tiled block used `(1368064, 2968640)` then
`(1236992, 3318848)`. The probe was removed immediately.
The incremental single-focus high-effort Opus review independently checked
cache-miss/cache-hit index parity, multi-coordinate state, op-owned lifetime,
fresh tensor rebinding, validation, and the performance evidence, and ended
with explicit `APPROVED`. It agreed that leaving `UpdateTensorArgs` unchanged
is the lower-risk outcome once measured parity is restored.

## Upsample (`tt-metal-rf8.8`)

### Read-only baseline and boundary inventory (2026-08-20)

This preparation overlapped the Slice Write Opus review; no Upsample source
was edited and implementation remains blocked on Slice Write approval. The
operation has four factories. Nearest interleaved, nearest sharded, and nearest
float already return `ProgramArtifacts`, but still require completion of their
device boundary and owned kernel migration. Bilinear remains a legacy
`ProgramDescriptor` path with reader, writer, compute, weight-LUT, and DFB
performance constraints. Across the operation there are seven owned kernel
sources plus the LUT header. The sharded path also owns a generated halo lookup
tensor whose lifetime and cache-safe rebinding must remain explicit.

Both complete existing operation-level files passed on the local T3K Wormhole:

- `tests/ttnn/unit_tests/operations/pool/test_upsample.py` collected 419:
  323 passed and 96 pre-existing skips in 149.57 seconds. The skips comprise
  20 TILE logical/padded-shape cases, four zero-shard-dimension cases, 60 plus
  eight illegal-shard cases tracked by issue 17795, and four illegal block-core
  ranges.
- `tests/ttnn/nightly/unit_tests/operations/pool/test_upsample.py` collected
  251: 231 passed and 20 pre-existing block-dtype/TILE-layout skips in 141.38
  seconds.

The final gate must therefore retain every currently passing integer-nearest,
float-nearest, fractional-scale, downsample, interleaved, sharded/orientation,
bilinear fidelity, uneven-shard, run-twice/cache, and production-shape case.
Tracy baselines are still limited to a small representative factory set; the
419- and 251-case matrices will not be run under Tracy.

### Representative performance baseline (2026-08-20)

The local target reports eight Wormhole B0 chips (local PCIe device ids 0-3,
remote chip ids 4-7), firmware bundle 18.12.1, KMD 2.9.0, and the 64-worker
core topology exercised by the sharded cases. Five existing exact-correctness
test bodies were wrapped temporarily for 22 calls each; the first two Upsample
samples are excluded below. Each wrapper passed once in normal mode where
needed, and the source was removed immediately. No full suite was profiled.

- Integer nearest, row-major interleaved: host median/p95
  3,332/3,724.85 ns, device 11,317.5/11,749 ns, first-to-last start
  182.5/205.05 ns; CSV
  `generated/profiler/reports/2026_08_20_04_15_42/ops_perf_results_2026_08_20_04_15_42.csv`.
- Integer nearest, tiled interleaved: host 3,280/4,651.2 ns, device
  9,610/9,756.95 ns, start 260.5/299.1 ns; CSV
  `generated/profiler/reports/2026_08_20_04_15_55/ops_perf_results_2026_08_20_04_15_55.csv`.
- Integer nearest, height-sharded row-major: host 2,460.5/2,666.05 ns,
  device 1,186.5/1,221.1 ns, start 176.5/205.1 ns; CSV
  `generated/profiler/reports/2026_08_20_04_19_45/ops_perf_results_2026_08_20_04_19_45.csv`.
- Fractional nearest 1.5x: host 2,746/3,083.75 ns, device
  2,115.5/2,257.4 ns, start 182/203 ns; CSV
  `generated/profiler/reports/2026_08_20_04_16_27/ops_perf_results_2026_08_20_04_16_27.csv`.
- Bilinear 2x, height-sharded: host 3,904/4,754.4 ns, device
  6,951.5/6,998.45 ns, start 438.5/447.1 ns; CSV
  `generated/profiler/reports/2026_08_20_04_18_40/ops_perf_results_2026_08_20_04_18_40.csv`.

### Implementation and final local gate (2026-08-20)

The five nearest-neighbor device entry points that still used interim named
argument APIs now use true `TT_KERNEL` entry points. Their factory topology,
runtime schemas, and algorithms are unchanged. The bilinear factory now returns
`ProgramArtifacts` and describes its borrowed sharded halo/output buffers and
four local tilize-reduce/scalar buffers with a semantic `ProgramSpec`. Its
reader, writer, and compute kernels use named compile-time/runtime arguments and
semantic DFB tokens. The exact legacy buffer sizes, face metadata, zero-work
runtime values, reduce SUM/H defines, math fidelity, approximation mode, and
FP32 accumulation behavior are preserved. The operation tree has no remaining
`ProgramDescriptor`, `create_descriptor`, positional argument accessor, or
`kernel_main` surface.

The exact `./build_metal.sh --release` build passed. A fresh-cache cold-JIT
bilinear representative passed with zero pre-existing kernel-cache hits, then
all 25 existing bilinear cases passed. A disposable cache/trace probe verified
that both loaded Python/C++ libraries came from this worktree, a same-key call
with fresh live input/output allocations reused the program-cache entry,
produced bit-exact output, and replayed a captured trace bit-exactly. The probe
was removed immediately. No test source was added or changed.

The final combined operation gate used a fresh `TT_METAL_CACHE` and the exact
safe-runner command over both existing Upsample files. It collected 670 cases:
554 passed and 116 pre-existing cases skipped in 341.12 seconds, with zero of
869 kernel compilations served by the pre-existing cache. The skips exactly
match baseline: 20 TILE logical/padded-shape cases, four zero shard dimensions,
60 plus eight illegal shard configurations tracked by issue 17795, four illegal
block core ranges, and 20 nightly block-dtype cases requiring TILE layout. No
hang occurred, so `tt-triage` was not needed.

Only the five frozen representatives were profiled after an independent
correctness pass; the full matrix was not run under Tracy. With 22 calls and
the first two samples excluded, post-migration host/device/start median/p95
times in ns are:

- Row-major interleaved: 1,437/1,746.65, 11,347/11,481.1,
  183/195.
- TILE interleaved: 1,463.5/2,325.65, 9,591.5/9,712.4,
  277.5/289.55.
- Integer height-sharded: 1,651.5/1,952.7, 1,189/1,204.05,
  178/190.3.
- Fractional nearest 1.5x: 1,499.5/1,738.95, 2,151.5/2,242.7,
  182.5/203.35.
- Bilinear height-sharded: 2,577.5/3,059.5, 6,945/7,146.15,
  436/455.15.

These measurements are in
`generated/profiler/reports/2026_08_20_04_44_15/ops_perf_results_2026_08_20_04_44_15.csv`.
All host and device medians are nonregressed. Because the first bilinear device
p95 was about 2.1 percent above its baseline despite a flat median, only that
representative was repeated. Its 20 post-warmup samples measured host
2,456.5/2,719.1 ns, device 6,947.5/7,144 ns, and start 433.5/454.1 ns
median/p95, with device values ranging from 6,891 to 7,144 ns. The repeat CSV
is
`generated/profiler/reports/2026_08_20_04_45_06/ops_perf_results_2026_08_20_04_45_06.csv`.
The repeat confirms stable median parity and a narrow device distribution; no
disposable profiling or probe source remains.

The incremental single-operation high-effort Opus review inspected all four
factory declarations and all seven Upsample kernels. It independently verified
strict named CTA/RTA set matching, exact legacy DFB byte sizes and face
metadata, legal borrowed-input/output endpoint wiring, compute fidelity and
reduce parity, zero-work values, absence of baked buffer addresses, cache-key
coverage, fresh-tensor rebinding, and the full correctness/performance record.
It identified only non-blocking observations about a deliberately explicit
compute-architecture branch, a pre-existing asymmetric bilinear scratch size,
and literal-table insertion style, and ended with explicit `APPROVED`.

## Halo (`tt-metal-rf8.9`)

### Read-only boundary and composite-test inventory (2026-08-20)

Halo is an internal sliding-window component, so its correctness gate is its
existing Conv2D and Pool2D consumers; no standalone Halo test will be added or
run. The production boundary contains one approximately 517-line
`WorkloadDescriptor` factory, two owned kernels (split-reader dataflow gather
and optional untilize compute), and four generated UINT16 configuration
tensors. The semantic migration must cover height-, width-, and block-sharded
input, row/column shard orientation, row-major skip-untilize and tiled
untilize, L1 and DRAM configuration placement, two-reader even/odd block
streams, padding, local/remote shard transfers, per-core runtime values, and
config tensor cache lifetime/rebinding.

The experimental Quasar Halo already supplies a useful ProgramSpec design
reference, including op-owned config tensors and named resources. It is not a
drop-in implementation or validation evidence for production Wormhole: its
kernel and implicit-sync behavior are architecture-specific and differ
materially. Production behavior will be frozen and validated through its own
composite consumers.

The public `remote_read` option is not truthful today. The host accepts it and
generates remote-read-oriented config, while the dataflow kernel rejects it
only with `static_assert(!remote_read)`. Unless remote read is actually
implemented and covered, the migration must reject it explicitly on the host
before device compilation. This converts a late compile-time failure into a
clear supported-contract check without pretending the specialization works.

The semantic resource mapping should distinguish queues from raw local memory.
The borrowed sharded input/output and the two compute-to-reader untilize streams
remain DFBs. The four generated config tensors become named op-owned tensor
parameters: readers use their local shard address directly for L1 placement or
read the indexed DRAM page into private scratch. Pad materialization and DRAM
config landing regions have no producer/consumer protocol, so Metal 2.0
`ScratchpadSpec`s describe them more truthfully than self-loop DFBs and remain
valid on both hardware generations. This also avoids baking the four config
addresses into compile-time arguments. Reader 0 retains the borrowed-input
queue bookkeeping; the split readers retain distinct even/odd untilize streams
and nominal opposite endpoints on the borrowed output they scatter-write.

The complete existing operation-level composite inventory is the unit and
nightly `test_conv2d.py` files, nightly `test_conv2d_ulp.py`, unit and nightly
`test_maxpool2d.py`, unit and nightly `test_avgpool2d.py`, and unit
`test_mpwi.py`. Required specialization anchors include
`test_halo_reshard_conv`, the MaxPool2D height-/width-/block-sharded matrices,
the Conv2D and Pool2D DRAM-config cases, transpose-shard cases, and
row-major/tiled consumers. The nightly MaxPool2D file alone collects 4,646
existing cases on this checkout; its pre-edit T3K safe-runner baseline was
started while the Upsample Opus review ran. Full results and the remaining
collection/pass/skip inventory will be recorded when complete.

The monolithic pre-edit nightly MaxPool2D run exposed a test-infrastructure
limit rather than a reproducible failing parameter. After approximately 13
minutes the device wedged, the safe runner's automatic full `tt-triage` process
remained CPU-bound without producing a log or CSV for 15 minutes, and the
runner ultimately reset all four PCI devices. The first node printed after the
hang was a BF8 height-sharded `[1, 576, 32, 32]` case, but an exact isolated
safe-runner rerun completed in 0.56 seconds as its expected BF8 skip. It was
therefore only the teardown point where accumulated device state surfaced, not
a deterministic Halo failure. The final operation-level gate will preserve
complete item coverage while dividing large matrices into bounded pytest
chunks, with each chunk run through the safe runner.

### Representative performance baseline (2026-08-20)

Five existing composite cases were selected exactly and repeated 22 times
under Tracy; the first two Halo calls in each trace are excluded. All 110
executions passed. No full matrix was profiled, and the temporary collection
filter used to pass exact parameter IDs through Tracy's shell boundary was
deleted after the final focused measurements.

- Height-sharded row-major, L1 configs: host median/p95
  7,912/10,871.15 ns, device 6,432/6,520.15 ns, first-to-last start
  194.5/215.2 ns; CSV
  `generated/profiler/reports/2026_08_20_05_28_06/ops_perf_results_2026_08_20_05_28_06.csv`.
- Width-sharded row-major, L1 configs: host 8,340/10,988.15 ns, device
  3,952/3,993.05 ns, start 201/225.35 ns; CSV
  `generated/profiler/reports/2026_08_20_05_29_12/ops_perf_results_2026_08_20_05_29_12.csv`.
- Block-sharded row-major, L1 configs: host 7,703/10,641.6 ns, device
  5,150/5,182.5 ns, start 201.5/225.1 ns; CSV
  `generated/profiler/reports/2026_08_20_05_30_15/ops_perf_results_2026_08_20_05_30_15.csv`.
- Height-sharded row-major, DRAM configs: host 8,133/10,344.95 ns, device
  1,565.5/1,640.9 ns, start 487/514.45 ns; CSV
  `generated/profiler/reports/2026_08_20_05_30_43/ops_perf_results_2026_08_20_05_30_43.csv`.
- Height-sharded tiled-input Conv2D untilize path: host 9,234/10,735.2 ns,
  device 4,125.5/4,249.25 ns, start 476/502 ns; CSV
  `generated/profiler/reports/2026_08_20_05_31_50/ops_perf_results_2026_08_20_05_31_50.csv`.

### Post-migration correctness evidence (in progress, 2026-08-20)

All completed suites were run in normal correctness mode through
`./scripts/run_safe_pytest.sh --run-all`; Tracy is reserved for the five small
representative comparisons above and is not used for full-matrix coverage.

- Unit MaxPool2D: 104 passed, 13 exact pre-existing DRAM auto-config skips.
- Unit AvgPool2D: 276 passed, 144 expected parameter skips.
- Unit max-pool-with-indices: 71 cases were collected; 67 passed and four
  followed the existing OOM skip path at `test_mpwi.py:220`. Every executed
  case reported `output_match=True`, `indices_valid=True`, and zero actual
  errors, value differences, and window violations.
- Unit Conv2D: 161 passed, 48 expected row-major/BF8 incompatibility skips.
  Executed coverage included tiled and row-major inputs, height-, width-, and
  block-sharded layouts, DRAM config placement, and large-shape cases.
- Conv2D ULP: both collected cases passed.
- The explicit Halo reshard consumer anchor
  `tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::test_halo_reshard_conv`
  passed all 28 applicable cases, with 2,378 unrelated cases deselected.
- Nightly AvgPool2D: all 4,275 collected cases are accounted for across 86
  bounded safe-runner processes: 745 passed and 3,530 followed existing skip
  paths. Every split ended with `SAFE_PYTEST_RESULT: PASS`; no split failed or
  hung. A fresh post-reset Conv2D ULP health check passed both cases before the
  bounded matrix began.
- Fresh-cache anchors passed for row-major L1 height sharding, row-major DRAM
  config placement, and tiled-input Conv2D untilize. The tiled Conv2D anchor
  achieved PCC 0.9999826419; each cold run showed zero JIT cache hits.

The remaining composite files are being executed serially on the device, with
large nightly matrices divided into disjoint bounded pytest groups to avoid
the already-proven cumulative-session infrastructure wedge while retaining
complete item coverage.

### Incremental review correction (2026-08-20)

The first high-effort Opus review reproduced an existing FP32 regression case
that ProgramSpec validation correctly exposed: Halo's untilize compute kernel
needed an explicit `src -> UnpackToSrc` entry when both its input DFB and Dest
were FP32. The review also found that the mechanical kernel rewrite had lost
the legacy stateful NOC padding-read setup and reloaded local NOC coordinates
inside the gather transfer loop. It further requested DFB-aware source
addressing, restoration of non-obvious split-reader invariants, and a hard
compile-time failure for the impossible nonzero-padding/no-scratch state.

All findings were addressed. A follow-up audit also caught that `KernelSpec`
defaults compute kernels to O2 whereas legacy `ComputeConfig` defaults to O3;
Halo now states O3 explicitly. The exact release build passes after these
changes. The existing `test_conv2d_fp32_input_no_fp16_saturation` regression
then passed from a cold cache with 0/19 JIT hits; it requires output magnitude
above 5e7 and PCC at least 0.99. Post-fix full-suite evidence supersedes any
pre-fix suite run.

The second review required two packaging/topology corrections. The installed
kernel glob now includes the shared `*.hpp` implementation, and Gen1 preserves
the legacy split-reader placement exactly: reader 0 uses `RISCV_0`/NOC0 and
reader 1 uses `RISCV_1`/NOC1. The Quasar placement remains an explicit separate
branch.

The third review found a real Gen2-only untilize hazard. Wormhole/BH PACK
recomputes its destination address from the runtime output DFB, so the legacy
single hoisted initialization safely alternates out0/out1. Quasar instead
bakes the destination buffer descriptor into the PACK MOP during init; a
single out0 init would therefore direct odd blocks to the wrong base. The
compute kernel now keeps the unchanged hoisted Gen1 init/uninit path while the
`ARCH_QUASAR` path uses `InitAndUninit` for every alternating block, matching
the proven experimental Quasar kernel. The exact release build and targeted
padded MaxPool2D and FP32 saturation Conv2D cases passed after this change.

The fourth incremental, single-Halo Opus review independently traced both LLK
implementations, verified that the Gen1 path is semantically unchanged, proved
why the Quasar per-block init is necessary, and found no correctness or
performance blocker. Its final verdict was explicit `APPROVED`. No Quasar
validation is claimed or planned for this work; it is explicitly out of scope.
All device validation and acceptance evidence is from the local T3K Wormhole.

After the host-style cleanup, a fifth incremental single-Halo review rechecked
the complete current diff. It verified explicit O3 only on the compute
`KernelSpec`, the unchanged data-movement optimization default, config-tensor
lifetime and cache-hit rebinding, every DFB producer/consumer direction, all
named TT_KERNEL bindings, Gen1 placement and runtime parity, `remote_read`
rejection, CMake installation of the shared kernel header, and the absence of
host convenience aliases or uppercase typed resource identifiers. It ended
with explicit `APPROVED` and no requested changes.

A later focused performance review correctly rejected the first profiled
version even though its functional review had passed. It found that the
row-major reader-0 `wait_front` had moved ahead of padding. The legacy sequence
is reserve/push, issue padding, wait for the resident input shard, then gather;
moving the wait earlier changed the two RISCs' NOC issue timing and produced a
stable width-sharded device regression. Restoring that order and materializing
the row-major input read address once recovered almost all of the loss, but two
independent 20-sample post-warmup traces still measured device median/p95
3,955/4,004.25 ns and 3,973.5/4,013.2 ns against the 3,952/3,993.05 ns
baseline. The review therefore remained `CHANGES_REQUIRED`; a flat median was
not used to excuse the repeatable upper-tail regression.

The same review pointed to the nonzero-pad fill loop, whose generated code had
one halfword load after every halfword store. Follow-up disassembly showed that
merely removing C++ `volatile`, including through `CoreLocalMem<uint16_t>`, did
not remove those readbacks: the SFPI toolchain emits the `sh`/`lhu` pair for
halfword stores. The correct optimization is representation-preserving rather
than qualifier-based. The pad scratch is allocated as `uint32_t`, the stick is
aligned to the device buffer alignment, and the kernel statically requires an
even number of BF16 elements. It now duplicates the arbitrary 16-bit pad value
into both halves of a 32-bit word and fills two elements per store, followed by
a compiler memory barrier before NOC consumes the private L1 scratch. The
generated loop is 16 `sw` instructions with no readbacks instead of 32
`sh`/`lhu` pairs for the frozen width case. The exact release build and an
independent normal-mode width-sharded MaxPool2D correctness run passed. Its 22
Tracy executions also passed; after excluding two warmups, device median/p95
improved to 3,526.5/3,551.2 ns from 3,952/3,993.05 ns, BRISC improved to
3,440/3,456 ns from 3,788/3,805.05 ns, and first-to-last start improved to
195.5/209.05 ns from 201/225.35 ns. The CSV is
`generated/profiler/reports/2026_08_20_11_59_36/ops_perf_results_2026_08_20_11_59_36.csv`.

The other four frozen representatives also completed with all 22 executions
passing. Height-sharded zero padding measured device median/p95
6,421.5/6,553.05 ns and 6,420.5/6,536.9 ns on a repeat, against
6,432/6,520.15 ns baseline; the median is stable and slightly improved, while
the upper 5 percent varies by 16--33 ns across overlapping ranges. The reports
are `2026_08_20_12_02_24` and `2026_08_20_12_08_17`. Block sharding measured
4,776.5/4,807.8 ns and 4,770/4,826 ns on repeat, against 5,150/5,182.5 ns;
reports `2026_08_20_12_04_10` and `2026_08_20_12_10_02`. DRAM-config placement
measured 1,366/1,402.35 ns and 1,366/1,434.05 ns on repeat, against
1,565.5/1,640.9 ns; reports `2026_08_20_12_04_32` and
`2026_08_20_12_10_22`. The tiled Conv2D untilize path measured
4,053.5/4,199.4 ns against 4,125.5/4,249.25 ns; report
`2026_08_20_12_05_41`. Every final device median is nonregressed, and four of
five improve materially. First-to-last start medians are also nonregressed for
width, block, DRAM, and tiled; height moved by only 8.5 ns within its baseline
range.

Host trace values are substantially noisier than device values, so they were
not interpreted from one late sample in isolation. The same migrated host
program, before the later kernel-only ordering and fill corrections, measured
height 8,143.5/9,581.35 ns, width 8,190/9,721.5 ns, block
7,797/8,649.35 ns, DRAM-config 8,294.5/9,973.25 ns, and tiled
8,984.5/9,981.15 ns median/p95. Against the frozen baselines, all p95 values
improve, three medians improve or differ by at most 1.2 percent, and the other
two differ by 2.0--2.9 percent inside broadly overlapping distributions. The
subsequent source changes were kernel-header-only and cannot add cached host
launch work; their device improvements are measured separately above.

The final incremental single-Halo Opus review was run with the required Opus
high-effort command after the packed fill and ordering correction. It traced
the legacy and migrated reader ordering, proved stable row-major versus dynamic
tiled source addressing, proved the packed fill's alignment, size, byte
identity, and compiler-ordering preconditions, rechecked cache rebinding and
Gen1 placement, and verified explicit compute O3 with unchanged data-movement
O2. It found no Wormhole correctness or performance blocker and ended with
explicit `APPROVED`. Its only architecture observation concerned cached L1 on
an unvalidated hypothetical Quasar nonzero-padding path; Quasar validation and
claims remain explicitly out of scope.

One monolithic unit MaxPool2D session later accumulated device state and hit a
five-second dispatch timeout at an L1 output-memory case. Automatic triage did
not produce a report before becoming CPU-bound. After a forced exit and
`tt-smi -r`, the exact surfaced case passed alone and again inside a bounded
group; all four bounded groups completed with the 104-pass/13-skip totals
above. Treat long composite sessions as an infrastructure risk: divide them
into disjoint pytest-split groups, but still run every collected item.

The final nightly Conv2D matrix was run as 12 disjoint groups. Groups 1 through
4 cover 804 selected items with 748 passes and 56 expected skips: group 1 is
146/55, groups 2 and 3 are 201/0 each, and group 4 is 200/1.

Nightly Conv2D group 8 later reproduced the same long-session infrastructure
failure pattern: after more than five minutes, dispatch timed out and invoked
`tt-triage`. Triage consumed CPU for about 80 seconds, emitted only its own
initialization stack, wrote no CSV, and exited nonzero; the safe runner then
reset the device. Recollecting the exact 201 selected node IDs and running them
as four explicit subchunks produced 51/51, 51/51, 44 passes plus seven expected
skips, and 35 passes plus 13 expected skips. Thus all 201 cases pass (181/20)
and no deterministic failing node exists. Keep the full coverage, but cap these
large composite processes near 50 cases on this machine when cumulative device
state makes a larger process unreliable.

All 2,406 nightly Conv2D cases are now accounted for: 1,689 passed and 717
followed existing skip paths. Per-group pass/skip counts were 146/55, 201/0,
201/0, 200/1, 173/28, 179/22, 177/24, 181/20, 50/151, 24/177, 30/171, and
127/68. The five representative post-migration Tracy comparisons are recorded
above and show nonregressed device medians for every frozen specialization.

### Reliable automatic hang triage on T3K (2026-08-20)

The old safe-pytest timeout hook was not reliable under the failure mode where
it mattered: it relied on implicit `in_use` device discovery after Inspector
state was already damaged, buffered all Python output, had no execution bound,
reused one CSV path, and could be invoked again during failed teardown. The
safe runner now targets `--dev=all`, emits diagnostics unbuffered, bounds each
NOC attempt to 30 seconds plus a five-second forced-kill grace period, falls
back from NOC0 to NOC1 using separate CSVs, and uses a PID-scoped run-once flag.
Hang classification keys off flag existence rather than nonempty log size, so
a hard failure that emits no bytes still takes the reset-and-exit-2 path.

The repository's existing intentional-hang fixture exercised the actual hook
on the local T3K. NOC0 completed in 5.27 seconds, wrote an 18,324-byte CSV, and
identified the three TRISCs stopped in `add_2_tiles_hang.cpp:40`. Safe-pytest
then reset all four local PCI devices, removed its dirty sentinel, and returned
the documented hang code 2. A healthy existing Conv2D ULP case passed before
the injection, and subsequent bounded MaxPool chunks passed normally. The
incremental high-effort Opus review returned explicit `APPROVED`; the isolated
fix is commit `9e9479b7ba2` and Bead `tt-metal-rf8.15` is closed.

The complete nightly MaxPool2D matrix is accounted for: 4,646 collected cases
produced 973 passes and 3,673 existing documented skips across 93 bounded,
disjoint safe-runner groups. Every nonempty group ended with
`SAFE_PYTEST_RESULT: PASS`, with no failure, hang, reset, or triage event. A
requested 94th duration-based pytest-split group contained zero selected cases
and therefore reported `No tests collected`; it is bookkeeping only and adds
no coverage. The 973 + 3,673 total proves that all 4,646 collected cases ran.

### Final post-packed exhaustive matrix closure (2026-08-20)

After the packed UINT32 padding-fill correction and final Opus approval, all
three large composite matrices were rerun from the final source. The exact
accounting is:

- Nightly Conv2D: all 2,406 collected cases, with 1,689 passed and 717 following
  existing skip paths.
- Nightly MaxPool2D: all 4,646 collected cases, with 973 passed and 3,673
  following existing skip paths. The requested 94th split selected zero cases;
  the 93 nonempty splits all ended with `SAFE_PYTEST_RESULT: PASS`.
- Nightly AvgPool2D: all 4,275 collected cases, with 745 passed and 3,530
  following existing skip paths.

The final Conv2D group 18 first accumulated device state and hung after 45
passes and four expected skips. After the safe runner reset all four devices,
the exact 50 selected node IDs passed in four disjoint retries: 13/0, 13/0,
11/2, and 8/3 pass/skip. AvgPool2D group 18 likewise hung after 20 passes and
four expected skips; its exact 50 node IDs completed as 13/0, 7/6, 0/13, and
0/11 after reset. No deterministic failing case existed in either group.

AvgPool2D groups 19 through 23 then passed normally. Group 24 later reproduced
the same cumulative-state failure, including an already-occupied Inspector RPC
port and stale UMD lock holder. Bounded NOC0/NOC1 triage timed out without a
CSV; the cascading fixture errors were interrupted and all four PCI devices
were explicitly reset. Every one of the exact 50 group-24 node IDs then
completed in four fresh retries: 10/3, 0/13, 0/13, and 10/1 pass/skip. This
included the surfaced asymmetric-padding 576-channel case, proving the event
was not a deterministic packed-fill or Halo failure. Groups 25 through 86 all
ended with the safe runner's PASS sentinel.

The three final totals account for 11,327 collected composite cases without a
new skip, xfail, relaxed threshold, or omitted node. Together with the complete
unit matrices, explicit 28/28 Halo reshard anchor, cold-JIT/cache evidence,
release build, disassembly proof, nonregressed Tracy results, and final explicit
Opus `APPROVED`, these post-packed reruns close Halo's T3K Wormhole acceptance
surface. No Quasar validation is claimed.

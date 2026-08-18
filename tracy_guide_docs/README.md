# Tracy Prefill Profiling Guide

This directory is the working reference for our Tracy prefill-profiling session. It contains renamed copies of every profiler artifact we analyze, a reusable analysis script, known-good commands, and conclusions from each capture.

## Directory contents

```text
tracy_guide_docs/
├── README.md
├── analyze_tracy_csv.py
├── captures/
│   └── gemma4-31b-prefill-trace-1x4.csv
├── logs/
    ├── gemma4-31b-prefill-trace-device.csv
    ├── gemma4-31b-prefill-trace-host-data.csv
    ├── gemma4-31b-prefill-trace-host-times.csv
    └── gemma4-31b-prefill-terminal-excerpt.log
└── tt_perf_report/
    ├── README.md
    ├── gemma4-prefill-detailed.csv
    ├── gemma4-prefill-by-{op,category,memory}.csv
    ├── gemma4-prefill-per-device.csv
    └── matching PNG plots and invocation logs
```

Artifact provenance:

- `gemma4-31b-prefill-trace-1x4.csv` is the automatically generated 18:53:48 report from the successful Gemma capture. The manually generated 18:55:57 report came from the same raw logs and was the copy initially analyzed.
- The host and device logs are snapshots from `generated/profiler/.logs/` and the 18:53:48 report directory.
- The live terminal had been cleared before this directory was created, so its retained file is a recovered excerpt rather than a complete terminal transcript.
- `tt_perf_report/` contains version 1.2.8 analysis output derived from the archived Gemma capture.
- The copied raw artifacts currently occupy approximately 7.6 GB. `.gitignore` excludes the large CSV snapshots so they cannot be committed accidentally.

## Recommended walkthrough order

Read the guide in this order when learning:

1. **Start here: first successful Tracy capture**
2. **Now run prefill for a specific model**
3. **Gemma 4 31B 1×4: traced prefill**
4. **Reading `ops_perf_results_*.csv`**
5. **Generated logs, profiles, and reports**
6. **Gemma example capture** and **Analyze with `tt-perf-report`**
7. **Ways to run Tracy** and **Specialized device profiling modes**
8. **Reusable Tracy utilities in this repository**
9. **What to think about when profiling**

The sections between these walkthrough topics are reference material; they can be consulted only when needed.

## Start here: first successful Tracy capture

This is the shortest path for someone who has never used Tracy.

Prerequisites:

- Run from the tt-metal repository root.
- Use an already configured environment with the Python virtual environment active.
- Ensure a Tenstorrent device is available and the current checkout has been built.
- Do not run `install_dependencies.sh` merely to use Tracy on an existing configured machine.

### 1. Remove stale raw profiler logs

```bash
rm -rf generated/profiler/.logs
```

This preserves existing timestamped reports.

### 2. Run the small mixed-operation profiler test

```bash
python -m tracy -p -r -v -m pytest \
  tests/ttnn/tracy/test_various_ops_profile.py \
  -v -s
```

This is a Tracy operation-report integration test, not a model benchmark. It:

- Creates two sub-devices on one physical device.
- Creates two tiled BF16 tensors with shape `[1, 1, 64, 64]`.
- Runs one small matmul on the first sub-device.
- Runs add, multiply, subtract, and another add across the two sub-devices.
- Synchronizes the device and removes the sub-device configuration.

The varied operations make it easy to confirm that Tracy records operation names, device timing, sub-device placement, and multiple kernel types. Its timings should not be treated as model-performance targets.

### 3. Find the generated report

The terminal prints a path similar to:

```text
generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
```

Because the command used `-r`, do not run `process_ops_logs.py` again.

### 4. Analyze it

```bash
python3 tracy_guide_docs/analyze_tracy_csv.py \
  generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
```

Expected evidence:

- The pytest test passed.
- At least one device appears.
- `MatmulDeviceOperation` and binary operations appear.
- Device kernel durations are populated.
- The CSV contains more than setup-only conversion operations.

### 5. Move to the intended workload

- For operation-level prefill tuning, use an existing model test with eager execution.
- For traced end-to-end latency, use a test that performs capture, warm replay, and a signposted measured replay.
- Add custom code only if the existing test cannot provide the required measurement boundary or lifecycle.

If this first capture fails, debug the Tracy/build/device setup before running a large model. If it passes but a model report contains only conversion operations, the problem is the model measurement mode or window rather than the basic Tracy installation.

## What Tracy is

Tracy observes execution across the stack:

```text
Python model
  → TTNN operation
  → host runtime and dispatch
  → device firmware
  → data-movement and compute kernels
```

Tracy answers questions such as:

- Which TTNN operations dominate device time?
- Is time spent on the host, in gaps, or in device kernels?
- Are all mesh devices balanced?
- Did the program cache warm successfully?
- Is a kernel compute-bound, bandwidth-bound, or communication-bound?

Tracy and Metal Trace are different:

- **Tracy** measures execution.
- **Metal Trace** records device command sequences and replays them.
- Tracy can measure eager execution, trace capture, and trace replay.

### Where Tracy comes from

Tracy was not originally created by Tenstorrent. It is an open-source profiler from the upstream Tracy project. Tenstorrent:

- Maintains a fork at `github.com/tenstorrent/tracy`.
- Vendors the fork under `tt_metal/third_party/tracy/`.
- Extends it with Tenstorrent device, Tensix, firmware, and Metal Trace support.
- Provides the `python -m tracy` wrapper, TTNN operation metadata, device-log correlation, operation reports, and visualization integration in this repository.

The upstream project supplies the core capture format, instrumentation model, timeline, GUI, and tools such as `tracy-capture`. The TT-Metal workflow adds the hardware-specific measurements and `ops_perf_results_*.csv` processing we use for model profiling.

## Standard profiling command

```bash
python -m tracy -p -r -v -m pytest <test> <pytest arguments>
```

Important options:

- `-p`: collect explicitly instrumented Python zones instead of every Python function.
- `-r`: process logs and generate `ops_perf_results_*.csv` automatically.
- `-v`: print verbose profiling information.
- `-m pytest`: execute pytest under Tracy.
- `--op-support-count N`: allow Tracy to retain metadata for up to `N` operations.
- `--dump-device-data-mid-run`: periodically drain device-profiler data during a large eager run.
- `--device-trace-profiler`: collect device-side trace timing. Do not combine it with `--dump-device-data-mid-run`.

`-r` already runs the operation-log processor. Do not run `process_ops_logs.py` afterward unless the capture was performed without `-r` or you intentionally want to regenerate a report.

## Ways to run Tracy

### Profile a Python script

```bash
python -m tracy my_script.py --script-argument value
```

This profiles a script directly. Add `-r` when an operation CSV is required:

```bash
python -m tracy -p -r -v my_script.py
```

### Profile a Python module or pytest

```bash
python -m tracy -p -r -v -m pytest path/to/test.py -k "test selector" -v -s
```

`-m` has the same purpose as Python's normal `-m`: it executes a library module such as `pytest`.

### Profile only a manually selected Python region

Use the profiler API in code:

```python
from tracy import Profiler

profiler = Profiler()
profiler.enable()
function_under_test()
profiler.disable()
```

Run the script or test with `-p`. Partial mode records Python calls only while the profiler is enabled, which is useful when full-session Python profiling is too large.

### Profile Python line by line

```bash
python -m tracy -p -l -m pytest path/to/test.py -k "test selector"
```

Line profiling is useful when a slow Python function does substantial inline work that child-function zones do not explain. It produces much more data and is supported only with partial profiling.

### Profile host code only

```bash
python -m tracy -p -r --no-device -m pytest path/to/test.py
```

Use this for Python/C++ host overhead, dispatch preparation, model preprocessing, or framework orchestration when device detail is unnecessary.

### Capture directly to a `.tracy` file

Start the capture server before the application:

```bash
build/tools/profiler/bin/tracy-capture -o prefill.tracy
```

Then run a Tracy-enabled application in another terminal. The `.tracy` file can be opened later in Tracy's native GUI. These files compress well for transfer.

### Use the live Tracy GUI

Run Tracy Profiler locally or remotely and connect it to the profiled application, normally on port 8086. For a remote host:

```bash
ssh -NL 8086:127.0.0.1:8086 user@remote-host
```

The GUI provides a timeline, nested zones, messages, frame markers, thread activity, and interactive navigation that cannot be represented fully in `ops_perf_results_*.csv`.

### Export a `.tracy` capture

```bash
build/tools/profiler/bin/tracy-csvexport prefill.tracy > prefill-zones.csv
```

This exports generic Tracy timeline data. It is different from TT-Metal's merged `ops_perf_results_*.csv`.

### Reprocess existing raw logs

```bash
python -m tracy --process-logs-only -r
```

or:

```bash
python tools/tracy/process_ops_logs.py --date
```

Use this only when raw logs already exist and a new capture is not required.

### Use the legacy convenience wrapper

```bash
python tools/tracy/profile_this.py -n experiment-name -c "pytest path/to/test.py"
```

This wrapper runs the command and generates a report, but the repository marks it for eventual replacement by `python -m tracy`.

### Instrument C++ code

Tracy-enabled builds collect marked C++ zones:

```cpp
#include "tracy/Tracy.hpp"

void function_under_test() {
    ZoneScoped;
    // Work to profile.
}
```

Other Tracy macros can add messages and frame markers, such as `TracyMessageL` and `FrameMarkNamed`.

## Specialized device profiling modes

### Dispatch-core profiling

```bash
python -m tracy -p -r --profile-dispatch-cores -m pytest path/to/test.py
```

Use this to investigate command-queue and dispatch-core overhead. It cannot be combined with `--dump-device-data-mid-run`.

### Device-side Metal Trace timing

```bash
python -m tracy -p -r --device-trace-profiler -m pytest path/to/trace_test.py
```

Use this for detailed device-side trace durations. It cannot be combined with `--dump-device-data-mid-run`.

### Device memory profiling

```bash
python -m tracy -p -r --device-memory-profiler -m pytest path/to/test.py
```

This enables profiling of allocated device L1 and DRAM buffers, useful when performance may be constrained by placement, fragmentation, or capacity.

### NoC event tracing

```bash
python -m tracy -p -r --collect-noc-traces -m pytest path/to/test.py
```

This captures Network-on-Chip events and can generate `npe_viz` timelines for TT-NN Visualizer when `tt-npe` is installed. Use it for congestion, communication routing, and collective investigations.

### Hardware performance counters

```bash
python -m tracy \
  --profiler-capture-perf-counters=fpu,pack,unpack,l1_0,instrn \
  -m "pytest path/to/test.py -x -v"
```

Counter groups:

- `fpu`: FPU/SFPU/math utilization.
- `pack`: packer activity, destination reads, and scoreboard behavior.
- `unpack`: unpacker activity, source writes, and math-pipeline stalls.
- `l1_0`: L1 ports 0–7, including unpacker, packer, TDMA, and NoC ring 0.
- `l1_1`: L1 ports 8–15 and NoC ring 1.
- `instrn`: instruction availability, issue rate, and thread stalls.
- `all`: recommended broad first pass.
- `l1_2`, `l1_3`, `l1_4`: additional Blackhole-only L1/NoC visibility.

`l1_0` and `l1_1` share a hardware mux and require separate runs. The resulting columns can expose FPU utilization, thread stalls, pack/unpack efficiency, semaphore waits, write-port contention, L1 utilization, NoC backpressure, and instruction issue rates.

### Synchronized host/device profiling

```bash
python -m tracy -p -r --sync-host-device -m pytest path/to/test.py
```

This forces host/device synchronization for easier timeline correlation. It perturbs normal asynchronous execution, so use it diagnostically rather than as the final latency benchmark.

### Mid-run device-data draining

```bash
python -m tracy -p -r --dump-device-data-mid-run -m pytest path/to/long_test.py
```

Use this for long eager runs that risk overflowing on-device profiler buffers. It is incompatible with dispatch-core and device-trace profiling.

### Sum and accumulate modes

```bash
python -m tracy --enable-sum-profiling -m pytest path/to/test.py
python -m tracy --enable-accumulate-profiling -m pytest path/to/test.py
```

- Sum profiling aggregates device-profiler samples.
- Accumulate profiling packs repeated worker-core kernel invocations before draining to DRAM.
- Accumulate mode stores no per-operation IDs and therefore cannot be combined with `-r`.

## Output and capture controls

- `-o <folder>`: choose the profiler artifact directory.
- `-n <name>`: append an experiment name to report artifacts.
- `--web-app-port <port>`: choose the Tracy WASM UI port.
- `-t <port>`: choose the internal capture port.
- `--no-op-info-cache`: emit full operation metadata even for cached operations.
- `--op-support-count <N>`: increase the number of operations whose metadata can be retained.
- `--child-functions f1,f2`: include selected child durations in parent operation accounting.
- `--disable-device-data-dump-to-files`: avoid writing device data to files.
- `--disable-device-data-push-to-tracy`: avoid pushing device data into the GUI.
- `--check-exit-code`: stop post-processing if the test command fails.
- `--no-runtime-analysis`: disable C++ runtime post-processing and use legacy processing.
- `--no-capture-tool`: run without starting Tracy's capture helper; useful with an independently managed capture server.
- `--tracy-tools-folder <folder>`: use Tracy binaries from a non-default location.
- `-a <analysis-type>`: request an additional supported device-analysis pass.

## What Tracy can investigate

- End-to-end model regions using signposts.
- Python function and line-level overhead.
- C++ runtime zones and nested call timing.
- TTNN operation order, host duration, and dispatch gaps.
- Device firmware and kernel duration.
- BRISC, NCRISC, TRISC0, TRISC1, TRISC2, and ERISC timing.
- Per-core minimum, maximum, and average duration.
- Core-grid utilization and load imbalance.
- Circular-buffer waits and producer/consumer backpressure.
- Program compilation, hashes, and eager program-cache reuse.
- Tensor shapes, layouts, datatypes, memory placement, and sharding.
- Metal Trace capture and replay sessions.
- Dispatch-core and command-queue behavior.
- Device L1 and DRAM allocation behavior.
- NoC traffic, congestion, collectives, and fabric behavior.
- FPU, SFPU, packer, unpacker, instruction, and L1 hardware counters.
- Compute-bound versus bandwidth-bound classification from performance-model columns.
- Cross-device balance and the mesh critical path.
- Comparisons between configurations, commits, shapes, fidelity levels, and sharding strategies.
- Interactive timeline investigation in Tracy GUI.
- Processed report exploration in TT-NN Visualizer, including NPE views when NoC traces exist.

## Reusable Tracy utilities in this repository

You do not need a new `.py` test for each profile. The normal workflow is to run Tracy around an existing test, demo, module, or script. Add code only when the existing entry point does not provide the warm-up, synchronization, repetition, or measurement boundary needed by the experiment.

### Core Python API

File: `tools/tracy/__init__.py`

- `Profiler`: enable and disable partial Python profiling.
- `signpost(header, message=None)`: emit a labeled row in the operation report and a message in the Tracy timeline.
- `runctx(...)`: execute Python code under profiling; mainly used by the CLI.
- `generate_report(...)`: export host data and build the operation report.
- `run_report_setup(...)`: start the Tracy capture process.
- `get_available_port()`: select an available capture port.

Typical use:

```python
from tracy import Profiler, signpost

profiler = Profiler()
profiler.enable()

signpost("start")
function_under_test()
ttnn.synchronize_device(device)
signpost("stop")

profiler.disable()
```

### Existing pytest fixture

The repository root `conftest.py` provides:

```python
@pytest.fixture(scope="function")
def tracy_profile():
    from tracy import Profiler

    profiler = Profiler()
    profiler.enable()
    yield
    profiler.disable()
```

Inject `tracy_profile` into an existing test and run with `python -m tracy -p ...` when partial Python profiling of that test is sufficient.

### Low-level TTNN profiler hooks

File: `ttnn/ttnn/profiler.py`

- `start_tracy_zone(...)` / `stop_tracy_zone(...)`: explicit Python Tracy zones.
- `tracy_message(...)`: timeline message.
- `tracy_frame()`: frame marker.
- `get_latest_programs_perf_data()`: latest device-program profiler data.
- `get_all_programs_perf_data()`: full collected program data.
- `ttnn.ReadDeviceProfiler(device)`: manually drain device-profiler buffers.

Most model work should prefer `signpost()` and the Tracy CLI. Use these hooks for custom instrumentation, sweep integration, or explicit profiler-buffer management.

### Report-generation utilities

File: `tools/tracy/process_ops_logs.py`

- `process_ops(...)`: merge host and device logs into `ops_perf_results_*.csv`.
- `import_tracy_op_logs(...)`: load exported host operations.
- `append_device_data(...)`: correlate host operations with device measurements.
- `generate_reports(...)`: write the final reports.
- `load_device_perf_report(...)`: load current C++-processed device data.

The current default path uses `cpp_device_perf_report.csv`. `process_device_log.py` is the legacy Python fallback used by conflicting modes or `--force-legacy-device-logs`.

### Model-oriented profiler driver

File: `tools/tracy/process_model_log.py`

- `run_device_profiler(command, subdir, ...)`: run an existing command under Tracy and collect its report.
- `post_process_ops_log(...)`: load and filter the latest report by signpost or operation.
- `get_latest_ops_log_filename(...)`: locate the newest report.
- `requires_multi_pass_profile(...)`, `get_multi_pass_configs(...)`, `merge_pass_csv(...)`: handle separate L1 counter passes and merge them.

This is useful for automating repeated model experiments without creating a specialized Tracy test.

### Generic model performance helpers

File: `models/perf/device_perf_utils.py`

- `run_device_perf(...)`: execute repeated profiler runs and aggregate selected columns.
- `run_device_perf_detailed(...)`: include standard deviation, ranges, and per-operation detail.
- `post_process_ops_log_detailed(...)`: filter signposts and handle multi-device interleaving.
- `check_device_perf(...)`: compare results against expected thresholds.
- `prep_device_perf_report(...)`: produce model device-performance reports and CI benchmark data.
- `run_model_device_perf_test(...)`: complete run, check, and report workflow.

Use these when the objective is a repeatable benchmark or CI threshold rather than one interactive capture.

DeepSeek-specific extensions are in `models/demos/deepseek_v3_d_p/utils/perf_utils.py`:

- `run_model_device_perf_test_with_merge(...)`
- `run_model_device_perf_test_per_op(...)`

### Existing traced-prefill session helper

File: `models/demos/gemma4/tests/unit/tracy_prefill_common.py`

- `run_prefill_trace_tracy_session(...)`: load, capture, warm replay, and signposted measured replay.
- `run_prefill_trace_capture_and_replays(...)`: lower-level replay orchestration.
- `_signpost_start_stop(...)`: reusable signpost context manager.
- `_flush_device_profiler(...)`: read profiler buffers and synchronize.

The corresponding `test_prefill_trace_tracy_csv.py` is a reusable reference for models that need a dedicated trace-session recipe.

### Existing operation-report analyzer

File: `models/tt_transformers/scripts/op_perf_results.py`

This provides LLM-oriented operation CSV summarization, including signpost selection. The local `tracy_guide_docs/analyze_tracy_csv.py` complements it with automatic trace-session selection and per-device critical-path aggregation.

### Comparison and validation

- `tools/tracy/compare_full_op_report.py`: cell-by-cell comparison of two full operation reports.
- `tools/tracy/compare_ops_logs.py`: compare device-report processing paths.
- `tools/tracy/perf_counter_analysis.py`: compute hardware-counter metrics.
- `tools/tracy/device_post_proc_config.py`: device-analysis configuration.
- `tools/tracy/common.py::clear_profiler_runtime_artifacts()`: remove current profiler runtime artifacts.

### Multi-host Tracy

File: `ttnn/ttnn/distributed/ttrun.py`

- `parse_tracy_args(...)`: parse `tt-run --tracy` options.
- `wrap_program_with_tracy(...)`: wrap each rank with distinct Tracy ports and artifact directories.
- `_normalize_program_for_tracy(...)`: convert commands into Tracy-compatible module execution.

Example pattern:

```bash
tt-run <distributed options> --tracy "-r" pytest path/to/test.py
```

Distributed Tracy integration tests are available under `tests/ttnn/distributed/`.

### Sweep and microbenchmark integration

File: `tests/sweep_framework/sweep_utils/perf_utils.py`

- `gather_single_test_perf(...)`
- `execute_test(...)`
- `run_single(...)`
- `run_with_cache_comparison(...)`

These use `ttnn.get_latest_programs_perf_data()` to collect per-vector sweep performance without creating a Tracy test for each vector.

### C++ and device-kernel instrumentation

- Standard host C++ zones: `ZoneScoped`, `TracyMessage`, and related macros from `tracy/Tracy.hpp`.
- Category-gated debug zones: `tt_metal/tools/profiler/tracy_debug_zones.hpp`, including `TTZoneScopedD`, `TTZoneScopedDN`, `TTTracyMessageDL`, and `TTTracyPlotD`.
- Metal Trace correlation: `tt_metal/tools/profiler/tt_metal_tracy.hpp`.
- Device kernel zones: `DeviceZoneScopedN("zone")` from `tt_metal/tools/profiler/kernel_profiler.hpp`.

Category-gated zones can be enabled through builds such as:

```bash
./build_metal.sh --build-perf-debug dispatch,program
```

### Visual and external analysis tools

- `tools/tracy/serve_wasm.py`: local WASM Tracy viewer and live-reload management.
- `build/tools/profiler/bin/tracy-capture`: native capture to `.tracy`.
- `build/tools/profiler/bin/tracy-csvexport`: export generic Tracy zones.
- TT-NN Visualizer: upload operation-report directories, including optional NPE data.
- External [`tt-perf-report`](https://github.com/tenstorrent/tt-perf-report): Tenstorrent's separately maintained operation-table, roofline, advice, grouping, and visualization tool. Install it from PyPI rather than looking for its implementation in this repository.

### Useful reference tests

- `tests/ttnn/tracy/test_perf_op_report.py`: CSV schema and golden checks.
- `tests/ttnn/tracy/test_trace_runs.py`: trace IDs and replay sessions.
- `tests/ttnn/tracy/test_dispatch_profiler.py`: dispatch profiling.
- `tests/ttnn/tracy/test_profiler_sync.py`: host/device synchronization.
- `tests/ttnn/tracy/test_various_ops_profile.py`: compact mixed-operation workload.
- `tests/tt_metal/tools/profiler/test_device_profiler.py`: broad device-profiler feature matrix.
- `models/tt_transformers/tests/test_device_perf.py`: wrapping an existing model demo with performance helpers.
- `tests/ttnn/profiling/profile_host_overhead_with_tracy.py`: repeated host-overhead profiling.

These are templates and validation tests, not prerequisites for profiling a model.

### When to create a new test

Create a dedicated pytest file only when:

1. The profiling setup itself should be repeatable and reviewed.
2. CI must validate report columns or enforce a performance threshold.
3. The workload needs a custom sequence such as compile → warm → signposted replay.
4. A special topology, cache state, trace lifecycle, or profiler flush cannot be expressed through an existing entry point.

Otherwise, profile an existing test/demo and reuse the utilities above.

## Clean capture hygiene

Remove stale raw logs before a new run, but preserve prior reports:

```bash
rm -rf generated/profiler/.logs
```

Use a new report timestamp for every experiment. Before deleting or replacing any report we care about, copy it into this directory with a descriptive name.

## Now run prefill for a specific model

Reuse an existing prefill test or demo for the model:

```bash
rm -rf generated/profiler/.logs

export HF_MODEL=/path/to/model/checkpoint

python -m tracy -p -r -v \
  --op-support-count 100000 \
  -m pytest <existing-prefill-test> \
  <model-topology-and-shape-arguments> \
  -v -s
```

Before running, decide:

- Use eager execution when the goal is individual operation analysis.
- Use a trace-aware test when the goal is replay latency.
- Add `--dump-device-data-mid-run` to a long eager run when profiler-buffer capacity is a concern.
- Record checkpoint, topology, layer count, batch size, sequence length, dtype, fidelity, and execution mode.
- Confirm the selected test compiles/warms outside the measured pass or emits signposts around the intended window.

Expected report:

- The expected devices and model operations appear.
- Matmul, attention, normalization, elementwise, cache, and collective operations appear when used by the model.
- A warmed eager pass has program-cache reuse, or a traced pass has trace/replay IDs.
- Initialization and weight-loading operations are excluded from the measured interpretation.

## Gemma 4 31B 1×4: traced prefill

Use this test to learn trace capture, warm replay, signposts, and measured replay:

```bash
rm -rf generated/profiler/.logs

export HF_MODEL=google/gemma-4-31b-it
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000

TT_VISIBLE_DEVICES=0,4,12,8 \
python -m tracy -p -r -v -m pytest \
  models/demos/gemma4/tests/unit/test_prefill_trace_tracy_csv.py \
  -k "batch1-prefill_128-1x4" \
  -v -s --timeout=1800
```

The successful run followed:

```text
model load
→ trace capture
→ warm trace replay
→ start signpost
→ measured trace replay
→ stop signpost
```

The prompt contained five real tokens but ran in the padded 128-token trace bucket. That is expected.

## Signposts

Signposts are named host-timeline markers. They label the region of interest; they do not enable or disable profiling.

```python
from tracy import signpost
import ttnn

# Compile and warm the program outside the measured region.
model.forward(inputs)
ttnn.synchronize_device(mesh_device)

signpost("start")
model.forward(inputs)
ttnn.synchronize_device(mesh_device)
signpost("stop")
```

Synchronization is important because device dispatch is asynchronous. Without it, `stop` can be emitted before the measured device work completes.

For trace replay, internal operation rows may retain host timestamps from trace capture. Select the measured replay using CSV order or `METAL TRACE REPLAY SESSION ID`, not only by comparing each operation's `HOST START TS` against the signpost timestamps.

## Reading `ops_perf_results_*.csv`

### 1. Validate the capture

Check:

- The expected number of devices appears.
- The test passed.
- The expected operations exist.
- Signposts exist when the test emits them.
- Trace/replay IDs are populated for traced execution.
- A warmed eager run contains cache hits.

A report containing only tilize, untilize, padding, and typecast operations usually captured setup or preprocessing rather than model inference.

### 2. Find the measured window

For eager execution:

- Prefer explicit `start` and `stop` signposts.
- Otherwise identify the repeated warmed sequence using program-cache hits and operation order.

For Metal Trace:

- Find the replay-session IDs.
- The final replay enclosed by signposts is the measured session.
- Filter `METAL TRACE REPLAY SESSION ID` to that session.

### 3. Do not add devices together for latency

One logical mesh operation produces one row per participating device. Devices generally execute concurrently.

Correct approach:

1. Sum operation durations independently for each `DEVICE ID`.
2. Compare the device totals.
3. Use the maximum device total as the critical-path estimate.

Adding all devices together measures aggregate device work, not wall-clock latency.

### 4. Group operations

Group by `OP CODE`, then sum `DEVICE KERNEL DURATION [ns]` independently per device. For every operation group, use the maximum per-device sum.

This answers:

- How much of the critical path is matmul?
- How much is attention?
- How much is communication?
- Are layout conversions significant?
- Is one device slower than the others?

### 5. Inspect individual hot operations

After identifying a large operation group, sort its rows by `DEVICE KERNEL DURATION [ns]`, then compare:

- Input and output shapes.
- Datatypes and math fidelity.
- DRAM versus L1 placement.
- Sharding and parallelization strategy.
- Core count and core grid.
- Program configuration in `ATTRIBUTES`.
- Performance-model compute and bandwidth estimates.
- Per-core minimum, maximum, and average durations.
- Circular-buffer waits, NOC utilization, and DRAM utilization when populated.

## Most useful columns

- `OP CODE`: TTNN device-operation name.
- `OP TYPE`: device operation or signpost.
- `GLOBAL CALL COUNT`: invocation identifier; it is not elapsed time.
- `DEVICE ID`: device that produced the row.
- `ATTRIBUTES`: operation and program configuration.
- `MATH FIDELITY`: compute precision/fidelity setting.
- `CORE COUNT`: number of worker cores used.
- `HOST START TS`, `HOST END TS`, `HOST DURATION [ns]`: host timing.
- `DEVICE FW DURATION [ns]`: device firmware envelope.
- `DEVICE KERNEL DURATION [ns]`: primary kernel timing.
- `DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG`: core-balance information.
- `OP TO OP LATENCY [ns]`: gap from the previous profiled operation; interpret only inside a valid window.
- `INPUT_*`, `OUTPUT_*`: tensor shape, layout, datatype, and memory placement.
- `PROGRAM HASH`, `PROGRAM CACHE HIT`: compiled-program identity and eager cache reuse.
- `METAL TRACE ID`: captured Metal Trace identity.
- `METAL TRACE REPLAY SESSION ID`: replay instance.
- `PM IDEAL`, `PM COMPUTE`, `PM BANDWIDTH`: performance-model estimates.
- `PM FPU UTIL (%)`: estimated floating-point utilization.
- NOC/DRAM/ETH utilization columns: movement and communication indicators.

## Automated analysis

Run:

```bash
python3 tracy_guide_docs/analyze_tracy_csv.py \
  tracy_guide_docs/captures/gemma4-31b-prefill-trace-1x4.csv
```

The script automatically chooses the highest numeric replay-session ID. Override it with:

```bash
python3 tracy_guide_docs/analyze_tracy_csv.py <csv> --session 2 --top 25
```

For a signposted eager report without replay IDs, it analyzes device rows between the final `start` and `stop`.

## Analyze with `tt-perf-report`

[`tt-perf-report`](https://github.com/tenstorrent/tt-perf-report) is a separate Tenstorrent repository and PyPI package. It consumes `ops_perf_results_*.csv`; it does not capture a workload itself.

Install it in an isolated environment:

```bash
pipx install tt-perf-report
```

Basic use:

```bash
tt-perf-report generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
```

For paired signposts, select both boundaries explicitly and save reusable outputs:

```bash
tt-perf-report \
  tracy_guide_docs/captures/gemma4-31b-prefill-trace-1x4.csv \
  --start-signpost start \
  --end-signpost stop \
  --no-color \
  --csv tracy_guide_docs/tt_perf_report/gemma4-prefill-detailed.csv \
  --summary-file tracy_guide_docs/tt_perf_report/gemma4-prefill-by-op
```

The default behavior selects the last signpost, which is convenient when a test places one marker before its final performance pass. With separate `start` and `stop` markers, do not rely on that default.

Useful options:

- `--id-range 5-10`, `31-`, or `-12`: narrow the detailed table to a model component after first inspecting the generated IDs.
- `--group-by op|memory|category`: summarize operation names, input-0 memory layouts, or compute/data-movement/tensor-manipulation categories.
- `--no-merge-devices`: retain device-specific rows and summaries.
- `--csv <file>`: export the detailed derived table.
- `--summary-file <base>`: export a summary CSV and stacked PNG.
- `--no-advice`: omit generic tuning advice.
- Multiple input CSVs: combine traces from the same multi-host workload; device IDs are offset per input file.

It adds derived columns that are easier to act on than the raw report:

- Device time and op-to-op gap in microseconds.
- Total-time percentage.
- Core count and input memory placement.
- Estimated DRAM GB/s and percent of architectural peak.
- Estimated TFLOPs and percent of peak for the selected math fidelity.
- `DRAM`, `FLOP`, `BOTH`, `SLOW`, or `HOST` bottleneck labels.
- Generic optimization advice and grouped CSV/PNG summaries.

For multi-device traces, understand its merge rule before comparing totals: corresponding non-collective operations use the maximum device duration, while AllGather, ReduceScatter, and AllReduce use the mean device duration. The local analyzer instead keeps per-device totals and reports the slowest device. These views answer related but different questions.

## Generated logs, profiles, and reports

Tracy produces three layers of artifacts:

### Raw Tracy capture

- `tracy_profile_log_host.tracy`: native Tracy timeline containing host zones, messages, signposts, threads, and timing. Open it in the Tracy GUI for interactive timeline analysis.

### Intermediate host and device logs

Normally stored under `generated/profiler/.logs/`:

- `tracy_ops_times.csv`: host operation and zone timing exported from the native capture.
- `tracy_ops_data.csv`: host-side TTNN operation metadata used to construct the merged report.
- `cpp_device_perf_report.csv`: current C++-processed device events and operation correlation.
- `profile_log_device.csv`: raw/legacy device-profiler events used for low-level debugging or legacy processing.
- `zone_src_locations.log` and related files: source-location mappings for Tracy zones.

These working logs can be overwritten or regenerated by later captures. Archive them before starting another experiment when they matter.

### Timestamped report directory

Stored under `generated/profiler/reports/<timestamp>/`:

- `ops_perf_results_<timestamp>.csv`: the primary merged operation report containing host timing, device timing, tensor metadata, kernel paths, program hashes, trace IDs, and performance-model columns.
- `profile_log_device.csv`: retained device-profile data associated with the report.
- `npe_viz/`: optional NoC/NPE timelines when `--collect-noc-traces` is used.

The operation CSV is the normal starting point. Return to the native Tracy timeline or intermediate logs when merged operation rows cannot explain a gap.

If the capture command contains `-r`, report processing happens automatically. Running `process_ops_logs.py` afterward produces another report from the same raw logs, not another measurement.

## Gemma example capture

### Gemma 4 31B traced prefill

File: `captures/gemma4-31b-prefill-trace-1x4.csv`

- 37,198 report rows: 37,196 device operations and two signposts.
- Four devices with 9,299 operation rows each.
- Three trace-associated sessions with 7,220 rows each.
- The final measured replay is session 3.
- Signposted end-to-end latency: 96.984 ms.
- Session 3 has 1,805 rows per device.
- Per-device summed kernel durations range from approximately 80.01 to 80.81 ms.

Where those numbers came from:

1. **37,198 rows:** count all report rows after the header: 37,196 device operations and two signposts.
2. **96.984 ms:** subtract the `start` signpost's `HOST START TS` from the `stop` signpost's timestamp and divide nanoseconds by 1,000,000.
3. **Session 3:** choose the highest replay-session ID enclosed by the final measured start/stop sequence.
4. **80.01–80.81 ms/device:** filter to session 3 and sum `DEVICE KERNEL DURATION [ns]` independently for each `DEVICE ID`.
5. **Operation contribution:** group session-3 rows by `OP CODE`, sum independently per device, and use the maximum device total.
6. **Percentage share:** divide each operation's maximum per-device sum by the slowest device's 80.814 ms total.

Maximum per-device contribution estimates for session 3:

- Matmul: 38.538 ms, 47.7%.
- LayerNorm: 19.690 ms, 24.4%.
- ReduceScatter: 5.701 ms, 7.1%.
- AllGather: 4.566 ms, 5.6%.
- NlpCreateHeads: 3.314 ms, 4.1%.
- Binary operations: 2.569 ms, 3.2%.
- Rotary embedding: 2.473 ms, 3.1%.
- SDPA: 2.245 ms, 2.8%.
- NLPConcatHeads: 1.879 ms, 2.3%.

### What `tt-perf-report` adds to the Gemma capture

The archived outputs and exact commands are in `tt_perf_report/README.md`.

Using the full `start`/`stop` signpost window:

- 7,248 raw device rows become 1,812 logical rows after four-device merging.
- The window contains 7,220 session-3 rows plus 28 device rows without a replay-session ID.
- Merged device time is 82.986 ms and op-to-op gaps total 12.564 ms.
- Device time plus gaps is 95.550 ms, leaving only about 1.434 ms between that decomposition and the 96.984 ms signposted wall time.
- The largest gap is 11.403 ms immediately before the first embedding operation. The embedding kernel itself is only 8.307 µs, so the gap is replay launch/orchestration rather than embedding compute.
- Compute accounts for 81.88% of merged device time, data movement 11.14%, tensor manipulation 6.65%, and unclassified work 0.32%.
- Matmul is 40.307 ms or 48.57%. All 301 merged matmul rows are labeled DRAM-bound; weighted FLOP utilization is 21.17%.
- The `128 x 5376 x 5376` matmul contributes 26.210 ms across 180 calls, approximately 65% of all matmul time.
- Full-window per-device totals span 82.211–83.056 ms, a 0.845 ms or approximately 1.03% spread.
- `PagedFillCacheDeviceOperation` is unclassified in version 1.2.8. Its 0.32% timing is retained under `Other`; only category labeling is affected.

These totals differ from the session-3 figures above because `tt-perf-report` includes the entire signpost window and merges corresponding operations across devices. Use session filtering for the replay's per-device critical path; use the signpost-window report for gap decomposition, roofline metrics, and tuning leads.

Possible things to look into for the Gemma example:

1. Test whether DRAM-sharded program configurations improve the dominant `128 x 5376 x 5376` matmul shape.
2. Inspect the 261 four-core LayerNorm rows, which total 18.658 ms, before assuming that a wider core grid will improve them.
3. Inspect whether the 10.267 ms combined ReduceScatter and AllGather cost is balanced, serialized, or overlap-capable.
4. Compare per-core minimum, maximum, and average duration on the slowest individual kernels.
5. Investigate whether the 11.403 ms replay-start gap can be reduced, then account for the remaining approximately 1.434 ms outside the report's device-plus-gap decomposition.
6. Verify that any kernel-level improvement lowers the 96.984 ms signposted replay and preserves correctness/PCC.

## What to think about when profiling

### Define the question

- Are you measuring end-to-end latency, an individual operation, host overhead, device imbalance, communication, memory behavior, or a regression?
- Does the question require eager operation detail or Metal Trace replay timing?
- Which metric will determine whether the experiment succeeded?

### Control the workload

- Record the model, checkpoint, topology, layers, batch, sequence length, dtype, fidelity, and software revision.
- Keep weight loading, compilation, and warm-up outside the measured region.
- Use the same shapes and cache state before and after a change.
- Consider whether a long run needs mid-run profiler-buffer draining.

### Validate the report

- Confirm the test passed and expected devices and operations appear.
- Confirm signposts or replay-session IDs identify the intended pass.
- Watch for profiler-buffer overflow warnings and missing operation classes.
- Treat devices as concurrent: compare them separately and use the slowest device for the critical path.

### Interpret carefully

- Device kernel duration, firmware duration, host duration, and signposted wall time answer different questions.
- A large aggregate operation category should be split by shape and configuration before tuning.
- Forced synchronization and detailed instrumentation can perturb normal asynchronous execution.
- Trace replay rows can retain capture timestamps; select by replay session or CSV ordering.
- Program-cache metadata on trace-replay rows does not mean the replay recompiled programs.

### Compare responsibly

- Change one variable at a time.
- Use the same test, topology, profiler mode, and measurement window.
- Repeat measurements and examine variation rather than trusting one sample.
- Re-run correctness/PCC after precision, sharding, memory, or program-configuration changes.
- Confirm that a local kernel improvement reduces signposted and unprofiled end-to-end latency.

## Common warnings

### Permission denied copying the Tracy UI trace

If the test and CSV generation succeed, this warning affects only copying the `.tracy` capture into the browser UI directory:

```text
Could not copy .../tracy_profile_log_host.tracy to build/profiler/build_wasm/traces/: Permission denied
```

Fix ownership only if the interactive Tracy UI is needed. The generated operation CSV remains valid.

### Pandas `DtypeWarning`

Mixed types in host-log columns can trigger a `DtypeWarning` during report generation. If the report completes successfully, this warning alone does not invalidate it.

### Deprecated collective arguments

Warnings about deprecated `all_gather` arguments indicate future API cleanup. They do not invalidate current timing results.

## Session maintenance rule

For the remainder of this Tracy prefill-learning session:

1. Copy every CSV or supporting log we analyze into `tracy_guide_docs/` first.
2. Give it a descriptive, stable filename.
3. Record its source, command, topology, model, mode, and conclusion here.
4. Update commands and interpretation guidance when we learn something new.
5. Keep raw captures unchanged; add derived scripts or summaries separately.

# TT-Metal Profiling Guide

This guide covers profiling TT-Metal applications using Tracy and the device profiler. It explains how to capture host and device profiling data, understand the output, and work around common limitations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Tracy GUI Installation](#tracy-gui-installation)
3. [Profiling Modes](#profiling-modes)
4. [The Tracy Python Module](#the-tracy-python-module)
5. [Environment Variables Reference](#environment-variables-reference)
6. [Capturing More Than 1000 Operations](#capturing-more-than-1000-operations)
7. [Host-Device Synchronization](#host-device-synchronization)
8. [Understanding Profiler Output](#understanding-profiler-output)
9. [Common Issues and Solutions](#common-issues-and-solutions)
10. [Example Workflows](#example-workflows)
11. [Instrumenting Your Own Code](#instrumenting-your-own-code)
    - [Python Instrumentation](#python-instrumentation)
    - [C++ Host Code Instrumentation](#c-host-code-instrumentation)
    - [Device Kernel Instrumentation](#device-kernel-instrumentation)

---

## Quick Start

```bash
# 1. Build with Tracy enabled (default)
./build_metal.sh

# 2. Run your workload with profiling
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v your_script.py

# 3. Find the report in generated/profiler/reports/<timestamp>/
```

For pytest:
```bash
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest path/to/test.py::test_function
```

---

## Tracy GUI Installation

### macOS (Homebrew)

```bash
# Remove old version if exists
brew uninstall tracy

# Create a custom tap and install TT's fork
brew tap-new $USER/tracy
wget -P $(brew --repository)/Library/Taps/$USER/homebrew-tracy/Formula/ \
  --no-check-certificate --no-cache --no-cookies \
  https://raw.githubusercontent.com/tenstorrent-metal/tracy/master/tracy.rb
brew install $USER/tracy/tracy

# Start the GUI
tracy
```

### Linux (Build from Source)

```bash
git clone https://github.com/tenstorrent/tracy.git
cd tracy/profiler/build/unix
make -j8

# Run the profiler
./Tracy-release
```

### Connecting to Remote Machine

The Tracy GUI acts as a **server** (counterintuitively). Your profiled application connects to it as a client.

For remote profiling, use SSH port forwarding:
```bash
# On your local machine (where Tracy GUI runs)
ssh -NL 8086:127.0.0.1:8086 user@remote-machine
```

Then in Tracy GUI, connect to `127.0.0.1:8086`.

---

## Profiling Modes

### Host-Only Profiling
Profile Python and C++ code on the host:
```bash
python -m tracy -p your_script.py
```

### Device Profiling
Profile device-side kernels:
```bash
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r your_script.py
```

### Combined Host + Device Profiling
Get synchronized host and device traces:
```bash
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r --sync-host-device your_script.py
```

**Important**: Use `--sync-host-device` to align host and device timelines. Without this flag, host and device events may appear misaligned in the Tracy GUI.

---

## The Tracy Python Module

### Basic Syntax

```bash
python -m tracy [OPTIONS] [-m module | scriptfile] [args...]
```

### Key Options

| Option | Description |
|--------|-------------|
| `-p`, `--partial` | Only profile enabled zones (recommended for cleaner traces) |
| `-r`, `--report` | Generate ops report with CSV and .tracy file |
| `-v`, `--verbose` | Print more info to stdout (helpful for debugging capture issues) |
| `-l`, `--lines` | Profile every line of Python code (expensive, use with `-p`) |
| `-m` | Profile a Python module (e.g., pytest) |
| `--no-device` | Exclude device data from the profile |
| `-o`, `--output-folder` | Custom output folder for profiler artifacts |
| `-n`, `--name-append` | Custom name to append to report name |

### Advanced Options

| Option | Description |
|--------|-------------|
| `--no-op-info-cache` | Show full op info for cached ops (use to see kernel paths for all ops) |
| `--op-support-count N` | Maximum number of ops supported by profiler (default: 1000) |
| `--sync-host-device` | Sync host with all devices for aligned timelines |
| `--profile-dispatch-cores` | Collect dispatch cores profiling data |
| `--dump-device-data-mid-run` | Dump device data to files and push to Tracy during execution |
| `--collect-noc-traces` | Collect NoC event traces when profiling |
| `--device-memory-profiler` | Profile allocated L1 and DRAM memory buffers |
| `--profiler-capture-perf-counters` | Capture perf counters: `fpu,pack,unpack,l1,instrn,all` |

### Examples

```bash
# Profile a pytest with verbose output and report generation
python -m tracy -p -r -v -m pytest models/demos/bert_tiny/demo/demo.py::test_demo

# Profile with host-device sync for aligned timelines
python -m tracy -p -r -v --sync-host-device your_script.py

# Profile with full op info (no caching of op names)
python -m tracy -p -r -v --no-op-info-cache your_script.py

# Profile with increased op limit
python -m tracy -p -r -v --op-support-count 5000 your_script.py

# Profile with NoC traces for network analysis
python -m tracy -p -r -v --collect-noc-traces your_script.py
```

---

## Environment Variables Reference

### Core Profiling Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TT_METAL_DEVICE_PROFILER` | 0 | **Required** - Enable device-side profiling |
| `TT_METAL_PROFILER_SYNC` | 0 | Enable synchronous profiling for more accurate timing |
| `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` | 1000 | Max number of programs (ops) the profiler can capture |
| `TT_METAL_PROFILER_MID_RUN_DUMP` | 0 | Force mid-run profiler dumps |
| `TT_METAL_PROFILER_CPP_POST_PROCESS` | 0 | Enable C++ post-processing for profiler |

### Advanced Profiling Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TT_METAL_DEVICE_PROFILER_DISPATCH` | 0 | Profile dispatch cores |
| `TT_METAL_DEVICE_PROFILER_NOC_EVENTS` | 0 | Enable NoC events profiling |
| `TT_METAL_TRACE_PROFILER` | 0 | Enable trace profiling |
| `TT_METAL_MEM_PROFILER` | 0 | Enable memory/buffer profiling |
| `TT_METAL_PROFILER_SUM` | 0 | Enable sum profiling |
| `TT_METAL_PROFILER_NO_CACHE_OP_INFO` | 0 | Show full op info for cached ops |

### Output Control Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TT_METAL_PROFILER_DIR` | `generated/profiler` | Output directory for profiler artifacts |
| `TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES` | 0 | Disable dumping device data to files |
| `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY` | 0 | Disable pushing device data to Tracy GUI |

### Example Usage

```bash
# Basic profiling
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r your_script.py

# With increased op limit
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000 \
python -m tracy -p -r your_script.py

# With host-device sync and mid-run dumps
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_SYNC=1 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
python -m tracy -p -r your_script.py
```

---

## Capturing More Than 1000 Operations

By default, the device profiler can capture **1000 programs (operations)** before buffers fill up. When buffers are full, markers are dropped and you'll see warnings like:

```
Profiler DRAM buffers were full, markers were dropped! device X, worker core Y, Z...
Please either decrease the number of ops being profiled or run read device profiler more often
```

### Solution 1: Increase Program Support Count

```bash
# Via command line
python -m tracy -p -r --op-support-count 5000 your_script.py

# Via environment variable
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000 python -m tracy -p -r your_script.py
```

### Solution 2: Read Profiler Results Mid-Run

For very long runs, call `ReadDeviceProfiler()` periodically in your Python code:

```python
import ttnn

# Run some operations...
for batch in range(num_batches):
    result = model(input_batch)

    # Read profiler every N batches to avoid buffer overflow
    if batch % 100 == 0:
        ttnn.ReadDeviceProfiler(mesh_device)
```

In C++:
```cpp
#include "tt_metal/impl/profiler/tt_metal_profiler.hpp"

// After running operations
tt::tt_metal::ReadMeshDeviceProfilerResults(*mesh_device);
```

### Solution 3: Use Mid-Run Dump Mode

Enable automatic mid-run dumps:

```bash
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
python -m tracy -p -r your_script.py
```

Or use the CLI flag:
```bash
python -m tracy -p -r --dump-device-data-mid-run your_script.py
```

### Buffer Limits

- **L1 Buffer**: 250 markers per RISC per program
- **DRAM Buffer**: Dynamically sized based on `PROFILER_PROGRAM_SUPPORT_COUNT`
- **Default Program Support**: 1000 programs

---

## Host-Device Synchronization

### The Problem

Host and device have different clocks. Without synchronization, host events and device events appear misaligned in Tracy.

### The Solution

Use `--sync-host-device` to perform clock synchronization:

```bash
python -m tracy -p -r -v --sync-host-device your_script.py
```

This performs a linear regression on 249 time samples to calculate the frequency offset and delay between host and device clocks.

### What You'll See

**Without sync**: Host operations and device kernels appear at completely different times, making it impossible to correlate them.

**With sync**: Host operations align with their corresponding device kernels on the timeline.

---

## Understanding Profiler Output

### Output Location

After running with `-r`, find results in:
```
${TT_METAL_HOME}/generated/profiler/reports/<timestamp>/
```

### Output Files

| File | Description |
|------|-------------|
| `ops_perf_results_<timestamp>.csv` | Main performance report with one row per operation |
| `profile_log_device.csv` | Raw device-side profiling data |
| `tracy_profile_log_host.tracy` | Host-side Tracy profiler log |
| `tracy_ops_times.csv` | Host-side operation timing from Tracy |
| `tracy_ops_data.csv` | Host-side operation data/messages |

### Key CSV Columns

| Column | Description |
|--------|-------------|
| `OP CODE` | Operation name |
| `OP TYPE` | Where the op ran: `python_fallback`, `tt_dnn_cpu`, `tt_dnn_device` |
| `DEVICE FW DURATION [ns]` | Firmware duration on device |
| `DEVICE KERNEL DURATION [ns]` | Kernel duration on device |
| `HOST DURATION [ns]` | Total host-side duration |
| `CORE COUNT` | Number of cores used |
| `COMPUTE KERNEL PATH` | Path to compute kernels |
| `DATAMOVEMENT KERNEL PATH` | Path to data movement kernels |

### Tracy GUI Navigation

1. **Timeline View**: Shows execution over time
2. **Zones**: Color-coded sections representing profiled code regions
3. **Statistics**: Aggregated timing information
4. **Messages**: `TT_SIGNPOST` markers and other messages

---

## Common Issues and Solutions

### Issue: "No profiling data could be captured"

**Cause**: Tracy tools not found or not connected.

**Solution**:
```bash
# Verify Tracy is enabled in build
cmake . -DENABLE_TRACY=ON
ninja install

# Check Tracy tools exist
ls build/tools/profiler/bin/
# Should see: capture-release, csvexport-release
```

### Issue: Host and Device Not Aligned

**Cause**: Clock synchronization not performed.

**Solution**:
```bash
python -m tracy -p -r --sync-host-device your_script.py
```

### Issue: Only Seeing Device Data, No Host Markers

**Cause**: Using capture without Tracy module wrapper.

**Solution**: Always use `python -m tracy` wrapper:
```bash
# Wrong
TT_METAL_DEVICE_PROFILER=1 python your_script.py

# Right
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r your_script.py
```

### Issue: Cached Ops Don't Show Full Kernel Paths

**Cause**: Op info caching for performance.

**Solution**:
```bash
python -m tracy -p -r --no-op-info-cache your_script.py
```

### Issue: Only One Training Step Captured

**Cause**: Often due to loading from a checkpoint and profiler buffer limits.

**Solution**:
1. Start from fresh training (no checkpoint)
2. Increase program support count
3. Call `ReadDeviceProfiler()` after each epoch

### Issue: DPRINT and Profiler Conflict

**Cause**: Both use the same SRAM space.

**Solution**: Don't use both simultaneously:
```bash
# Unset DPRINT when profiling
unset TT_METAL_DPRINT_CORES
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r your_script.py
```

### Issue: Timer Skew on Grayskull After Reset

**Cause**: Tensix cores have skewed timer starts after `tt_smi` reset.

**Solution**: Full reboot required on Grayskull:
```bash
sudo reboot
```
Note: Wormhole does not have this issue.

---

## Example Workflows

### Workflow 1: Basic Model Profiling

```bash
# Profile a model demo
TT_METAL_DEVICE_PROFILER=1 \
python -m tracy -p -r -v \
  -m pytest models/demos/bert_tiny/demo/demo.py::test_demo

# View results
ls generated/profiler/reports/
# Open the .tracy file in Tracy GUI or analyze the CSV
```

### Workflow 2: Training Profiling with tt-train

```bash
# Set environment and run profiler
env -u TT_METAL_DPRINT_CORES \
TT_METAL_DEVICE_PROFILER=1 \
python -m tracy -r -v -p \
  ${TT_METAL_HOME}/build/tt-train/sources/examples/nano_gpt/nano_gpt

# Analyze results with Jupyter notebook
jupyter lab tt-train/notebooks/profiler_results.ipynb
```

### Workflow 3: Long-Running Workload

```python
import ttnn

mesh_device = ttnn.open_mesh_device(...)

for epoch in range(100):
    for batch in dataloader:
        loss = train_step(batch)

    # Read profiler after each epoch to avoid buffer overflow
    ttnn.ReadDeviceProfiler(mesh_device)
    print(f"Epoch {epoch} complete, profiler data captured")

mesh_device.close()
```

Run with:
```bash
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
python -m tracy -p -r your_training_script.py
```

### Workflow 4: Analyzing Specific Operations

```python
from tracy import signpost

# Add markers in your code
signpost("Warmup Start")
for _ in range(5):
    model(input_tensor)

signpost("Measurement Start", "Beginning timed runs")
for _ in range(10):
    model(input_tensor)
signpost("Measurement End")
```

These signposts appear in both the CSV report and Tracy GUI for easy navigation.

### Workflow 5: Copying Tracy File for Local Analysis

If you can't use port forwarding:

```bash
# On remote machine - generate the .tracy file
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r your_script.py

# Copy to local machine
scp user@remote:tt-metal/generated/profiler/reports/*/tracy_profile_log_host.tracy ./

# Open in local Tracy GUI
tracy
# Then File -> Open and select the .tracy file
```

---

## Instrumenting Your Own Code

This section covers how to add custom profiling zones and signposts to your code at different levels: Python, C++ host code, and device kernels.

### Python Instrumentation

#### Signposts (Markers)

Signposts are the simplest way to add markers to your Python code. They appear in both the CSV report and Tracy GUI.

```python
from tracy import signpost

# Simple signpost with just a header
signpost("Warmup Start")

for i in range(5):
    model(input_tensor)

signpost("Warmup End")

# Signpost with header and detailed message
signpost("Measurement Phase", "Running 100 iterations for benchmarking")

for i in range(100):
    model(input_tensor)

signpost("Measurement Complete", f"Processed {100} iterations")
```

**Use cases for signposts:**
- Marking phases of execution (warmup, measurement, cooldown)
- Identifying specific iterations or batches
- Adding context to profiling data
- Creating navigation points in long traces

#### Profiler Context Manager

For profiling specific code sections:

```python
from tracy import Profiler

profiler = Profiler()

# Profile a specific section
profiler.enable()
result = my_expensive_function()
profiler.disable()
```

**Important**: Call `enable()` and `disable()` outside the function being profiled, not inside it. This ensures proper event capture.

#### Tracy Messages

For sending custom messages to Tracy (visible in the Messages panel):

```python
import ttnn

# Send a simple message
ttnn.tracy_message("Starting batch processing")

# Send a message with custom color (RGB as integer)
ttnn.tracy_message("Critical section entered", color=0xFF0000)  # Red
```

#### Tracy Zones (Manual)

For more control over zone creation:

```python
import ttnn

# Start a zone
ttnn.start_tracy_zone("my_file.py", "MY_CUSTOM_ZONE", line_number=42)

# ... your code ...

# Stop the zone (with optional name and color)
ttnn.stop_tracy_zone("MY_CUSTOM_ZONE", color=0x00FF00)  # Green
```

### C++ Host Code Instrumentation

Include the Tracy header and use the provided macros:

```cpp
#include <tracy/Tracy.hpp>
```

#### Basic Zone (Automatic Naming)

```cpp
void my_function() {
    ZoneScoped;  // Zone named after the function
    // ... your code ...
}
```

#### Named Zones

```cpp
void complex_function() {
    {
        ZoneScopedN("Phase1_Setup");
        // Setup code
    }

    {
        ZoneScopedN("Phase2_Compute");
        // Computation code
    }

    {
        ZoneScopedN("Phase3_Cleanup");
        // Cleanup code
    }
}
```

#### Zones with Colors

```cpp
void my_function() {
    ZoneScopedN("MyZone");
    ZoneColor(0x00FF00);  // Green
    // ... your code ...
}
```

#### Zones with Text

```cpp
void process_batch(int batch_id) {
    ZoneScopedN("ProcessBatch");
    std::string info = fmt::format("batch_id: {}", batch_id);
    ZoneText(info.c_str(), info.size());
    // ... your code ...
}
```

#### Tracy Messages (C++)

```cpp
#include <tracy/Tracy.hpp>

void my_function() {
    TracyMessageL("Starting important operation");

    // With dynamic string
    std::string msg = fmt::format("Processing {} items", count);
    TracyMessage(msg.c_str(), msg.size());

    // ... your code ...
}
```

#### Frame Marks

For marking logical frames (useful for game-loop style applications):

```cpp
void main_loop() {
    while (running) {
        FrameMarkNamed("main");  // Mark the start of each frame

        update();
        render();
    }
}
```

### Device Kernel Instrumentation

Device kernels run on Tensix cores and require special macros.

#### Include the Header

```cpp
#include <tools/profiler/kernel_profiler.hpp>
```

#### Basic Device Zone

```cpp
void kernel_main() {
    DeviceZoneScopedN("MyKernelOperation");
    // ... kernel code ...
}
```

#### Multiple Zones in a Kernel

```cpp
void kernel_main() {
    {
        DeviceZoneScopedN("LoadData");
        // Load data from DRAM/L1
    }

    {
        DeviceZoneScopedN("Compute");
        // Perform computation
    }

    {
        DeviceZoneScopedN("StoreData");
        // Store results
    }
}
```

#### Guaranteed Zones (Always Captured)

Regular zones may be dropped if buffers fill up. Use guaranteed zones for critical measurements:

```cpp
void kernel_main() {
    // Main zone - guaranteed to be captured (slot 0)
    DeviceZoneScopedMainN("KERNEL-MAIN");

    // ... your kernel code ...
}
```

For a child of the main zone:

```cpp
void kernel_main() {
    DeviceZoneScopedMainN("KERNEL-MAIN");

    {
        // Child zone - also guaranteed (slot 1)
        DeviceZoneScopedMainChildN("KERNEL-INNER-LOOP");
        // ... inner loop code ...
    }
}
```

#### Accumulating Zones (Sum Profiling)

For measuring total time across multiple invocations:

```cpp
void kernel_main() {
    for (int i = 0; i < ITERATIONS; i++) {
        {
            DeviceZoneScopedSumN1("AccumulatedCompute");
            // This measures total time across all iterations
        }
    }
}
```

Use `DeviceZoneScopedSumN1` and `DeviceZoneScopedSumN2` for up to two different accumulating zones.

**Note**: Requires `--enable-sum-profiling` or `TT_METAL_PROFILER_SUM=1`.

#### Timestamped Data

Record a timestamp with associated data (useful for tracking values over time):

```cpp
void kernel_main() {
    for (int i = 0; i < iterations; i++) {
        DeviceTimestampedData("IterationData", i);
        // ... process iteration i ...
    }
}
```

#### Recording Events

For simple event markers without scope:

```cpp
void kernel_main() {
    DeviceRecordEvent(1);  // Event ID 1

    // ... some work ...

    DeviceRecordEvent(2);  // Event ID 2
}
```

#### Validating Profiler Data

Mark profiler data as valid/invalid based on conditions:

```cpp
void kernel_main() {
    bool data_valid = check_preconditions();
    DeviceValidateProfiler(data_valid);

    if (data_valid) {
        // ... kernel work ...
    }
}
```

### Device Zone Limitations

- **L1 Buffer**: Each RISC can store ~250 zone markers per program
- **Guaranteed Markers**: Only 4 guaranteed marker slots available
- **Overhead**: Each `DeviceZoneScopedN` adds some overhead; use selectively
- **Name Length**: Zone names are hashed; keep them descriptive but not excessively long

### Complete Example: End-to-End Profiling

#### Python Test Script

```python
import ttnn
from tracy import signpost, Profiler

def run_model_with_profiling(mesh_device, model, inputs, num_warmup=5, num_iterations=100):
    profiler = Profiler()

    # Warmup phase
    signpost("Warmup Phase Start", f"Running {num_warmup} warmup iterations")
    for i in range(num_warmup):
        _ = model(inputs)
    ttnn.synchronize_device(mesh_device)
    signpost("Warmup Phase End")

    # Measurement phase with detailed profiling
    signpost("Measurement Phase Start", f"Running {num_iterations} measured iterations")
    profiler.enable()

    for i in range(num_iterations):
        if i % 10 == 0:
            signpost(f"Iteration {i}")
        result = model(inputs)

    ttnn.synchronize_device(mesh_device)
    profiler.disable()
    signpost("Measurement Phase End")

    # Read profiler data
    ttnn.ReadDeviceProfiler(mesh_device)

    return result
```

#### C++ Host Code

```cpp
#include <tracy/Tracy.hpp>
#include "tt_metal/host_api.hpp"

void run_inference(tt::tt_metal::IDevice* device, Program& program) {
    ZoneScopedN("RunInference");

    {
        ZoneScopedN("EnqueueProgram");
        EnqueueProgram(device->command_queue(), program, false);
    }

    {
        ZoneScopedN("WaitForCompletion");
        Finish(device->command_queue());
    }
}
```

#### Device Kernel

```cpp
#include <tools/profiler/kernel_profiler.hpp>
#include "dataflow_api.h"

void kernel_main() {
    DeviceZoneScopedMainN("DATAFLOW-KERNEL");

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    {
        DeviceZoneScopedN("ReadFromDRAM");
        for (uint32_t i = 0; i < num_tiles; i++) {
            noc_async_read_tile(i, src_addr, cb_id);
            noc_async_read_barrier();
        }
    }

    {
        DeviceZoneScopedN("WriteToDRAM");
        for (uint32_t i = 0; i < num_tiles; i++) {
            noc_async_write_tile(i, dst_addr, cb_id);
            noc_async_write_barrier();
        }
    }
}
```

---

## Performance Impact

Profiling adds overhead:
- **Host profiling**: Minimal (~1-5%)
- **Device profiling**: Moderate (~5-15% depending on zone density)
- **Line-level profiling** (`-l`): Significant (use only when needed)

For production benchmarks, run without profiling to get true performance numbers.

---

## Further Reading

- [Tracy Documentation](https://github.com/wolfpld/tracy/releases/latest/download/tracy.pdf)
- [Device Program Profiler](device_program_profiler.rst)
- [Tracy Profiler](tracy_profiler.rst)
- [TT-NN Visualizer](https://docs.tenstorrent.com/ttnn-visualizer/)

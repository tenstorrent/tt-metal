# Memory Tracking Tools

## Memory Usage Tracker (`memory_utils.hpp`)

### Core Concept

The memory tracker captures graph-level memory operations across training phases. It works by:
1. Beginning a capture session via `begin_capture()`
2. Taking snapshots at key training checkpoints via `snapshot(name)`
3. Ending capture and printing results via `print_memory_usage()`

Each snapshot records:
- **DRAM allocations/deallocations** since last snapshot
- **Segment Peak**: max memory usage during this phase (relative to phase start)
- **Cumulative values**: running totals accounting for all previous phases

### Usage

#### C++

```cpp
#include "ttml/utils/memory_utils.hpp"

// Begin capture (returns RAII guard that auto-cleans on scope exit)
ttnn::ScopeGuard guard = ttml::utils::MemoryUsageTracker::begin_capture();

// ... model creation code ...
ttml::utils::MemoryUsageTracker::snapshot("MODEL_CREATION");

// ... optimizer creation code ...
ttml::utils::MemoryUsageTracker::snapshot("OPTIMIZER_CREATION");

// ... training iteration ...
ttml::utils::MemoryUsageTracker::snapshot("FORWARD_PASS");
ttml::utils::MemoryUsageTracker::snapshot("BACKWARD_PASS");

// Print and cleanup
ttml::utils::MemoryUsageTracker::end_capture("ITERATION_COMPLETE");
ttml::utils::MemoryUsageTracker::print_memory_usage();
ttml::utils::MemoryUsageTracker::clear(); // Not required, since guard will clear anyway.
                                          // You might want to clear earlier to remove traces from memory
```

#### Python

```python
import ttml

# Access MemoryUsageTracker from ttml.core.utils
MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker

# Begin capture (returns guard that auto-cleans when destroyed)
guard = MemoryUsageTracker.begin_capture()

# ... model creation code ...
MemoryUsageTracker.snapshot("MODEL_CREATION")

# ... optimizer creation code ...
MemoryUsageTracker.snapshot("OPTIMIZER_CREATION")

# ... training iteration ...
MemoryUsageTracker.snapshot("FORWARD_PASS")
MemoryUsageTracker.snapshot("BACKWARD_PASS")

# Print and cleanup
MemoryUsageTracker.end_capture("ITERATION_COMPLETE")
MemoryUsageTracker.print_memory_usage()
MemoryUsageTracker.clear()
guard.release()  # Prevent double cleanup when guard is garbage collected
```

See [train_nanogpt.py](/tt-train/sources/examples/nano_gpt/train_nanogpt.py) with `--track_memory` flag for a complete example.

### NO_DISPATCH Mode

To measure memory for models that don't fit in device memory, use NO_DISPATCH mode:

**C++:**
```cpp
ttnn::ScopeGuard guard = ttml::utils::MemoryUsageTracker::begin_capture(
    tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH
);
```

**Python:**
```python
import ttnn

guard = MemoryUsageTracker.begin_capture(ttnn.graph.RunMode.NO_DISPATCH)
```

This mode records allocations without actually allocating memory and executing operations on device. Note that training loss is messed up after running first iteration in NO_DISPATCH, which is expected, since none of ttnn operations were actually dispatched on the device.

### Output Format

```
--- PHASE_NAME ---
  DRAM: Segment Peak X MB, Allocations Y MB, Deallocations Z MB, Segment Change ±W MB
  DRAM: Cumulative Peak A MB, Cumulative Current B MB
  L1: Peak CB C MB, Peak Buffer D MB, Peak Total E MB
```

**Field Definitions:**

| Field | Meaning |
|-------|---------|
| **Segment Peak** | Maximum memory usage during this phase (relative to phase start) |
| **Allocations** | Total memory allocated in this phase |
| **Deallocations** | Total memory freed in this phase |
| **Segment Change** | Net change: Allocations - Deallocations |
| **Cumulative Peak** | Maximum total memory across all phases so far |
| **Cumulative Current** | Total memory in use after this phase |
| **Peak CB** | Max circular buffer usage in L1 |
| **Peak Buffer** | Max buffer usage in L1 |
| **Peak Total** | Max total L1 usage |

**Key Relations:**
- `Segment Change = Allocations - Deallocations`
- `Cumulative Current[N] = Cumulative Current[N-1] + Segment Change[N]`
- `Cumulative Peak[N] = max(Cumulative Peak[N-1], Cumulative Current[N-1] + Segment Peak[N])`

## Example: NanoGPT Training Loop

From `tt-train/sources/examples/nano_gpt/main.cpp`:

```cpp
// Line 327: Begin capture
ttnn::ScopeGuard memory_usage_guard =
    ttml::utils::MemoryUsageTracker::begin_capture(RunMode::NO_DISPATCH);

// Line 553: After model creation
ttml::utils::MemoryUsageTracker::snapshot("MODEL_CREATION");

// Line 638: After optimizer creation
ttml::utils::MemoryUsageTracker::snapshot("OPTIMIZER_CREATION");

// Lines 697-703: During training iteration
memory_snapshot("FORWARD_PASS");
loss->backward();
memory_snapshot("BACKWARD_PASS");

// Lines 754-756: Print results
ttml::utils::MemoryUsageTracker::end_capture("FIRST_ITERATION_COMPLETE");
ttml::utils::MemoryUsageTracker::print_memory_usage();
```

### TinyLlama Results Analysis

**Memory-Efficient Mode (with activation recomputation):**
```
--- MODEL_CREATION ---
  Segment Change: +1863.25 MB  # Model weights (bf16)
  Cumulative Current: 1863.25 MB

--- OPTIMIZER_CREATION ---
  Segment Change: +3708.76 MB  # Optimizer state (2x model weights)
  Cumulative Current: 5572.01 MB

--- FORWARD_PASS ---
  Segment Peak: 996.35 MB
  Segment Change: +192.95 MB   # Activations kept (reduced via recomputation)
  Cumulative Peak: 6568.36 MB
  Cumulative Current: 5764.96 MB

--- BACKWARD_PASS ---
  Segment Peak: 2513.69 MB     # Peak during gradient computation
  Segment Change: +1663.07 MB  # Gradients
  Cumulative Peak: 8278.65 MB  # Overall peak
  Cumulative Current: 7428.04 MB

Overall DRAM Peak: 8278.65 MB  # ✓ Fits in 12GB device memory
```

**Default Mode (no activation recomputation):**
```
--- FORWARD_PASS ---
  Segment Peak: 10019.39 MB
  Segment Change: +9636.02 MB  # All activations stored
  Cumulative Peak: 15591.27 MB
  Cumulative Current: 15207.89 MB

--- BACKWARD_PASS ---
  Segment Peak: 458.75 MB
  Segment Change: -7780.87 MB  # Net deallocation (activations freed)
  Cumulative Peak: 15666.65 MB

Overall DRAM Peak: 15666.65 MB  # ✗ Exceeds 12GB device memory
```

**Key Insight:** Memory-efficient mode trades ~9.4GB of activation storage for recomputation, reducing peak from 15.7GB to 8.3GB.

## Memory Analysis Script (`analyze_memory.py`)

### Usage

```bash
python scripts/analyze_memory.py --logs <logfile> [options]
```

**Required:**
- `--logs`: A single log file containing memory usage summaries created by training script ([cpp](/tt-train/sources/examples/nano_gpt/main.cpp) or [python](/tt-train/sources/examples/nano_gpt/train_nanogpt.py)). Note: you can concatenate multiple trainig logs in one file, it would analyze them separately, and add all of them on the histogram

**Optional:**
- `--device_memory`: Device memory in bytes (default: 12GB)
- `--model_size`: Theoretical model size in bytes (auto-calculated from logs if not provided)
- `--optimizer_size`: Optimizer size in bytes (default: 2 × model_size)
- `--gradients_size`: Gradients size in bytes (default: model_size)
- `--visualize_peak`: Generate histogram visualization
- `--use_actual_sizes`: Use measured values from logs instead of theoretical on histogram for model & optimizer sizes. If not provided, then difference is shown as "Other"
- `--title`: Title for visualization
- `--output`: Output filename (default: `memory_peak_visualization.png`)

**Example:**
```bash
python scripts/analyze_memory.py \
  --logs mem_example.log \
  --visualize_peak \
  --title "TinyLlama Memory Comparison"
```

### Value Calculations

#### Theoretical Sizes (default, assumes bf16):
- **Model Size**: `num_parameters × 2 bytes`
- **Optimizer State**: `2 × model_size` (AdamW stores first and second moments)
- **Gradients**: `model_size` (same dtype as model)

#### Extracted from Logs:
- **Actual Model Size**: `MODEL_CREATION Segment Change`
- **Actual Optimizer Size**: `OPTIMIZER_CREATION Segment Change`
- **Activations**: `FORWARD_PASS Segment Change`
- **Gradients (calculated)**: `BACKWARD_PASS Cumulative Current - OPTIMIZER_CREATION Cumulative Current`

#### Comparison Metrics:
- **Percentage Difference**: `((actual - expected) / expected) × 100%`
- **Device Usage**: `(Peak DRAM / Device Memory) × 100%`

### Histogram Breakdown

The visualization shows stacked bars with components (bottom to top):

1. **Other** (gray, #cccccc): Unaccounted memory = `Peak DRAM - (Model + Optimizer + Activations + Gradients Overhead)`

2. **Model Parameters** (blue, #4da6b8):
   - Default: Theoretical model size
   - With `--use_actual_sizes`: Actual from `MODEL_CREATION Segment Change`

3. **Gradients Overhead** (pink, #e789ab): `BACKWARD_PASS Segment Peak`
   - Peak memory spike during backward pass for gradient computation

4. **Optimizer State** (purple, #cec0fa):
   - Default: Theoretical optimizer size (2× model)
   - With `--use_actual_sizes`: Actual from `OPTIMIZER_CREATION Segment Change`

5. **Activations** (orange, #e38a42): `FORWARD_PASS Segment Change`
   - Memory retained from forward pass for backward computation

**Red Dashed Line**: Device memory limit (12GB default)

### Use Cases

**Compare configurations:**
```bash
# Generate histogram comparing memory-efficient vs default
python scripts/analyze_memory.py --logs mem_example.log --visualize_peak
```

**Analyze with theoretical values:**
```bash
# Use number of parameters to estimate sizes
python scripts/analyze_memory.py \
  --logs training.log \
  --model_size $(echo "969369600 * 2" | bc)  # 969M params × 2 bytes
```

**Debug memory discrepancies:**
```bash
# Use actual measured values to see true allocation patterns
python scripts/analyze_memory.py \
  --logs training.log \
  --visualize_peak \
  --use_actual_sizes # This means that actual model and optimizer sizes will be shown on the histogram
```

## Quick Reference

**Check if configuration fits in memory:**
```
Peak DRAM < Device Memory → Configuration viable
Peak DRAM > Device Memory → Need optimization or larger device
```

**Common optimizations:**
- Enable gradient checkpointing (a.k.a activations recomputation) (`memory_efficient` mode)
- Reduce batch size
- Use gradient accumulation (allows smaller per-iteration batch sizes)

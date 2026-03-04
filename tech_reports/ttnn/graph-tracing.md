# TT-NN Graph Tracing

TT-NN provides a mechanism for tracing operations and memory activities during neural network execution. This enables performance analysis, memory profiling, and visualization without modifying your code.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Basic Usage](#basic-usage)
- [Saving Reports](#saving-reports)
- [Advanced Features](#advanced-features)
  - [Reducing Capture Overhead](#reducing-capture-overhead)
- [Levelized Graph](#levelized-graph)
- [Testing & Validation](#testing--validation)
- [Reference: Node Types](#reference-node-types)
- [Reference: Sample Trace](#reference-sample-trace)

---

## Quick Start

### Python (5 lines)

```python
import ttnn

ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
result = ttnn.add(tensor_a, tensor_b)  # Your operations here
graph = ttnn.graph.end_graph_capture()
print(graph)  # List of traced nodes
```

### C++

```cpp
#include "ttnn/graph/graph_processor.hpp"

ttnn::graph::GraphProcessor::begin_graph_capture(ttnn::graph::RunMode::NORMAL);
auto result = ttnn::add(tensor_a, tensor_b);  // Your operations here
auto graph = ttnn::graph::GraphProcessor::end_graph_capture();
```

### Save to File (for ttnn-visualizer)

For a complete capture with all data (per-op sub-graphs, buffer pages, stack traces):

```python
import ttnn

with ttnn.manage_config("enable_fast_runtime_mode", False):
    ttnn.graph.enable_buffer_pages()
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    # ... your operations ...
    ttnn.graph.end_graph_capture_to_file("my_report.json")
    ttnn.graph.disable_buffer_pages()
```

Then import into the visualizer database:

```bash
python -m ttnn.graph_report my_report.json ./visualizer_db/
```

> **Note**: `enable_fast_runtime_mode=False` uses `Operation` (slow dispatch) which produces per-operation captured sub-graphs. The default `FastOperation` (fast dispatch) records arguments and tensor IDs but reconstructs sub-graphs synthetically. See [FastOperation vs Operation](#fastoperation-vs-operation) for details. Stack traces are enabled by default. To reduce overhead, see [Reducing Capture Overhead](#reducing-capture-overhead).

---

## Core Concepts

### What Gets Captured

The trace records:
- **Operations**: Function calls (`ttnn::add`, `ttnn::matmul`, etc.)
- **Tensors**: Shape, dtype, layout, memory location
- **Memory events**: Buffer allocations/deallocations, circular buffers
- **Timing**: Operation durations (wall-clock time)
- **Call hierarchy**: Nested operations (e.g., `ttnn::add` → `ttnn::prim::binary`)
- **Stack traces**: C++ call stacks for each operation (enabled by default)
- **Deallocations**: `Tensor::deallocate` is tracked at the C++ level when a device buffer is actually freed (both explicit via `ttnn.deallocate()` and implicit via destructor)
- **Python I/O**: Function arguments, input tensor IDs, and output tensor IDs recorded by the Python decorators

### Two-Phase Architecture

Graph capture is split into two phases:

1. **C++ runtime capture** — During model execution, the C++ `GraphProcessor` records an in-memory graph of operations, tensors, buffers, and timing. The Python decorators (`FastOperation`, `Operation`) additionally record Python-level arguments and tensor IDs into a `python_io` list. At the end, both are serialized to a JSON report file.

2. **Offline Python import** — A separate step reads the JSON report and imports it into a SQLite database that the ttnn-visualizer can consume. No database operations happen during model execution.

### How Operations Are Tracked

Operations are captured at two levels:

1. **Python-level**: The `ttnn` decorators (`FastOperation`, `Operation`) automatically inject `function_start`/`function_end` graph nodes when capture is active. This produces high-level operation names like `ttnn.conv2d`, `ttnn.linear`, `ttnn.deallocate`. Both decorators also record Python-visible arguments and tensor IDs into `python_io` records that are embedded in the JSON report.

2. **C++ level**: Internal C++ operations emit their own `function_start`/`function_end` nodes (e.g., `ttnn::matmul`, `Tensor::to_device`). These appear nested inside the Python-level operation.

The result is a hierarchical graph: `ttnn.conv2d` (Python) → `ttnn::conv2d` (C++) → `Device Operation` → `create_device_tensor`, etc.

**Deallocate tracking**: `Tensor::deallocate` emits tracking only when a device buffer is actually freed (not for host tensors or shared-reference tensors that skip deallocation). When called via Python `ttnn.deallocate()`, it appears nested: `ttnn.deallocate` → `Tensor::deallocate` → `buffer_deallocate`.

### FastOperation vs Operation

Both decorator classes participate in graph capture, but they differ in scope:

| Feature | `FastOperation` | `Operation` (slow dispatch) |
|---|---|---|
| `function_start`/`function_end` nodes | Yes | Yes |
| Python-level arguments in `python_io` | Yes | Yes |
| Input tensor ID tracking | Yes | Yes |
| Output tensor IDs (versioned, `force=True`) | Yes | Yes |
| Per-operation C++ captured sub-graph | No | Yes |

`FastOperation` is the default when `enable_fast_runtime_mode=True`. It records arguments and tensor IDs but does not wrap each operation in a nested `begin_graph_capture`/`end_graph_capture` call, so the per-operation captured sub-graph is synthetically reconstructed by the importer from the flat graph.

### Tensor Connectivity

The visualizer relies on tensor IDs to show data flow between operations:

```
op A  →  output tensor (id=42)  →  op B (consumes tensor id=42 as input)
```

Each `ttnn.Tensor` gets a unique `tensor_id` attribute assigned by `set_tensor_id()` in the decorators. Output tensors always get a fresh ID (`force=True`) so that in-place operations produce a distinct ID. The importer uses `python_io.input_tensor_ids` and `python_io.output_tensor_ids` to establish these connections in the `input_tensors` and `output_tensors` database tables.

### Run Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `NORMAL` | Real device execution with actual memory allocation | Production profiling, real addresses |
| `NO_DISPATCH` | Simulated execution, no device dispatch | Fast analysis, memory estimation |

```python
# Fast analysis without device
ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)

# Real execution with actual addresses
ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
```

**Note**: In `NO_DISPATCH` mode, you can trace unrealistically large tensors for memory estimation since no real allocation occurs.

---

## Basic Usage

### Extracting Operation Durations

```python
graph = ttnn.graph.end_graph_capture()

for node in graph:
    if node["node_type"] == "function_end" and "duration_ns" in node:
        name = node["params"]["name"]
        duration_ms = node["duration_ns"] / 1_000_000
        print(f"{name}: {duration_ms:.2f} ms")
```

### Tracking Memory Usage

```python
def get_peak_memory(graph):
    total_buffer = 0
    peak_buffer = 0

    for node in graph:
        if node["node_type"] == "buffer_allocate":
            total_buffer += int(node["params"]["size"])
            peak_buffer = max(peak_buffer, total_buffer)
        elif node["node_type"] == "buffer_deallocate":
            # Get size from connected buffer node
            buffer_id = node["connections"][0]
            buffer_node = graph[buffer_id]
            total_buffer -= int(buffer_node["params"]["size"])

    return peak_buffer
```

### Generating Visualizations

The trace can be visualized as a call graph:

![trace](https://github.com/user-attachments/assets/42501a1f-8354-4b3b-a5d9-707f30b23f4f)

Or as tabular data:

```
        current_op                           event  total_cb  total_buffer
0            ttnn::add                        begin_op         0       9011200
1         ttnn::repeat                        begin_op         0       9011200
2         ttnn::repeat                 buffer_allocate         0      17203200
...
```

---

## Saving Reports

### Save Complete Report to File

For offline analysis or ttnn-visualizer integration:

```python
ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
# ... your operations ...
ttnn.graph.end_graph_capture_to_file("my_report.json")
```

The report file contains:
- `version`: Report format version
- `graph`: Full graph trace (list of nodes)
- `devices`: Device information (architecture, grid size, memory)
- `metadata`: Capture timestamp
- `python_io`: Python-level I/O records (arguments, input/output tensor IDs per operation)
- `per_operation_buffers`: Real buffer snapshots per operation (from `get_buffers()`)
- `buffer_pages_by_address`: Compact per-address page data (when buffer pages are enabled)
- `cluster_descriptor`: Cluster configuration YAML (if available)
- `mesh_coordinate_mapping`: Physical chip mesh coordinate mapping YAML (if available, saved as `physical_chip_mesh_coordinate_mapping_1_of_1.yaml`)

### Import into Visualizer Database

```bash
# Import a single report
python -m ttnn.graph_report my_report.json ./visualizer_db/

# Import all reports from a directory
python -m ttnn.graph_report ./reports/ ./visualizer_db/

# Generate SVG visualizations during import
python -m ttnn.graph_report --svg my_report.json ./visualizer_db/
```

The importer creates these database tables:
- `operations`: Operations with names and durations (IDs start at 1)
- `operation_arguments`: Python-level arguments (from `python_io`), falling back to C++ arguments
- `tensors`: Shape, dtype, layout, memory_config, device_id, address, buffer_type
- `device_tensors`: Per-device addresses for multi-device tensors
- `input_tensors`, `output_tensors`: Tensor-to-operation relationships (from `python_io` tensor IDs)
- `buffers`: Cumulative memory allocation snapshots per operation (prefers `per_operation_buffers`)
- `buffer_pages`: Per-page buffer detail (when enabled)
- `captured_graph`: Per-operation subgraph JSON for the visualizer
- `stack_traces`: C++ call stacks (captured by default)
- `edges`: Graph edges extracted from captured subgraphs
- `errors`: Error information

#### Import Behavior

The importer produces output compatible with the ttnn-visualizer:

**Operation filtering**: The raw C++ graph contains many nested internal operations (`create_device_tensor`, `Device Operation`, etc.). The importer keeps only top-level operations that are meaningful to the user:
- Nested operations (those inside another `function_start`/`function_end` pair) are automatically filtered, with their inputs and outputs lifted to the parent
- Certain C++ wrapper operations are always treated as transparent (their children are still visible but they themselves don't become top-level operations):
  - `ttnn::convert_python_tensor_to_tt_tensor`
  - `tt::tt_metal::detail::convert_tt_tensor_to_framework_tensor`
  - `Tensor::deallocate`
  - `Tensor::to_device`
  - `Tensor::reshape`
  - `tt::tt_metal::to_dtype`
- Leading `ttnn.from_torch` operations (weight loading before any compute) are filtered; subsequent `from_torch` calls after the first real compute operation are kept

**Python I/O resolution**: When `python_io` records are present in the report, the importer uses them as the primary source for:
- Operation arguments (instead of C++ serialized arguments)
- Input tensor IDs (instead of heuristic extraction from C++ graph connections)
- Output tensor IDs (instead of C++ function_end connections)

The importer matches `python_io` records to graph nodes by operation name, consuming them in order.

**Tensor lifting**: When nested operations are filtered, their input/output tensor associations are "lifted" to the parent operation. For example, `ttnn.conv2d` might internally call `ttnn::matmul` which produces an output tensor — that tensor is attributed to `ttnn.conv2d` in the database. Internal tensors (produced and consumed within the same parent scope) are excluded.

**Tensor reconciliation**: Python decorators may assign fresh tensor IDs (via `set_tensor_id(force=True)`) that don't correspond to any C++ graph tensor node. The importer reconciles these by:
1. Looking up the memory address of the C++ tensor and cloning its properties
2. Falling back to the `pyid_to_cpp_tensor` mapping (built during import) to associate Python output IDs with their C++ counterpart nodes

**Deallocate synthesis**: If a `buffer_deallocate` event occurs outside any `function_start`/`function_end` pair, the importer synthesizes a `ttnn::deallocate` operation for it.

**Tensor deduplication**:
- Device tensors are deduplicated by address
- Host tensors are kept only when referenced as operation inputs or outputs

**Buffer type mapping**:

| String | Integer |
|---|---|
| `DRAM` | 0 |
| `L1` | 1 |
| `SYSTEM_MEMORY` | 2 |
| `L1_SMALL` | 3 |
| `TRACE` | 4 |

The importer prioritizes the `exact_buffer_type` field from `buffer_allocate` nodes (which uses the precise C++ enum) over the generic `type` field.

```bash
python -m ttnn.graph_report my_report.json ./visualizer_db/
```

Or from Python:

```python
from ttnn.graph_report import import_report
db_path = import_report("my_report.json", "./output/")
```

---

## Advanced Features

### Stack Trace Capture

Stack traces are **enabled by default**. Each `function_start` node in the captured graph includes a `stack_trace` array with demangled C++ call frames:

```json
{
    "node_type": "function_start",
    "params": { "name": "ttnn::add" },
    "stack_trace": [
        "/path/to/lib.so(ttnn::add(...)+0x123) [0x7f...]",
        "/path/to/lib.so(some_caller_function+0x456) [0x7f...]"
    ]
}
```

Stack traces are imported into the `stack_traces` table in the visualizer database (one row per operation).

To disable stack traces for reduced overhead:

```python
ttnn.graph.disable_stack_traces()

ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
# ... your operations ...
graph = ttnn.graph.end_graph_capture()

# Re-enable when needed
ttnn.graph.enable_stack_traces()

# Check status
print(ttnn.graph.is_stack_trace_enabled())  # True by default
```

### Buffer Page Capture

Capture detailed per-page information for L1 buffers:

```python
ttnn.graph.enable_buffer_pages()

ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
# ... your operations ...
ttnn.graph.end_graph_capture_to_file("report.json")

ttnn.graph.disable_buffer_pages()
```

The report includes `buffer_pages_by_address` (compact, per-address snapshots with versioned timelines for re-allocations) combined with `per_operation_buffers` to reconstruct per-operation page data. Fields per page:
- `device_id`, `address`: Buffer location
- `core_x`, `core_y`, `bank_id`: Core and bank placement
- `page_index`, `page_address`, `page_size`: Page details
- `buffer_type`: 0=DRAM, 1=L1, 2=SYSTEM_MEMORY, 3=L1_SMALL, 4=TRACE

### Reducing Capture Overhead

By default, examples in this guide capture all available data (slow dispatch, buffer pages, stack traces). To reduce overhead or report size, disable individual features:

| Feature | Disable With | Effect |
|---|---|---|
| Per-op captured sub-graphs | Keep `enable_fast_runtime_mode=True` (default) | Uses `FastOperation`; sub-graphs are reconstructed synthetically by the importer |
| Buffer pages | Don't call `ttnn.graph.enable_buffer_pages()` | Skips per-page L1 detail (~5.8M rows for ResNet-50) |
| Stack traces | `ttnn.graph.disable_stack_traces()` | Omits C++ call stacks from `function_start` nodes |

```python
# Minimal capture: fast dispatch, no buffer pages, no stack traces
ttnn.graph.disable_stack_traces()
ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
# ... your operations ...
ttnn.graph.end_graph_capture_to_file("lightweight_report.json")
ttnn.graph.enable_stack_traces()  # Restore default
```

### Operation Arguments

Python-level arguments are automatically captured by both `FastOperation` and `Operation` decorators when graph capture is active. They are embedded in the `python_io` section of the JSON report and imported into the `operation_arguments` table.

For C++ level arguments:

```python
graph = ttnn.graph.end_graph_capture()

for node in graph:
    if node["node_type"] == "function_start" and "arguments" in node:
        print(f"{node['params']['name']}: {node['arguments']}")
```

### Debugging Hanging Operations

Set a timeout for operations:

```bash
export TT_METAL_OPERATION_TIMEOUT_SECONDS=30
```

Then wrap in try/catch to capture the trace even on timeout:

```python
ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
try:
    result = some_operation(tensor)
    ttnn._ttnn.device.synchronize_device(device)
except Exception as e:
    graph = ttnn.graph.end_graph_capture()
    print(f"Operation hung. Captured trace: {graph}")
    raise
```

---

## Levelized Graph

The `LevelizedGraph` provides a simplified, hierarchical view focused on operation-level data flow. Useful for IR generation and understanding operation composition.

### Basic Usage

```python
graph = ttnn.graph.end_graph_capture()

# Extract with max_level=1 (top-level operations only)
levelized = ttnn.graph.extract_levelized_graph(graph)

# Or expand to see internal operations
levelized = ttnn.graph.extract_levelized_graph(graph, max_level=2)

for vertex in levelized:
    print(f"{vertex['name']} (level {vertex['stacking_level']})")
    print(f"  inputs: {vertex['in_edges']}")
    print(f"  outputs: {vertex['out_edges']}")
    print(f"  internals: {vertex['internals']}")
```

### Vertex Structure

Each vertex contains:
- `counter`: Unique ID
- `stacking_level`: Nesting depth (1=top-level)
- `name`: Operation name or `"tensor[id]"` for inputs
- `in_edges`: IDs of input vertices
- `out_edges`: IDs of consumers
- `internals`: IDs of nested operations
- `output_shape`: Output tensor shapes
- `arguments`: Serialized arguments

### Example: Level 1 vs Level 2

For `ttnn::add(a, ttnn::multiply(b, c))`:

**Level 1** shows:
```
tensor[a] → ttnn::multiply → ttnn::add
tensor[b] ↗               ↗
tensor[c] ↗
```

**Level 2** expands internal operations:
```
tensor[a] → ttnn::multiply ──────────→ ttnn::add
          ↘ ttnn::prim::binary_ng     ↘ ttnn::prim::binary_ng
```

---

## Testing & Validation

### Test Structure

Graph tracing tests are in `tests/ttnn/unit_tests/base_functionality/test_graph_report.py` and are organized into:

#### Unit Tests (no hardware required)

| Test Class | Description |
|---|---|
| `TestImportGraphUnit` | Core import logic: JSON graph → SQLite (operations, tensors, buffers, edges) |
| `TestInputTensorResolution` | Input tensor deduplication and resolution by address |
| `TestImportValidation` | Referential integrity validation of imported data |
| `TestBufferMaxSizePerBank` | Buffer size computation with various layouts and bank counts |
| `TestLinearModelImport` | Mock linear model import with detailed structural assertions |
| `TestDurationExtraction` | Extracting operation durations from graph nodes |
| `TestGraphReportImport` | End-to-end JSON file → SQLite import via `import_report()` |
| `TestReportVersion` | Version mismatch handling |
| `TestResNet50Patterns` | Mock ResNet50 patterns (deallocate synthesis, conv2d lifting, buffer snapshots) |
| `TestSafeArgStr` | Safe argument stringification (avoids triggering graph-tracked ops) |
| `TestRecordPythonOperation` | Python I/O recording (arguments, tensor IDs) |
| `TestStoreCapturedGraph` | Attaching captured sub-graphs to `_python_io_data` |
| `TestBeginGraphCaptureClearing` | Python I/O state clearing on capture start |
| `TestPythonIOArgumentImport` | Python I/O arguments used during import (with C++ fallback) |
| `TestPerOpCapturedGraphImport` | Per-operation captured graph import (with fallback extraction) |
| `TestFromTorchFiltering` | Leading `from_torch` ops filtered, post-compute ones kept |
| `TestFilteredOpPrefixes` | All filtered C++ wrapper operation prefixes excluded |
| `TestPythonIONameMatching` | Multi-op same-name matching and unmatched record handling |
| `TestCapturedGraphFallbackExtraction` | Synthetic sub-graph extraction with counter remapping |
| `TestVersionedBufferPages` | Buffer page timeline with re-allocation snapshots |
| `TestPyIdToCppTensorReconciliation` | Python-assigned tensor IDs reconciled with C++ tensor nodes |

#### Device Tests (require hardware)

| Test Class / Function | Hardware | Description |
|---|---|---|
| `TestGraphCaptureToFile` | Any device | Real capture API (file output, device info, stack traces) |
| `TestStackTraces` | Any device | Stack trace enable/disable and capture verification |
| `TestFastOperationGraphTracking` | Any device | FastOperation records function names, tensor IDs, and connectivity |
| `TestLinearModelE2E` | Wormhole B0 | Structural validation: linear model (ones + linear) |
| `test_resnet50_e2e_graph_capture` | Wormhole B0 | Full ResNet50 E2E graph capture and structural validation |

### Generating a Database for ttnn-visualizer

To generate a complete database with all data for the ttnn-visualizer:

```python
import ttnn
from ttnn import graph_report

device = ttnn.open_device(device_id=0, l1_small_size=24576)

# Full capture: slow dispatch for per-op sub-graphs, buffer pages for L1 detail
with ttnn.manage_config("enable_fast_runtime_mode", False):
    ttnn.graph.enable_buffer_pages()
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    # ... run your model ...
    ttnn.graph.end_graph_capture_to_file("my_report.json")
    ttnn.graph.disable_buffer_pages()

# Import to SQLite (creates db.sqlite, cluster_descriptor.yaml, etc.)
graph_report.import_report("my_report.json", "./my_visualizer_db/")

ttnn.close_device(device)
```

For a lighter-weight capture (faster, smaller reports), see [Reducing Capture Overhead](#reducing-capture-overhead).

### What the E2E Tests Validate

The golden comparison test performs structural checks across every table:

| Table | Compared Properties |
|---|---|
| `operations` | `operation_id`, `name` |
| `tensors` | `shape`, `dtype`, `layout`, `memory_config`, `address`, `buffer_type` |
| `input_tensors` | `operation_id`, `input_index`, resolved tensor `address` |
| `output_tensors` | `operation_id`, `output_index`, resolved tensor `address` |
| `device_tensors` | `address` |
| `buffers` | `operation_id`, `address`, `max_size_per_bank`, `buffer_type`, `buffer_layout` |
| `captured_graph` | `operation_id` |
| `stack_traces` | `operation_id` |
| `buffer_pages` | `address`, `core_y`, `core_x`, `bank_id`, `page_index`, `page_address`, `page_size`, `buffer_type` |

Plus structural integrity checks:
- **No dangling references**: every `tensor_id` in input/output tables exists in `tensors`
- **No orphaned host tensors**: host tensors must be referenced by at least one I/O
- **No backwards edges**: an output of operation N must not be consumed by an earlier operation M (M < N)

---

## Reference: Node Types

The trace is a directed graph where each node has:
- `counter`: Unique node ID
- `node_type`: Type (see below)
- `params`: Key-value parameters
- `connections`: IDs of connected nodes

### capture_start / capture_end

Markers for trace boundaries. `capture_end` includes `duration_ns` for total capture time.

### function_start / function_end

Operation boundaries.

**function_start params:**
- `name`: Operation name (e.g., `"ttnn::add"`)
- `inputs`: Number of input parameters

**function_start fields:**
- `stack_trace`: Array of demangled C++ call frames (present by default, disable with `ttnn.graph.disable_stack_traces()`)
- `arguments`: Serialized operation arguments

**function_end params:**
- `name`: Operation name
- `duration_ns`: Wall-clock duration (optional)

### tensor

Tensor metadata.

**params:**
- `tensor_id`: Unique tensor ID
- `shape`: Tensor shape
- `dtype`: Data type (`"bfloat16"`, `"float32"`, etc.)
- `layout`: Memory layout (`"TILE"`, `"ROW_MAJOR"`)

**Device tensor params (when on device):**
- `memory_config`: Memory configuration string
- `device_id`: Device ID
- `address`: Memory address
- `buffer_type`: Buffer type (`"DRAM"`, `"L1"`, `"L1_SMALL"`)
- `device_tensors`: Array of per-device addresses (multi-device)

### buffer

Buffer metadata.

**params:**
- `size`: Size in bytes
- `type`: Memory type (`"DRAM"`, `"L1"`)
- `layout`: Memory layout
- `device_id`: Device ID

### buffer_allocate / buffer_deallocate

Memory allocation/deallocation events.

**buffer_allocate params:**
- `size`, `address`, `type`, `layout`, `device_id`
- `exact_buffer_type`: Precise C++ buffer type enum (`"DRAM"`, `"L1"`, `"L1_SMALL"`, `"TRACE"`)
- `max_size_per_bank`: Pre-computed per-bank buffer size (from C++ allocator)

### circular_buffer_allocate / circular_buffer_deallocate_all

Circular buffer events for streaming/multi-buffering.

**circular_buffer_allocate params:**
- `size`, `address`, `core_range_set`, `globally_allocated`, `device_id`

### error

Error during capture.

**params:**
- `error_type`: Error type (e.g., `"exception"`)
- `error_message`: Error description
- `error_operation`: Operation that failed

---

## Reference: Sample Trace

Below is a trace of `ttnn::add(Shape[1,1,32,32], Shape[4,1,32,32])` which requires broadcasting via `ttnn::repeat`.

### Call Hierarchy

```
ttnn::add
├── ttnn::repeat
│   ├── ttnn::prim::repeat
│   │   └── Device Operation
│   │       └── create_device_tensor
├── ttnn::prim::binary
│   └── Device Operation
│       └── create_device_tensor
```

### Pretty Print

```
Capture Start
Begin: ttnn::add
    Begin: ttnn::repeat
        Begin: ttnn::prim::repeat
            Begin: Device Operation
                Begin: create_device_tensor
                    Add Device Buffer
                    Allocate Device Buffer
                End: create_device_tensor
                Add Tensor: 2
                Allocate Circular Buffer
            End: Device Operation
        End: ttnn::prim::repeat
    End: ttnn::repeat
    Begin: ttnn::prim::binary
        Begin: Device Operation
            Begin: create_device_tensor
                Add Device Buffer
                Allocate Device Buffer
            End: create_device_tensor
            Add Tensor: 3
            Allocate Circular Buffer (x3)
        End: Device Operation
    End: ttnn::prim::binary
    Deallocate Device Buffer
End: ttnn::add
Deallocate Device Buffer
Capture End
```

### Visualizer View

![visualizer](https://github.com/user-attachments/assets/03df00c6-4902-416d-a26a-6ffe874537a5)

<details>
<summary><b>Full JSON Trace (click to expand)</b></summary>

```json
[
    {
        "connections": [1, 32],
        "counter": 0,
        "node_type": "capture_start",
        "params": {}
    },
    {
        "connections": [3, 5, 6, 18, 30, 31],
        "counter": 1,
        "node_type": "function_start",
        "params": {"inputs": "2", "name": "ttnn::add"}
    },
    {
        "connections": [1, 18],
        "counter": 2,
        "node_type": "tensor",
        "params": {"shape": "ttnn.Shape([4, 3, 32, 32])"}
    },
    {
        "connections": [2, 2],
        "counter": 3,
        "node_type": "buffer",
        "params": {"layout": "INTERLEAVED", "size": "24576", "type": "L1"}
    },
    {
        "connections": [1, 6],
        "counter": 4,
        "node_type": "tensor",
        "params": {"shape": "ttnn.Shape([1, 3, 32, 32])"}
    },
    {
        "connections": [4, 4],
        "counter": 5,
        "node_type": "buffer",
        "params": {"layout": "INTERLEAVED", "size": "6144", "type": "L1"}
    },
    {
        "connections": [7, 17],
        "counter": 6,
        "node_type": "function_start",
        "params": {"inputs": "2", "name": "ttnn::repeat"}
    },
    {
        "connections": [8, 16],
        "counter": 7,
        "node_type": "function_start",
        "params": {"inputs": "5", "name": "ttnn::prim::repeat"}
    },
    {
        "connections": [9, 14, 15],
        "counter": 8,
        "node_type": "function_start",
        "params": {"inputs": "2", "name": "Device Operation"}
    },
    {
        "connections": [10, 11, 12],
        "counter": 9,
        "node_type": "function_start",
        "params": {"inputs": "5", "name": "create_device_tensor"}
    },
    {
        "connections": [13, 13, 13, 13, 13],
        "counter": 10,
        "node_type": "buffer",
        "params": {"layout": "INTERLEAVED", "size": "24576", "type": "L1"}
    },
    {
        "connections": [10],
        "counter": 11,
        "node_type": "buffer_allocate",
        "params": {"address": "1953396066", "layout": "INTERLEAVED", "size": "24576", "type": "L1"}
    },
    {
        "connections": [13],
        "counter": 12,
        "node_type": "function_end",
        "params": {"name": "create_device_tensor"}
    },
    {
        "connections": [18],
        "counter": 13,
        "node_type": "tensor",
        "params": {"shape": "ttnn.Shape([4, 3, 32, 32])"}
    },
    {
        "connections": [],
        "counter": 14,
        "node_type": "circular_buffer_allocate",
        "params": {"address": "0", "core_range_set": "{[(x=0,y=0) - (x=0,y=7)], [(x=1,y=0) - (x=1,y=3)]}", "size": "4096", "globally_allocated": "false"}
    },
    {
        "connections": [13],
        "counter": 15,
        "node_type": "function_end",
        "params": {"name": "Device Operation"}
    },
    {
        "connections": [13],
        "counter": 16,
        "node_type": "function_end",
        "params": {"name": "ttnn::prim::repeat"}
    },
    {
        "connections": [13, 18],
        "counter": 17,
        "node_type": "function_end",
        "params": {"name": "ttnn::repeat"}
    },
    {
        "connections": [19, 29],
        "counter": 18,
        "node_type": "function_start",
        "params": {"inputs": "10", "name": "ttnn::prim::binary"}
    },
    {
        "connections": [20, 25, 26, 27, 28],
        "counter": 19,
        "node_type": "function_start",
        "params": {"inputs": "2", "name": "Device Operation"}
    },
    {
        "connections": [21, 22, 23],
        "counter": 20,
        "node_type": "function_start",
        "params": {"inputs": "5", "name": "create_device_tensor"}
    },
    {
        "connections": [24, 24, 24, 24],
        "counter": 21,
        "node_type": "buffer",
        "params": {"layout": "INTERLEAVED", "size": "24576", "type": "L1"}
    },
    {
        "connections": [21],
        "counter": 22,
        "node_type": "buffer_allocate",
        "params": {"address": "0", "layout": "INTERLEAVED", "size": "24576", "type": "L1"}
    },
    {
        "connections": [24],
        "counter": 23,
        "node_type": "function_end",
        "params": {"name": "create_device_tensor"}
    },
    {
        "connections": [],
        "counter": 24,
        "node_type": "tensor",
        "params": {"shape": "ttnn.Shape([4, 3, 32, 32])"}
    },
    {
        "connections": [],
        "counter": 25,
        "node_type": "circular_buffer_allocate",
        "params": {"address": "0", "core_range_set": "{[(x=0,y=0) - (x=7,y=7)]}", "size": "4096", "globally_allocated": "false"}
    },
    {
        "connections": [],
        "counter": 26,
        "node_type": "circular_buffer_allocate",
        "params": {"address": "0", "core_range_set": "{[(x=0,y=0) - (x=7,y=7)]}", "size": "4096", "globally_allocated": "false"}
    },
    {
        "connections": [],
        "counter": 27,
        "node_type": "circular_buffer_allocate",
        "params": {"address": "0", "core_range_set": "{[(x=0,y=0) - (x=7,y=7)]}", "size": "4096", "globally_allocated": "false"}
    },
    {
        "connections": [24],
        "counter": 28,
        "node_type": "function_end",
        "params": {"name": "Device Operation"}
    },
    {
        "connections": [24],
        "counter": 29,
        "node_type": "function_end",
        "params": {"name": "ttnn::prim::binary"}
    },
    {
        "connections": [10],
        "counter": 30,
        "node_type": "buffer_deallocate",
        "params": {"layout": "INTERLEAVED", "size": "0", "type": "L1"}
    },
    {
        "connections": [24, 33],
        "counter": 31,
        "node_type": "function_end",
        "params": {"name": "ttnn::add"}
    },
    {
        "connections": [21],
        "counter": 32,
        "node_type": "buffer_deallocate",
        "params": {"layout": "INTERLEAVED", "size": "0", "type": "L1"}
    },
    {
        "connections": [],
        "counter": 33,
        "node_type": "capture_end",
        "params": {}
    }
]
```

</details>

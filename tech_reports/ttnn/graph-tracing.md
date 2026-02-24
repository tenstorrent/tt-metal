# TT-NN Graph Tracing

TT-NN provides a mechanism for tracing operations and memory activities during neural network execution. This enables performance analysis, memory profiling, and visualization without modifying your code.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Basic Usage](#basic-usage)
- [Saving Reports](#saving-reports)
- [Advanced Features](#advanced-features)
- [Levelized Graph](#levelized-graph)
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

```python
ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
# ... your operations ...
ttnn.graph.end_graph_capture_to_file("my_report.json")
```

Then import into the visualizer database:

```bash
python -m ttnn.graph_report my_report.json ./visualizer_db/
```

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
- `graph`: Full graph trace
- `devices`: Device information (architecture, grid size, memory)
- `metadata`: Capture timestamp
- `cluster_descriptor`: Cluster configuration (if available)
- `buffer_pages`: Detailed page info (when enabled)

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
- `operation_arguments`: Arguments as `arg_0`, `arg_1`, etc.
- `tensors`: Shape, dtype, layout, memory_config, device_id, address
- `device_tensors`: Per-device addresses for multi-device tensors
- `input_tensors`, `output_tensors`: Tensor-to-operation relationships
- `buffers`, `buffer_pages`: Cumulative memory allocation snapshots per operation
- `captured_graph`: Per-operation subgraph JSON for the visualizer
- `stack_traces`: C++ call stacks (captured by default)
- `errors`: Error information

#### Compatible vs Detailed Mode

By default, the importer runs in **compatible mode** which produces output matching the legacy Python-based visualizer flow:
- Only device tensors are imported (host-only tensors are filtered)
- Tensors are deduplicated by address
- Nested internal operations (e.g., `create_device_tensor`) are filtered out

Use `--detailed` for the full C++ capture including host tensors and internal operations:

```bash
# Compatible mode (default) — matches legacy visualizer format
python -m ttnn.graph_report my_report.json ./visualizer_db/

# Detailed mode — includes host tensors and nested operations
python -m ttnn.graph_report --detailed my_report.json ./visualizer_db/
```

Or from Python:

```python
from ttnn.graph_report import import_report

# Compatible mode (default)
db_path = import_report("my_report.json", "./output/")

# Detailed mode
db_path = import_report("my_report.json", "./output/", detailed=True)
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

The report includes a `buffer_pages` array with:
- `device_id`, `address`: Buffer location
- `core_x`, `core_y`, `bank_id`: Core and bank placement
- `page_index`, `page_address`, `page_size`: Page details
- `buffer_type`: 0=DRAM, 1=L1, 2=SYSTEM_MEMORY, 3=L1_SMALL, 4=TRACE

### Operation Arguments

Capture full argument data for each operation:

```python
ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
# ... your operations ...
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
- `buffer_type`: Buffer type (`"DRAM"`, `"L1"`)
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

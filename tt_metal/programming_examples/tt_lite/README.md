# tt-lite: Lightweight Trace-Based Model Replay

## Overview

tt-lite is a proof-of-concept for a lightweight inference runtime on Tenstorrent devices.
It splits model execution into two phases:

1. **Capture (Python + full TT-NN stack):** Compile and trace a model, serialize the raw
   dispatch command stream plus model weights to a `.ttb` binary file.
2. **Replay (C++ only):** Load the `.ttb` file and replay the trace on hardware — no
   Python, PyTorch, or JIT compilation required.

### Verified Models

| Model | Weights | Output PCC vs TTNN | Top-1 Match |
|-------|---------|-------------------|-------------|
| eltwise add (5+3) | N/A | exact (17.0) | N/A |
| ResNet50 (batch 16, bfloat8_b) | 108 tensors, 28.7 MB | **1.000000** | **16/16 (100%)** |

## Architecture

```
 Capture Phase (Python)                  Replay Phase (C++ binary)
 ==============================          ==============================

 PyTorch model                           .ttb file (from disk)
       |                                        |
       v                                        v
 TT-NN: compile, trace capture           read_trace_binary()
       |                                        |
       v                                        v
 export_trace() -> .ttb                  MeshDevice::create()
  - trace command stream                        |
  - weight data (raw DRAM bytes)                v
  - IO buffer metadata                  Allocate all DRAM buffers
                                        (address-ordered, deterministic)
                                                |
                                                v
                                        Write weights to DRAM
                                        Register trace descriptor
                                                |
                                                v
                                        Write input -> replay -> read output
```

## File Structure

| File | Description |
|------|-------------|
| `trace_binary.py` | Python `.ttb` writer + `export_trace()` helper |
| `trace_binary.h` | C++ `.ttb` reader/writer (header-only) |
| `replay_from_file.cpp` | Generic C++ replay binary — loads any `.ttb` |
| `capture_eltwise_add.py` | Capture script for a simple eltwise add (sanity test) |
| `capture_resnet50.py` | Capture script for ResNet50 with model weights |
| `compare_resnet50.py` | Evaluation script: PCC, top-5, top-1 comparison |
| `capture_and_dump.cpp` | C++ capture tool for eltwise add (standalone) |
| `CMakeLists.txt` | Build config for C++ binaries |

### TTNN API additions (outside tt_lite/)

| File | Change |
|------|--------|
| `ttnn/cpp/ttnn/operations/trace.hpp` | Added `read_raw_buffer_data()` declaration |
| `ttnn/cpp/ttnn/operations/trace.cpp` | Added `read_raw_buffer_data()` — reads raw DRAM bytes via `EnqueueReadMeshBuffer` |
| `ttnn/cpp/ttnn-nanobind/operations/trace.cpp` | nanobind binding for `read_raw_buffer_data` |
| `ttnn/ttnn/__init__.py` | Exports `read_raw_buffer_data` to Python |

## Building

```bash
cd tt-metal

# Build C++ replay binary
cmake --build build_Release --target tt_lite_replay -j$(nproc)

# Build TTNN (needed if you modified read_raw_buffer_data)
cmake --build build_Release --target ttnn -j$(nproc)
```

Output binary: `build_Release/programming_examples/tt_lite/tt_lite_replay`

## Usage

### Quick Start: Eltwise Add (no weights, sanity test)

```bash
# Capture
python tt_metal/programming_examples/tt_lite/capture_eltwise_add.py \
  --output eltwise_add.ttb

# Replay
./build_Release/programming_examples/tt_lite/tt_lite_replay eltwise_add.ttb
```

### ResNet50: Full Model with Weights

**Step 1: Capture** (requires Python, PyTorch, torchvision, TT-NN)

```bash
export PYTHONPATH=$(pwd)
export TT_METAL_HOME=$(pwd)

python tt_metal/programming_examples/tt_lite/capture_resnet50.py \
  --output resnet50.ttb \
  --save-ref resnet50_ref/
```

This produces:
- `resnet50.ttb` — trace binary with 108 weight tensors (~30 MB)
- `resnet50_ref/input.bin` — raw bfloat16 input data
- `resnet50_ref/ttnn_trace_output.pt` — TTNN reference output
- `resnet50_ref/pytorch_reference_output.pt` — PyTorch reference output

**Step 2: Replay** (C++ only, no Python/PyTorch needed)

```bash
./build_Release/programming_examples/tt_lite/tt_lite_replay \
  resnet50.ttb \
  --trace-region-size 5554176 --l1-small-size 24576 \
  --input dram_input=resnet50_ref/input.bin \
  --output output=resnet50_ref/replay_output.bin
```

**Step 3: Evaluate** (compare replay output vs references)

```bash
python tt_metal/programming_examples/tt_lite/compare_resnet50.py \
  resnet50_ref/ \
  resnet50_ref/replay_output.bin
```

Expected output:
```
--- Pearson Correlation Coefficient ---
  TTNN trace vs PyTorch:   0.989742
  C++ replay vs PyTorch:   0.989742
  C++ replay vs TTNN trace: 1.000000

--- Top-1 Agreement (batch=16) ---
  C++ replay vs TTNN trace: 16/16 (100.0%)

--- Summary ---
  PASS: C++ replay output matches TTNN trace output (PCC=1.000000 > 0.999)
```

### Docker Environment

On multi-device systems, run inside a Docker container with a single device:

```bash
sudo docker run --rm --memory=16g \
  -v /home:/home \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --device /dev/tenstorrent/0 \
  <container_image> bash -c '
    export PYTHONPATH=/tt-metal TT_METAL_HOME=/tt-metal
    cd /path/to/tt-metal
    python tt_metal/programming_examples/tt_lite/capture_resnet50.py --output resnet50.ttb
  '
```

### Replay CLI Reference

```
tt_lite_replay <model.ttb> [options]

Options:
  --input  NAME=FILE    Write binary file contents to IO buffer NAME before replay
  --output NAME=FILE    Read IO buffer NAME after replay and save to FILE
  --trace-region-size N Override trace region size (bytes)
  --l1-small-size N     Override L1 small size (bytes)
```

## .ttb File Format

```
+------------------------------+
| Header (24 bytes)            |
|   magic: "TTB0"              |
|   version, counts            |
+------------------------------+
| Worker Descriptors[]         |
|   sub_device_id, core counts |
+------------------------------+
| Trace Streams[]              |
|   raw dispatch command data  |
+------------------------------+
| Persistent Buffers[]         |
|   addr, size, page_size,     |
|   buffer_type, raw data      |
+------------------------------+
| IO Buffers[]                 |
|   addr, size, page_size,     |
|   buffer_type, name          |
+------------------------------+
| Trace Buffer Placement       |
|   addr, page_size, num_pages |
+------------------------------+
```

Buffer types: `0 = DRAM` (allocated and written by replay), `1 = L1` (trace-managed, skipped).

## Key Design Decisions

### DRAM→L1 reshard inside the trace

The trace includes the `to_memory_config` operation that moves input data from
DRAM (interleaved) to L1 (height-sharded). This is critical: without it, the C++
replay would need to populate L1 directly, which requires knowing the exact
sharding layout. By including it in the trace, the replay only needs to write
flat data to an interleaved DRAM buffer.

### Raw DRAM reads for weight serialization

Weight data is read from device via `ttnn.read_raw_buffer_data()` which calls
`EnqueueReadMeshBuffer` to get the exact bytes as stored in DRAM — no detilization
or dtype conversion. This is necessary because the trace expects weights in their
on-device format (tilized, bfloat8_b packed, etc.), not in host-side float format.

### Address-ordered allocation

All DRAM buffers (persistent weights + IO) are allocated in ascending address order
during replay, matching the capture-time allocator state. This ensures each buffer
lands at the same address the trace commands reference.

## Limitations

1. **Single-chip only** — 1x1 MeshDevice. Multi-chip would need per-device trace streams.
2. **Address determinism** — relies on deterministic allocator. A production version
   would need address patching or fixed-address allocation.
3. **libtt_metal dependency** — replay still links full `libtt_metal.so` for device init
   and dispatch firmware. A minimal runtime extraction is future work.
4. **Architecture-specific** — trace commands are Blackhole-specific. No cross-arch portability.
5. **Dispatch FW coupling** — trace is tied to the dispatch firmware version at capture time.

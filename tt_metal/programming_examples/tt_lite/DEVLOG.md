# tt-lite Development Log

## 2026-05-14: v0 — Initial PoC (commit 1e9ebff)

**Goal:** Prove that a TT-NN traced model can be replayed from a standalone C++ binary
using only serialized dispatch commands and weight data.

**What was built:**
- `.ttb` binary format (v0): header, worker descriptors, trace streams, persistent
  buffer data, IO buffer metadata
- Python capture: `export_trace()` serializes trace + weights from a running TTNN model
- C++ replay: `replay_from_file.cpp` loads .ttb, allocates buffers in address order,
  registers trace, replays
- TTNN API additions: `read_raw_buffer_data()` for exact DRAM byte reads,
  `get_trace_data()` for trace metadata export

**Models verified:**
- eltwise add (5+3): exact output match
- ResNet50 (batch 16, bfloat8_b): PCC=1.000000, Top-1 16/16

**Key insight:** Including `to_memory_config` (DRAM -> L1 reshard) inside the trace
means the C++ replay only writes flat interleaved DRAM data — no need to replicate
the sharding layout logic.

---

## 2026-05-20: v1 — JIT Cache Embedding (commit 980b576)

**Problem:** When replaying on a clean system (or inside Docker), the device
initialization triggers RISC-V cross-compilation of dispatch kernels. This is slow
(~150s for ResNet50's 260 kernels) and can fail with "virtual memory exhausted" on
some host configurations.

**Solution:** Embed the entire JIT cache (`~/.cache/tt-metal-cache/`) in the .ttb file.
At replay time, extract all cached kernel artifacts before `MeshDevice::create()`.

**Changes:**
- `trace_binary.h/.py`: Added `JitCacheFile` struct, TTB_VERSION bumped to 1,
  serialization/deserialization for JIT cache section
- `capture_resnet50.py`: Prints JIT cache path info for debugging
- `replay_from_file.cpp`: Added `extract_jit_cache()`, called before device init
- `trace_binary.py` (`export_trace()`): Walks `~/.cache/tt-metal-cache/` and embeds
  all files in .ttb

**Verification (Blackhole, Docker, single device):**
- 2585 JIT cache files (234 MB) embedded in .ttb (total .ttb size: 265 MB)
- After deleting JIT cache, replay extracted all files and achieved 16/16 JIT hits
- PCC (replay vs TTNN trace) = 1.000000
- Top-1 agreement: 16/16 (100%)

**Environment notes:**
- Must run capture/replay in Docker on this machine (`metalcon:may11build`);
  host environment has stale RISC-V compiler that OOMs on dispatch kernel compilation
- `TT_METAL_RUNTIME_ROOT` must be set for the C++ binary (not auto-detected from CWD)
- JIT cache is per-device-architecture; a .ttb captured on Blackhole won't provide
  valid cache for Wormhole

---

## Bringing Up a New Model

To add a new model to tt-lite:

1. **Create `capture_<model>.py`** following `capture_resnet50.py` as template:
   - Load model, preprocess parameters, move to device
   - Run compile pass (warmup)
   - Set up DRAM input buffer (interleaved) and include `to_memory_config` in trace
   - `ttnn.begin_trace_capture()` / `ttnn.end_trace_capture()`
   - Collect weight tensors with `collect_device_tensors_from_model()`
   - Call `export_trace()` with IO tensors and weight tensors

2. **Capture inside Docker:**
   ```bash
   sudo docker run --rm -v /home:/home -v /dev/hugepages-1G:/dev/hugepages-1G \
     --device /dev/tenstorrent/0 metalcon:may11build bash -c '
     export TT_METAL_HOME=/home/yito/ttwork/tt-metal
     export PYTHONPATH="/home/yito/ttwork/tt-metal/tools:/home/yito/ttwork/tt-metal/ttnn:/home/yito/ttwork/tt-metal"
     export MPLCONFIGDIR=/tmp/mpl
     cd $TT_METAL_HOME/tt_metal/programming_examples/tt_lite
     python3 capture_<model>.py --output <model>.ttb --save-ref <model>_ref
   '
   ```

3. **Replay:**
   ```bash
   sudo docker run --rm -v /home:/home -v /dev/hugepages-1G:/dev/hugepages-1G \
     --device /dev/tenstorrent/0 metalcon:may11build bash -c '
     export TT_METAL_HOME=/home/yito/ttwork/tt-metal
     export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
     cd $TT_METAL_HOME/tt_metal/programming_examples/tt_lite
     $TT_METAL_HOME/build/programming_examples/tt_lite/tt_lite_replay <model>.ttb \
       --input <input_name>=<model>_ref/input.bin \
       --output <output_name>=<model>_ref/replay_output.bin
   '
   ```

4. **Verify:** Compare replay output against TTNN trace reference (PCC > 0.999).

### Common Issues

- **"virtual memory exhausted"** during device init: Run in Docker, or ensure JIT cache
  is populated (capture first, or use a .ttb with embedded JIT cache)
- **"Root Directory is not set"**: Set `TT_METAL_RUNTIME_ROOT` env var
- **Address mismatch warnings**: Buffer allocation order may differ if device config
  (`trace_region_size`, `l1_small_size`) doesn't match capture time. Use `--trace-region-size`
  and `--l1-small-size` flags to override.
- **L1 buffers skipped**: Expected. L1 (sharded) buffers are managed by the trace;
  only DRAM buffers need explicit allocation.

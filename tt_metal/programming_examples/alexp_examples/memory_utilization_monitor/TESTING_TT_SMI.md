# Testing Guide: tt_smi with TT-Metal Workloads

This guide provides step-by-step instructions for testing `tt_smi` with TT-Metal workloads.

---

## Prerequisites

- Tenstorrent hardware (N300, T3000, Galaxy, etc.)
- Ubuntu/Linux system with PCIe access to Tenstorrent devices
- Python 3.8+ environment

---

## Initial Setup

### 1. Create Environment

```bash
cd /path/to/tt-metal
./create_env.sh
```

This creates a Python virtual environment with all necessary dependencies.

### 2. Source Environment Variables

```bash
source ./env_vars_setup.sh
```

This sets up:
- `ARCH_NAME` (e.g., `wormhole_b0`)
- `TT_METAL_HOME` (path to tt-metal)
- Python environment activation
- UMD library paths

### 3. Build TT-Metal

```bash
./build_metal_with_flags.sh
```

**What it builds:**
- TT-Metal runtime libraries
- Device firmware
- UMD (User Mode Driver)
- Programming examples including `tt_smi`

**Build artifacts:**
- `build/lib/` - Metal libraries
- `build/programming_examples/tt_smi` - Monitoring tool

### 4. Install Python Package

```bash
pip install -e .
```

This installs `tt_metal` Python package in editable mode.

---

## Verify Installation

### Check Devices

```bash
# List detected Tenstorrent devices
tt-topology
```

Expected output:
```
Chip 0: Wormhole B0 (PCIe)
Chip 1: Wormhole B0 (Remote via Chip 0)
Chip 2: Wormhole B0 (PCIe)
Chip 3: Wormhole B0 (Remote via Chip 2)
```

### Check SHM Tracking (Default: Enabled)

```bash
# SHM tracking is enabled by default
# To verify, run a simple Metal operation:
python -c "import ttnn; device = ttnn.open_device(0); ttnn.close_device(device)"

# Check for SHM files
ls -lh /dev/shm/tt_device_*
```

Expected output:
```
-rw-rw-r-- 1 user user 256K Dec 19 08:00 /dev/shm/tt_device_469504_memory
-rw-rw-r-- 1 user user 256K Dec 19 08:00 /dev/shm/tt_device_469505_memory
```

---

## Basic Testing

### Test 1: Monitor Idle Devices

**Terminal 1: Start monitoring**
```bash
cd /path/to/tt-metal
./build/programming_examples/tt_smi -w
```

Expected output:
```
+==============================================================+
| tt-smi - Tenstorrent System Management Interface            |
+==============================================================+

ID          Arch          Temp      Power     AICLK       DRAM Usage          L1 Usage            Status
-------------------------------------------------------------------------------------------------------
261834045   Wormhole_B0   35°C     12W       500 MHz     0.0B / 12.0GiB      0.0B / 171.6MiB     OK
361834045R  Wormhole_B0   33°C     10W       500 MHz     0.0B / 12.0GiB      0.0B / 171.6MiB     OK
26191901e   Wormhole_B0   34°C     11W       500 MHz     0.0B / 12.0GiB      0.0B / 171.6MiB     OK
36191901eR  Wormhole_B0   32°C     10W       500 MHz     0.0B / 12.0GiB      0.0B / 171.6MiB     OK

Per-Process Memory Usage:
(No active processes)
```

**Observations:**
- ✅ All devices show 0 memory usage (idle state)
- ✅ Telemetry working (temperature, power, AICLK)
- ✅ Remote devices accessible (marked with 'R' suffix)

---

## Advanced Testing

### Test 2: Monitor Llama Inference (All Devices)

**Terminal 1: Start fast monitoring (150ms refresh)**
```bash
./build/programming_examples/tt_smi -w -r 150
```

**Terminal 2: Run Llama inference on all devices**
```bash
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
    -k "batch-32 and performance"
```

Expected output in Terminal 1:
```
ID          Arch          Temp      Power     AICLK       DRAM Usage          L1 Usage            Status
-------------------------------------------------------------------------------------------------------
261834045   Wormhole_B0   51°C     35W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK
361834045R  Wormhole_B0   48°C     33W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK
26191901e   Wormhole_B0   49°C     34W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK
36191901eR  Wormhole_B0   46°C     32W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK

Per-Process Memory Usage:
----------------------------------------------------------------------------------------------------------------------------------
Dev         PID     Process         DRAM        L1        L1 Small  Trace     CB        Kernel
----------------------------------------------------------------------------------------------------------------------------------
261834045   12345   pytest          4.5GiB      2.2MiB    0.0B      22.5MiB   29.8MiB   53.5MiB
361834045R  12345   pytest          4.5GiB      2.2MiB    0.0B      22.5MiB   29.8MiB   53.5MiB
26191901e   12345   pytest          4.5GiB      2.2MiB    0.0B      22.5MiB   29.8MiB   53.5MiB
36191901eR  12345   pytest          4.5GiB      2.2MiB    0.0B      22.5MiB   29.8MiB   53.5MiB
```

**Observations:**
- ✅ Memory allocations appear during model loading
- ✅ DRAM contains model weights (~4.5 GiB)
- ✅ L1 contains kernels, CBs, and activations (~85 MiB)
- ✅ Per-process tracking shows which PID owns which allocations
- ✅ Remote devices show same allocations as local (chip-to-chip workload)
- ✅ AICLK increases to 1000 MHz during inference
- ✅ Temperature and power increase under load

---

### Test 3: Selective Device Testing with TT_VISIBLE_DEVICES

**Important:** `TT_VISIBLE_DEVICES` only accepts **MMIO-capable device IDs**.

For N300 systems:
- **Device 0** (MMIO) controls **both** Chip 0 (local) **and** Chip 1 (remote)
- **Device 1** (MMIO) controls **both** Chip 2 (local) **and** Chip 3 (remote)

You **cannot** selectively use only local or only remote chips via `TT_VISIBLE_DEVICES`.

#### 3a. Run on First N300 Board Only (Chips 0 and 1)

**Terminal 1: Monitor**
```bash
./build/programming_examples/tt_smi -w -r 150
```

**Terminal 2: Use Device 0 (controls both chips 0 and 1)**
```bash
export TT_VISIBLE_DEVICES=0
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
    -k "batch-32 and performance"
```

Expected output:
```
ID          Arch          Temp      Power     AICLK       DRAM Usage          L1 Usage            Status
-------------------------------------------------------------------------------------------------------
261834045   Wormhole_B0   51°C     35W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK    ← Active (local)
361834045R  Wormhole_B0   48°C     33W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK    ← Active (remote)
26191901e   Wormhole_B0   34°C     11W       500 MHz     0.0B / 12.0GiB      0.0B / 171.6MiB     OK    ← Idle
36191901eR  Wormhole_B0   32°C     10W       500 MHz     0.0B / 12.0GiB      0.0B / 171.6MiB     OK    ← Idle
```

**Observations:**
- ✅ Both chips 0 and 1 show memory allocations (local + remote on Device 0)
- ✅ Chips 2 and 3 remain idle (Device 1 not used)
- ✅ You get **both** local and remote chips when using an MMIO device

#### 3b. Run on Second N300 Board Only (Chips 2 and 3)

**Terminal 2: Use Device 1 (controls both chips 2 and 3)**
```bash
export TT_VISIBLE_DEVICES=1
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
    -k "batch-32 and performance"
```

Expected output:
```
ID          Arch          Temp      Power     AICLK       DRAM Usage          L1 Usage            Status
-------------------------------------------------------------------------------------------------------
261834045   Wormhole_B0   34°C     11W       500 MHz     0.0B / 12.0GiB      0.0B / 171.6MiB     OK    ← Idle
361834045R  Wormhole_B0   32°C     10W       500 MHz     0.0B / 12.0GiB      0.0B / 171.6MiB     OK    ← Idle
26191901e   Wormhole_B0   51°C     35W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK    ← Active (local)
36191901eR  Wormhole_B0   48°C     33W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK    ← Active (remote)
```

**Observations:**
- ✅ Both chips 2 and 3 show memory allocations (local + remote on Device 1)
- ✅ Chips 0 and 1 remain idle (Device 0 not used)
- ✅ SHM files are chip-specific, unaffected by `TT_VISIBLE_DEVICES`

#### 3c. Run on All Devices (Default)

**Terminal 2: Use both MMIO devices**
```bash
export TT_VISIBLE_DEVICES=0,1
# or simply don't set TT_VISIBLE_DEVICES (all devices by default)
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
    -k "batch-32 and performance"
```

Expected output:
```
ID          Arch          Temp      Power     AICLK       DRAM Usage          L1 Usage            Status
-------------------------------------------------------------------------------------------------------
261834045   Wormhole_B0   51°C     35W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK    ← Active
361834045R  Wormhole_B0   48°C     33W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK    ← Active
26191901e   Wormhole_B0   51°C     35W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK    ← Active
36191901eR  Wormhole_B0   48°C     33W       1000 MHz    4.5GiB / 12.0GiB    85.3MiB / 171.6MiB  OK    ← Active
```

**Observations:**
- ✅ All 4 chips active (2 MMIO devices × 2 chips each)
- ✅ Maximum compute capacity utilized

---

### Test 4: Multi-Process Workloads

Run multiple workloads simultaneously to test per-PID tracking:

**Terminal 1: Monitor**
```bash
./build/programming_examples/tt_smi -w -r 150
```

**Terminal 2: Run Llama on Device 0 (Chips 0 and 1)**
```bash
export TT_VISIBLE_DEVICES=0
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
    -k "batch-32 and performance"
```

**Terminal 3: Simultaneously run Llama on Device 1 (Chips 2 and 3)**
```bash
export TT_VISIBLE_DEVICES=1
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
    -k "batch-32 and performance"
```

Expected output:
```
Per-Process Memory Usage:
----------------------------------------------------------------------------------------------------------------------------------
Dev         PID     Process         DRAM        L1        L1 Small  Trace     CB        Kernel
----------------------------------------------------------------------------------------------------------------------------------
261834045   12345   pytest          4.5GiB      2.2MiB    0.0B      22.5MiB   29.8MiB   53.5MiB    ← Process 1
361834045R  12345   pytest          4.5GiB      2.2MiB    0.0B      22.5MiB   29.8MiB   53.5MiB    ← Process 1
26191901e   12346   pytest          4.5GiB      2.2MiB    0.0B      22.5MiB   29.8MiB   53.5MiB    ← Process 2
36191901eR  12346   pytest          4.5GiB      2.2MiB    0.0B      22.5MiB   29.8MiB   53.5MiB    ← Process 2
```

**Observations:**
- ✅ Two different PIDs visible
- ✅ Each process isolated to its assigned devices
- ✅ Memory tracking correct for each process
- ✅ No interference between processes

---

### Test 5: Dead Process Cleanup (Automatic)

Test that `tt_smi` automatically cleans up dead PIDs:

**Terminal 1: Monitor**
```bash
./build/programming_examples/tt_smi -w -r 150
```

**Terminal 2: Start a workload (Device 0 = Chips 0 and 1)**
```bash
export TT_VISIBLE_DEVICES=0
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
    -k "batch-32 and performance"
```

**During workload execution:**
- ✅ `tt_smi` shows process memory allocations

**Kill the process (Ctrl+C in Terminal 2)**

**After killing:**
- ✅ Within 150ms, `tt_smi` detects dead PID via `kill(pid, 0)`
- ✅ Yellow warning appears: "⚠ Cleaned up 1 dead process(es)"
- ✅ Memory counters decrease to 0
- ✅ Per-process table no longer shows the dead PID

**What's happening:**
1. `tt_smi` checks if each PID is alive using `kill(pid, 0)`
2. For dead PIDs, it:
   - Subtracts their allocations from aggregate SHM counters
   - Clears their entry from `processes[]` array
3. All SHM files are scanned (local and remote devices)

---

## Testing with Different Refresh Rates

### High-frequency monitoring (100ms)
```bash
./build/programming_examples/tt_smi -w -r 100
```
- Good for: Detailed observation during loading/inference
- Overhead: ~1-2ms per refresh (negligible)

### Standard monitoring (1s)
```bash
./build/programming_examples/tt_smi -w
# or
./build/programming_examples/tt_smi -w -r 1000
```
- Good for: General monitoring, production use
- Overhead: Negligible

### Low-frequency monitoring (5s)
```bash
./build/programming_examples/tt_smi -w -r 5000
```
- Good for: Long-running workloads, minimal overhead
- Overhead: None

---

## Disabling SHM Tracking (For Benchmarking)

To test overhead-free inference:

```bash
# Disable SHM tracking
export TT_METAL_SHM_TRACKING_DISABLED=1

# Run workload
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
pytest models/tt_transformers/demo/simple_text_demo.py::test_demo_text \
    -k "batch-32 and performance"
```

**Result:**
- ✅ `tt_smi` shows "No tracking data"
- ✅ Inference runs ~110-140ns faster per allocation (negligible impact)
- ✅ No SHM files created

---

## Troubleshooting

### Issue: tt_smi shows "No tracking data"

**Possible causes:**
1. SHM tracking explicitly disabled
2. No Metal application running
3. SHM files not created

**Solution:**
```bash
# Check if tracking is disabled
echo $TT_METAL_SHM_TRACKING_DISABLED
# Should be empty or "0"

# Check for SHM files
ls -lh /dev/shm/tt_device_*

# If no files, run a simple Metal operation
python -c "import ttnn; device = ttnn.open_device(0); ttnn.close_device(device)"
```

### Issue: Stale memory allocations after process crash

**Solution:**
- `tt_smi` automatically cleans dead PIDs by default
- If stale data persists, manually clean SHM files:
```bash
rm /dev/shm/tt_device_*_memory
```

### Issue: Remote devices not showing telemetry

**Possible causes:**
1. Ethernet link down
2. Remote firmware hung
3. Ethernet cores busy

**Solution:**
```bash
# Check UMD topology discovery
tt-topology

# Reset devices
tt-smi --reset

# Check tt_smi logs
./build/programming_examples/tt_smi -w
# Look for "ETH busy" or "chip_offline" messages
```

### Issue: Build errors

**Solution:**
```bash
# Clean build
rm -rf build
./build_metal_with_flags.sh
```

---

## Expected Performance Characteristics

### SHM Tracking Overhead
- **Per allocation**: ~110-140ns
- **First allocation per chip**: ~20-50μs (one-time setup)
- **Total for 500K allocations**: ~70ms (negligible)

### tt_smi Overhead
- **Telemetry query**:
  - Local chips: ~100μs
  - Remote chips: ~5-10ms
- **Dead PID cleanup**: ~1-2ms (all SHM files)
- **Display refresh**: <1ms

### Memory Tracking Accuracy
- **Real-time**: All buffer types tracked immediately
- **Kernel estimate**: Calculated at compile time (binary_size × cores)
- **Cleanup**: Dead PIDs cleaned within next refresh interval

---

## Summary

This testing guide covers:
1. ✅ **Environment setup** - Create, source, build, install
2. ✅ **Basic testing** - Idle monitoring, single workload
3. ✅ **Advanced testing** - Selective devices, multi-process
4. ✅ **Cleanup testing** - Automatic dead PID detection
5. ✅ **Performance** - Different refresh rates, overhead measurement
6. ✅ **Troubleshooting** - Common issues and solutions

**Key Features Tested:**
- Default-enabled SHM tracking
- Per-PID memory breakdown
- Automatic dead process cleanup
- Remote device telemetry via Ethernet
- Multi-chip workload monitoring
- Selective device usage with `TT_VISIBLE_DEVICES`

**Result:** Complete observability of Tenstorrent devices with negligible overhead! 🚀


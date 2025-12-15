# Testing Fabric Telemetry Initialization

This guide explains how to test the fabric telemetry initialization changes.

## Overview

The changes add initialization of fabric telemetry structures in ERISC firmware:
1. Zeros telemetry structures to eliminate garbage values
2. Populates `static_info` fields (mesh_id, device_id, direction)
3. Sets `supported_stats` bitmask to enable telemetry features

## Prerequisites

### Remote Machine Access
According to CLAUDE.md, you have access to:
- **IRD machines:** `ssh yyz-ird` then `ird reserve`
- **Closetbox cluster:** `ssh closetbox` (bare Slurm)

### Build Requirements
On the remote machine:
```bash
cd /path/to/tt-metal
git fetch origin
git checkout kkfernandez/fabric-telemetry-firmware-init

# Build tt-metal (user responsibility per CLAUDE.md)
# Follow your standard build process
```

## Testing Methods

### Method 1: C++ Unit Test (Recommended)

A C++ test has been added to verify telemetry initialization.

**Location:** `tests/tt_metal/tt_fabric/test_telemetry_init.cpp`

**To run:**
```bash
# Build tests
./build.sh --build-tests

# Run the specific telemetry test
./build/test/tt_metal/tt_fabric/test_telemetry_init
```

**What it checks:**
- ✅ `mesh_id` matches device's fabric node ID
- ✅ `device_id` matches device's chip ID
- ✅ `direction` is valid (0-3 for EAST/WEST/NORTH/SOUTH)
- ✅ `supported_stats` is non-zero
- ✅ BANDWIDTH telemetry bit is enabled
- ✅ Counters are zeroed (not garbage values > 10^12)

**Expected output:**
```
[ RUN      ] TGFabricFixture.TelemetryStaticInfoInitialized
[INFO    ] Channel 0 telemetry: mesh_id=0, device_id=0, direction=0, supported_stats=0x0f
[INFO    ] Channel 1 telemetry: mesh_id=0, device_id=0, direction=1, supported_stats=0x0f
...
[  PASSED  ] TGFabricFixture.TelemetryStaticInfoInitialized
```

### Method 2: Python Verification Script

A Python script is provided for manual verification.

**Location:** `test_fabric_telemetry_init.py`

**To run:**
```bash
python test_fabric_telemetry_init.py
```

**What it outputs:**
```
================================================================================
Fabric Telemetry Initialization Verification
================================================================================

Found 8 device(s): [0, 1, 2, 3, 4, 5, 6, 7]

Initializing mesh device...

================================================================================
Device 0
================================================================================
Fabric Node ID: mesh_id=0, chip_id=0

Found 8 ethernet channel(s) with telemetry

  Channel 0:
    mesh_id:         0
    device_id:       0
    direction:       0 (EAST)
    fabric_config:   0x00000000
    supported_stats: 0x0f (ROUTER_STATE, BANDWIDTH, HEARTBEAT_TX, HEARTBEAT_RX)
    ✅ PASS: Telemetry properly initialized

  Channel 1:
    mesh_id:         0
    device_id:       0
    direction:       1 (WEST)
    ...

================================================================================
Summary
================================================================================
Total channels tested: 64
✅ ALL TESTS PASSED
```

### Method 3: Existing Fabric Tests

Run existing fabric tests to ensure no regressions:

```bash
# Basic fabric smoke tests
./build/test/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_smoke

# Basic fabric data movement
./build/test/tt_metal/tt_fabric/fabric_data_movement/test_basic_1d_fabric

# Fabric APIs
./build/test/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_apis
```

**Expected:** All existing tests should pass unchanged.

### Method 4: tt-telemetry Integration (Full Stack)

If tt-telemetry is available, test the full telemetry stack:

```bash
# Set environment variable to enable telemetry
export TT_METAL_FABRIC_TELEMETRY=1

# Run a fabric workload (e.g., from tests/tt_metal/tt_metal/perf_microbenchmark/routing/)
python tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_tt_fabric.py

# In another terminal, if tt_telemetry_server is built:
./build/tt_telemetry/tt_telemetry_server --fsd=fsd.textproto --watchdog-timeout 60

# Query metrics
curl -s http://localhost:8080/api/metrics | grep -i bandwidth
```

**Expected output:**
```
# HELP txBandwidthMBps Transmit bandwidth in MB/s
# TYPE txBandwidthMBps gauge
txBandwidthMBps{mesh_id="0",device_id="0",channel="0",direction="EAST"} 1234.56
...
```

**Before these changes:** Bandwidth metrics would not appear (missing static_info)

**After these changes:** Bandwidth metrics should be present with correct labels

### Method 5: Manual Telemetry Read

For debugging, manually read telemetry from L1:

```python
import ttnn

# Initialize device
device_id = 0
device = ttnn.open_device(device_id)

# Telemetry address (from eth_l1_address_map.h)
TELEMETRY_ADDR = 0x43C00  # MEM_AERISC_FABRIC_TELEMETRY_BASE for Wormhole
TELEMETRY_SIZE = 128

# Get first ethernet core
eth_cores = device.get_active_ethernet_cores()
if eth_cores:
    core = eth_cores[0]

    # Read telemetry structure
    data = ttnn.experimental.core_read(device, core, TELEMETRY_ADDR, TELEMETRY_SIZE)

    # Parse first 8 bytes (StaticInfo header)
    import struct
    mesh_id, device_id, direction, fabric_config, supported_stats = struct.unpack('<HBBIB', data[:9])

    print(f"mesh_id: {mesh_id}")
    print(f"device_id: {device_id}")
    print(f"direction: {direction}")
    print(f"supported_stats: 0x{supported_stats:02x}")

ttnn.close_device(device)
```

**Expected values:**
- `mesh_id`: Should match device's mesh ID (typically 0)
- `device_id`: Should match device's chip ID within mesh (0-N)
- `direction`: 0-3 (EAST/WEST/NORTH/SOUTH)
- `supported_stats`: 0x0F (all features enabled)

## What to Look For

### ✅ Success Indicators

1. **No garbage values in logs**
   - Before: "Suspiciously large cycle delta (17287876335105121438)"
   - After: No such warnings

2. **Bandwidth metrics appear**
   - Before: Missing from telemetry output
   - After: Present with valid values

3. **Static info populated**
   - `mesh_id` != 0 (unless actually mesh 0)
   - `device_id` matches device position
   - `direction` in range [0-3]
   - `supported_stats` != 0

4. **Counters start at zero**
   - All bandwidth counters < 10^12 at first read
   - No massive deltas on second read

### ❌ Failure Indicators

1. **Garbage values persist**
   - Counters > 10^12 on first read
   - Warnings about "suspiciously large deltas"

2. **Static info not set**
   - `mesh_id` = 0 and `device_id` = 0 for all channels
   - `supported_stats` = 0

3. **Bandwidth metrics missing**
   - tt-telemetry doesn't expose bandwidth gauges
   - dynamic_info unavailable in telemetry reader

4. **Tests fail**
   - Unit test assertions fail
   - Python script reports failures

## Debugging Tips

### Check Firmware Loaded
```bash
# Verify firmware contains initialization code
strings build/tt_metal/hw/firmware/src/tt-1xx/active_erisc.elf | grep -i telemetry
```

### Check Memory Layout
Telemetry address depends on architecture:
- **Wormhole:** `0x43C00` (MEM_AERISC_FABRIC_TELEMETRY_BASE)
- **Blackhole:** Check `tt_metal/hw/inc/tt-1xx/blackhole/dev_mem_map.h`

### Check Compile-Time Args
Enable debug output in builder:
```cpp
// In erisc_datamover_builder.cpp, before get_compile_time_args returns:
log_debug(tt::LogMetal, "mesh_id CT arg: {}", this->local_fabric_node_id.mesh_id.value());
log_debug(tt::LogMetal, "device_id CT arg: {}", this->local_fabric_node_id.chip_id);
```

### Compare Before/After
1. Check out main branch, build, run test → note failures
2. Check out feature branch, build, run test → should pass
3. Compare telemetry dumps from both versions

## CI/CD Considerations

These tests should be integrated into fabric CI:
- Add to `tests/tt_metal/tt_fabric/` test suite
- Run as part of post-commit fabric tests
- Verify on both Wormhole and Blackhole architectures

## Next Steps After Testing

1. ✅ Verify telemetry initialization works
2. ✅ Check bandwidth metrics appear in tt-telemetry
3. ✅ Run on multi-device/multi-mesh configurations
4. ✅ Test after device reset (power cycle)
5. ✅ Verify no performance regression in fabric tests

## Questions / Issues

If tests fail or behavior is unexpected:
1. Check firmware build logs for errors
2. Verify memory layout matches architecture
3. Confirm fabric is properly initialized before reading telemetry
4. Reach out to fabric team (@ubcheema, @aliuTT, @SeanNijjar per FIRMWARE_TELEMETRY_GUIDE.md)

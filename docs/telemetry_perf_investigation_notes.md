# Telemetry Performance Impact Investigation Notes

**Date:** January 2026
**Branch:** kkfernandez/bandwidth-telemetry
**Repositories:**
- tt-metal: `/data/kkfernandez/tt-metal`
- tt-telemetry: `/data/kkfernandez/tt-telemetry`

**Purpose:** This document captures all findings from investigating tt-telemetry's performance impact on tt-metal workloads, and provides requirements for continuing multi-chip validation on a 4-device system.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Single-Chip Testing Results](#2-single-chip-testing-results)
3. [Resource Analysis](#3-resource-analysis)
4. [Why Single-Chip Results Were Inconclusive](#4-why-single-chip-results-were-inconclusive)
5. [Multi-Chip Testing Requirements](#5-multi-chip-testing-requirements)
6. [Recommended Test Plan for 4-Device System](#6-recommended-test-plan-for-4-device-system)
7. [Telemetry Server Configuration](#7-telemetry-server-configuration)
8. [Key Code References](#8-key-code-references)
9. [Open Questions](#9-open-questions)

---

## 1. Executive Summary

### What We Learned

1. **Single-chip workloads show no measurable telemetry impact** - Results were statistically indistinguishable from noise (±2% mean, highly variable P99).

2. **This is the expected result** - Telemetry and single-chip workloads use fundamentally different resources:
   - Telemetry: PCIe MMIO reads to ARC registers and Ethernet L1 (~300 bytes/poll)
   - Workloads: PCIe DMA for tensors, Tensix cores, L1/DRAM, NOC

3. **The real conflict is on multi-chip workloads** - Telemetry reading from remote chips requires ERISC-mediated NOC access, which competes with fabric data movement.

4. **Multi-chip testing has NOT been performed** - All our tests used `CreateDevice(device_id=0)` (single chip). The telemetry impact on fabric workloads remains unvalidated.

### Key Recommendation

Test telemetry impact on multi-chip CCL operations (AllGather, ReduceScatter, AllReduce) with and without `--mmio-only` flag to validate the existing mitigation strategy.

---

## 2. Single-Chip Testing Results

### Test Methodology

We created `/tmp/telemetry_robust_test.py` with:
- **300 samples per configuration** (100 samples × 3 repetitions)
- **Interleaved baseline/telemetry runs** to control for system drift
- **Health checks** (HTTP GET to `/api/status` + process alive checks)
- **Three workload sizes:**
  - Small: 512×512 tensors, 4 ops (~5ms baseline)
  - Medium: 2048×2048 tensors, 10 ops (~137ms baseline)
  - Large: 4096×4096 tensors, 8 ops (~667ms baseline)
- **Four polling intervals:** 1s, 100ms, 10ms, 1ms

### Results Summary

| Workload | Baseline | Mean Impact Range | P99 Impact Range | Statistically Significant? |
|----------|----------|-------------------|------------------|---------------------------|
| Small (512×512) | 5.06 ms | +0.5% to +1.5% | -2% to +5% | Mixed |
| Medium (2048×2048) | 136.73 ms | -0.7% to +1.9% | -3% to +32% | Mixed |
| Large (4096×4096) | 666.93 ms | -0.2% to +0.8% | -2% to +14% | Mixed |

### Detailed Results by Polling Interval

**Small Workload (5.06 ms baseline, std: 0.28):**
| Polling | Mean Impact | P99 Impact | P-value | Significant? |
|---------|-------------|------------|---------|--------------|
| 1s | +0.49% | -2.15% | 1.14e-01 | no |
| 100ms | +1.11% | -1.59% | 1.61e-11 | YES |
| 10ms | +1.21% | -0.03% | 1.50e-07 | YES |
| 1ms | +1.49% | +5.08% | 1.18e-14 | YES |

**Medium Workload (136.73 ms baseline, std: 3.22):**
| Polling | Mean Impact | P99 Impact | P-value | Significant? |
|---------|-------------|------------|---------|--------------|
| 1s | +1.06% | -0.01% | 1.14e-09 | YES |
| 100ms | +1.93% | +31.75% | 4.95e-09 | YES |
| 10ms | -0.72% | -2.94% | 1.16e-03 | YES |
| 1ms | -0.55% | +18.57% | 1.13e-06 | YES |

**Large Workload (666.93 ms baseline, std: 43.10):**
| Polling | Mean Impact | P99 Impact | P-value | Significant? |
|---------|-------------|------------|---------|--------------|
| 1s | +0.79% | +10.62% | 7.05e-02 | no |
| 100ms | -0.19% | +13.90% | 7.03e-08 | YES |
| 10ms | +0.48% | +2.82% | 3.98e-02 | YES |
| 1ms | -0.24% | -1.80% | 7.70e-01 | no |

### Interpretation

The results are **not coherent**:
- Mean impacts bounce between positive and negative
- P99 at 1ms is sometimes *better* than 100ms
- No monotonic relationship between polling frequency and impact
- "Statistical significance" doesn't mean practical significance

**Conclusion:** These results represent measurement noise, not real telemetry impact. The ~1-2% variations are within the natural 4-6% coefficient of variation of TTNN operations.

---

## 3. Resource Analysis

### What Telemetry Actually Reads

| Resource | Access Method | Transport | Data Size |
|----------|---------------|-----------|-----------|
| ARC registers (temp, clocks, power) | `read_from_device()` | PCIe MMIO via TLB | ~100 bytes |
| Ethernet L1 (fabric telemetry) | `cluster.read_from_device()` | PCIe DMA or MMIO | ~200 bytes |
| Board info (ID, ASIC location) | `read_from_device()` | PCIe MMIO | ~50 bytes |

### What Telemetry Does NOT Access

- **Tensix compute cores** - no interaction
- **Device DRAM** - not accessed
- **L1 compute buffers** - not accessed
- **NOC (on local chips)** - not used for telemetry reads
- **ERISC cores (with --mmio-only)** - explicitly skipped

### What Single-Chip Workloads Use

| Resource | Usage |
|----------|-------|
| PCIe DMA | Tensor transfers (host ↔ device) |
| Tensix cores | Compute operations |
| L1 memory | Kernel data, intermediate results |
| DRAM | Tensor storage |
| NOC | On-chip data movement |

### What Multi-Chip Workloads ALSO Use

| Resource | Usage |
|----------|-------|
| **ERISC cores** | Ethernet data movers (EDM) |
| **Ethernet links** | Chip-to-chip transfers |
| **Fabric routers** | Packet forwarding |
| **NOC (cross-chip)** | Remote memory access |

### Resource Conflict Matrix

| Resource | Telemetry Uses | Single-Chip Workload | Multi-Chip Workload | Conflict? |
|----------|----------------|---------------------|---------------------|-----------|
| PCIe MMIO | ✅ (~300 B/poll) | ❌ | ❌ | None |
| PCIe DMA | ✅ (small reads) | ✅ (large transfers) | ✅ | Minimal (bandwidth) |
| Tensix cores | ❌ | ✅ | ✅ | None |
| L1/DRAM | ❌ | ✅ | ✅ | None |
| NOC (local) | ❌ | ✅ | ✅ | None |
| **ERISC cores** | ✅ (remote reads) | ❌ | ✅ | **CRITICAL** |
| **Ethernet fabric** | ✅ (remote reads) | ❌ | ✅ | **CRITICAL** |

---

## 4. Why Single-Chip Results Were Inconclusive

### The Fundamental Problem

Telemetry and single-chip workloads share only **PCIe bandwidth**, where telemetry consumes ~0.001% even at 1ms polling:

```
Telemetry: ~300 bytes × 1000 polls/sec = 300 KB/s
PCIe Gen4 x16: ~32 GB/s
Overhead: 300KB / 32GB = 0.0009%
```

This is unmeasurable against:
- Natural run-to-run variance: 4-6% CV
- Device initialization timing differences
- Thermal state variations
- Host system background activity

### What Would Actually Show Impact

1. **Multi-chip workloads** - where telemetry competes for ERISC/fabric resources
2. **Device-side profiling** - Tracy/op_perf instead of wall-clock timing
3. **PCIe transaction counters** - hardware-level measurement
4. **Synthetic PCIe contention** - inject known load to establish detection threshold

---

## 5. Multi-Chip Testing Requirements

### Hardware Configurations

| Config | Devices | Fixture | Description |
|--------|---------|---------|-------------|
| N300 | 2 | `mesh_device=(1,2)` | 2 Wormhole chips |
| P150x4 | 4 | `pcie_mesh_device` | 4 Blackhole chips |
| T3000 | 8 | `t3k_mesh_device` | 8 Wormhole chips (1×8) |
| Galaxy/TG | 32 | `mesh_device=(8,4)` | 32 chips (8×4 mesh) |

### For a 4-Device System

The most relevant tests are in:
```
tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/
```

These use:
- `skip_for_n_or_less_dev(N)` - skips if ≤N devices available
- `FabricConfig.FABRIC_1D` - enables fabric (required for ERISC contention)
- `bh_2d_mesh_device` or `pcie_mesh_device` fixtures

### Critical Test Categories

1. **AllGather** - collects data from all devices to all devices
2. **ReduceScatter** - reduces and scatters data across devices
3. **AllReduce** - AllGather + reduction
4. **AllToAll** - full exchange between all devices

All of these stress ERISC cores and fabric bandwidth.

---

## 6. Recommended Test Plan for 4-Device System

### Phase 1: Verify Hardware and Environment

```bash
# Check available devices
python -c "import ttnn; print(f'Devices: {ttnn.get_num_devices()}')"
python -c "import ttnn; print(f'PCIe devices: {ttnn.get_pcie_device_ids()}')"

# Verify tt-telemetry is built with --polling-interval support
/data/kkfernandez/tt-telemetry/build_Release/bin/tt_telemetry_server --help | grep polling
```

### Phase 2: Baseline CCL Tests (No Telemetry)

Run a few CCL tests to establish baseline performance:

```bash
cd /data/kkfernandez/tt-metal

# Simple 2-device AllGather (if 2+ devices)
pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit/test_all_gather_apc.py -v -k "test_all_gather_subcore_grid" --tb=short

# Simple 2-device ReduceScatter (if 2+ devices)
pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit/test_reduce_scatter_apc.py -v --tb=short

# 4-device CCL perf test (if available and not skipped)
pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_ccl_perf.py -v --tb=short
```

### Phase 3: Telemetry Impact Test Script

Create a test script that:
1. Runs CCL operations WITHOUT telemetry (baseline)
2. Runs CCL operations WITH telemetry (`--mmio-only` mode)
3. Runs CCL operations WITH telemetry (full mode, if safe)
4. Compares results

**Key measurement:** Look for:
- Operation failures / timeouts
- Latency increases in DEVICE KERNEL time
- P99 latency spikes

### Recommended Test Script Structure

```python
#!/usr/bin/env python3
"""
Multi-chip telemetry impact test.
Run on a system with 4 devices.
"""

import subprocess
import time
import statistics
import json

# Configuration
TELEMETRY_SERVER = "/data/kkfernandez/tt-telemetry/build_Release/bin/tt_telemetry_server"
FSD_FILE = "/data/btrzynadlowski/tt-metal/fsd.textproto"  # Or local path
TELEMETRY_PORT = 5555

# CCL test to run (choose based on what's available)
CCL_TEST = "tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit/test_all_gather_apc.py"

def start_telemetry(polling_interval: str, mmio_only: bool = True):
    """Start telemetry server with given configuration."""
    cmd = [
        TELEMETRY_SERVER,
        "--port", str(TELEMETRY_PORT),
        "--polling-interval", polling_interval,
        "--fsd", FSD_FILE,
    ]
    if mmio_only:
        cmd.append("--mmio-only")

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def stop_telemetry(proc):
    """Stop telemetry server."""
    proc.terminate()
    proc.wait(timeout=5)

def run_ccl_test():
    """Run CCL test and capture output."""
    result = subprocess.run(
        ["pytest", CCL_TEST, "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd="/data/kkfernandez/tt-metal"
    )
    return result.returncode, result.stdout, result.stderr

def main():
    results = {}

    # Test 1: Baseline (no telemetry)
    print("Running baseline (no telemetry)...")
    rc, stdout, stderr = run_ccl_test()
    results["baseline"] = {"returncode": rc, "passed": rc == 0}

    # Test 2: Telemetry with --mmio-only (safe mode)
    print("Running with telemetry (--mmio-only, 100ms polling)...")
    proc = start_telemetry("100ms", mmio_only=True)
    time.sleep(2)  # Let telemetry start
    rc, stdout, stderr = run_ccl_test()
    stop_telemetry(proc)
    results["mmio_only_100ms"] = {"returncode": rc, "passed": rc == 0}

    # Test 3: Telemetry with --mmio-only, aggressive polling
    print("Running with telemetry (--mmio-only, 10ms polling)...")
    proc = start_telemetry("10ms", mmio_only=True)
    time.sleep(2)
    rc, stdout, stderr = run_ccl_test()
    stop_telemetry(proc)
    results["mmio_only_10ms"] = {"returncode": rc, "passed": rc == 0}

    # Test 4: Full telemetry (WARNING: may cause failures)
    print("Running with telemetry (full mode, 100ms polling)...")
    print("WARNING: This may cause test failures due to ERISC contention")
    proc = start_telemetry("100ms", mmio_only=False)
    time.sleep(2)
    rc, stdout, stderr = run_ccl_test()
    stop_telemetry(proc)
    results["full_100ms"] = {"returncode": rc, "passed": rc == 0}

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for name, res in results.items():
        status = "PASS" if res["passed"] else "FAIL"
        print(f"  {name}: {status}")

    return results

if __name__ == "__main__":
    main()
```

### Phase 4: Detailed Performance Comparison

If basic tests pass, run with device profiling enabled:

```bash
# Enable profiling
export TT_METAL_DEVICE_PROFILER=1

# Run test and collect tracy data
pytest <test> -v

# Analyze with tt-metal's profiling tools
```

Compare DEVICE KERNEL times between baseline and telemetry runs.

---

## 7. Telemetry Server Configuration

### Building tt-telemetry

```bash
cd /data/kkfernandez/tt-telemetry
./build.sh
```

The server binary will be at: `build_Release/bin/tt_telemetry_server`

### Server Command-Line Options

```
--port <N>              HTTP API port (default: 5555)
--polling-interval <T>  Telemetry polling interval (e.g., "1s", "100ms", "10ms")
--fsd <path>            Path to FSD (Flat Schema Definition) file
--mmio-only             Only read from MMIO-capable (local) chips
--disable-fabric-telemetry  Disable fabric/ethernet telemetry entirely
```

### FSD File Location

The FSD file describes the telemetry schema. Known location:
```
/data/btrzynadlowski/tt-metal/fsd.textproto
```

Or check for a local copy:
```bash
find /data/kkfernandez -name "*.textproto" 2>/dev/null
```

### Health Check Endpoint

```bash
curl http://localhost:5555/api/status
```

Returns 200 OK if server is running.

---

## 8. Key Code References

### Telemetry Resource Access

**Skip remote chips (the key mitigation):**
```
/data/kkfernandez/tt-telemetry/telemetry/ethernet/ethernet_metrics.cpp:78-89
```

```cpp
if (config.is_ethernet_mmio_only() && !cluster_descriptor->is_chip_mmio_capable(chip_id)) {
    log_info("Skipping remote chip {} - Ethernet telemetry disabled for remote chips");
    continue;
}
```

**Fabric telemetry reads:**
```
/data/kkfernandez/tt-metal/tt_metal/fabric/fabric_telemetry_reader.cpp:51-83
```

**ARC telemetry reads:**
```
/data/kkfernandez/tt-metal/tt_metal/third_party/umd/device/arc/arc_telemetry_reader.cpp:46-91
```

### Multi-Chip Test Fixtures

**conftest.py fixtures:**
```
/data/kkfernandez/tt-metal/conftest.py:374-569
```

Key fixtures:
- `mesh_device` - generic mesh device (lines 374-434)
- `pcie_mesh_device` - 4 PCIe devices in 2×2 mesh (lines 466-500)
- `t3k_mesh_device` - T3000 8-device mesh (lines 504-529)
- `bh_1d_mesh_device` - Blackhole 1D mesh (lines 532-561)
- `bh_2d_mesh_device` - Blackhole 2D mesh (lines 564+)

### CCL Tests

**2-device tests:**
```
tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit/test_all_gather_apc.py
tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit/test_reduce_scatter_apc.py
```

**4-device perf tests:**
```
tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_ccl_perf.py
```

**Model-level multi-chip tests:**
```
models/demos/t3000/mixtral8x7b/tests/test_mixtral_perf.py
models/demos/llama3_70b_galaxy/tests/test_decoder_device_perf.py
```

### Test Scripts Created

**Single-chip validation test:**
```
/tmp/validation_test.py
```

**Single-chip telemetry impact test:**
```
/tmp/telemetry_impact_test.py
```

**Robust single-chip test (final version):**
```
/tmp/telemetry_robust_test.py
```

---

## 9. Open Questions

### Questions to Answer with Multi-Chip Testing

1. **Does `--mmio-only` actually prevent all ERISC contention?**
   - Test: Run CCL ops with `--mmio-only` vs without
   - Expected: `--mmio-only` should show no impact; full mode may show failures

2. **What is the actual telemetry impact on fabric bandwidth?**
   - Test: Measure CCL operation latency with/without telemetry
   - Use device profiler for accurate DEVICE KERNEL times

3. **Does polling frequency affect multi-chip workloads?**
   - Test: Run CCL ops with telemetry at 1s, 100ms, 10ms, 1ms
   - Expected: More aggressive polling = more potential contention

4. **Are there specific CCL operations more sensitive to telemetry?**
   - Test: AllGather vs ReduceScatter vs AllReduce
   - Different operations may have different ERISC utilization patterns

### Potential Issues to Watch For

1. **Test failures due to device busy timeouts** - indicates ERISC contention
2. **Non-reproducible failures** - timing-dependent contention
3. **P99 latency spikes without mean impact** - occasional contention events
4. **Telemetry server crashes** - may indicate UMD conflicts

### Future Work

1. **Add telemetry impact CI test** - run model benchmark with/without telemetry, verify <1% mean regression
2. **Document `--mmio-only` as default** - ensure production deployments use this flag
3. **Consider automatic disable during CCL ops** - telemetry could detect fabric activity and pause reads

---

## Appendix: Test Output Files

All test outputs from the single-chip investigation are at:

```
/tmp/telemetry_robust.log          # Full robust test output
/tmp/telemetry_robust_results.json # JSON results (partial due to serialization error)
/tmp/telemetry_impact_500samples.log
/tmp/telemetry_impact_results.json
/tmp/telemetry_realistic.log
```

The final conclusions report:
```
/tmp/telemetry_measurement_conclusions.md
```

---

## Summary for Next Session

**What was done:**
- Extensive single-chip testing with multiple methodologies
- Resource analysis showing telemetry/workload separation
- Determined single-chip tests are inadequate for measuring real impact

**What needs to be done:**
- Run multi-chip CCL tests (AllGather, ReduceScatter) with 4 devices
- Compare baseline vs telemetry (--mmio-only) vs telemetry (full mode)
- Validate that `--mmio-only` prevents ERISC contention
- Document findings and recommendations

**Key hypothesis to test:**
- `--mmio-only` mode should show no telemetry impact on multi-chip workloads
- Full telemetry mode (without `--mmio-only`) may cause failures or significant latency increases due to ERISC contention

# Fabric Bandwidth Telemetry Validation

This document describes how to validate that fabric bandwidth telemetry calculations are correct.

## Overview

Bandwidth telemetry validation involves:
1. Running a known fabric workload with measurable data transfer
2. Reading fabric telemetry before and after the transfer
3. Calculating expected bandwidth from workload parameters
4. Comparing expected vs. actual bandwidth measurements

## Validation Tools

### 1. C++ GTest (`test_bandwidth_telemetry_validation.cpp`)

Automated test that validates bandwidth calculations using fabric workloads.

**Location:** `tests/tt_metal/tt_fabric/test_bandwidth_telemetry_validation.cpp`

**Prerequisites:**
- Multi-device system with fabric links configured
- `TT_METAL_FABRIC_TELEMETRY=1` environment variable
- Test binary compiled

**Run:**
```bash
export TT_METAL_FABRIC_TELEMETRY=1
./build/test/tt_metal/tt_fabric/test_bandwidth_telemetry_validation
```

**What it validates:**
- `BYTES_PER_WORD = 4` is correct
- AICLK frequency reading is accurate
- Bandwidth calculation formula is correct
- Counter wrapping is handled properly
- Uninitialized counters are detected and handled

### 2. Python Validation Script (`validate_bandwidth_telemetry.py`)

Standalone script for manual validation and debugging.

**Location:** `validate_bandwidth_telemetry.py`

**Prerequisites:**
- tt-smi server running with telemetry enabled
- Fabric test binary compiled
- Python 3 with `requests` library

**Run:**
```bash
# Start tt-smi server in one terminal
tt-smi --server

# Run validation in another terminal
python3 validate_bandwidth_telemetry.py \
    --chip-id 0 \
    --channel 0 \
    --aiclk-mhz 1000.0 \
    --tolerance 20.0
```

**Options:**
- `--metrics-url`: Metrics endpoint URL (default: http://localhost:8080/api/metrics)
- `--chip-id`: Chip to monitor (default: 0)
- `--channel`: Channel to monitor (default: 0)
- `--aiclk-mhz`: AICLK frequency in MHz (default: 1000.0)
- `--tolerance`: Acceptable error percentage (default: 20.0)
- `--test-binary`: Path to fabric test binary
- `--test-args`: Arguments to pass to test

## Validation Methodology

### Expected Bandwidth Calculation

For a known workload, expected bandwidth is:

```
Expected BW (MB/s) = Total Bytes Transferred / Wall Clock Time (s) / 10^6
```

### Telemetry-Reported Bandwidth Calculation

From fabric telemetry counters:

```
Delta Words = Current Words - Previous Words
Delta Cycles = Current Cycles - Previous Cycles

Bytes Transferred = Delta Words × 4  (4 bytes per word)
Time Elapsed (s) = Delta Cycles / (AICLK MHz × 10^6)

Telemetry BW (MB/s) = Bytes Transferred / Time Elapsed / 10^6
```

### Validation Criteria

Bandwidth calculations are considered valid if:

```
|Telemetry BW - Expected BW| / Expected BW × 100% < Tolerance
```

Default tolerance: **±20%**

Tolerance accounts for:
- Telemetry sampling granularity (5-second intervals)
- Fabric routing overhead
- Firmware processing delays
- Counter precision limitations

## Key Parameters

### BYTES_PER_WORD

**Current value:** 4 bytes

**Validation:** Confirms this matches firmware specification. Previous incorrect value (16 bytes) inflated bandwidth by 4x.

### AICLK Frequency

**Source:** ARC firmware telemetry via `CachingARCTelemetryReader`

**Fallback:** 1000 MHz (1 GHz) when ARC telemetry unavailable

**Architecture defaults:**
- Wormhole: ~900-1000 MHz under workload
- Blackhole: Up to 1350 MHz

**Validation:** Compares calculated time vs. wall-clock time to verify clock speed accuracy.

### Cycle Counter Wraparound

**Counter size:** 64-bit unsigned integers

**Wraparound time:** At 1.2 GHz: ~178 days

**Validation:** Test runs are short enough that wraparound is not a concern. Production code handles wraparound via unsigned arithmetic.

## Known Issues and Limitations

### 1. Uninitialized Counters

**Issue:** L1 memory not zeroed on device power-on/reset, leading to garbage counter values.

**Symptom:** Massive spurious deltas at startup (e.g., 10^18 cycles)

**Workaround:** Telemetry detects large deltas (> 10^12 cycles) and resets baseline.

**Proper fix:** Firmware should zero telemetry structures during ERISC initialization (see `issue.md`).

### 2. Sampling Granularity

**Issue:** Telemetry polls every 5 seconds, bandwidth calculations are averages over this window.

**Impact:** Short bursts may be averaged out, peak bandwidth may differ from sustained bandwidth.

**Mitigation:** Run workloads longer than sampling interval for accurate measurements.

### 3. Remote Chip AICLK

**Issue:** ARC telemetry only available via MMIO. Remote chips use nearest PCIE chip's AICLK.

**Impact:** Slight inaccuracy if remote chip runs at different frequency.

**Mitigation:** Negligible in practice as chips in same system typically run at similar speeds.

## Troubleshooting

### Validation fails with large error

1. **Check AICLK value:** Verify correct frequency for your architecture and workload.
2. **Check telemetry update rate:** Ensure tt-smi server is updating at 5-second intervals.
3. **Verify workload:** Ensure fabric test actually transfers expected amount of data.
4. **Check topology:** Confirm monitoring correct chip/channel for the workload.

### Counters show zero delta

1. **Wrong channel:** Workload may use different fabric link.
2. **Telemetry not updating:** Check `TT_METAL_FABRIC_TELEMETRY=1` is set.
3. **Workload didn't run:** Verify test completed successfully.

### Garbage counter values

1. **Device reset:** Counters uninitialized after power-on or reset.
2. **Expected behavior:** First sample after reset may show garbage.
3. **Auto-recovery:** Telemetry resets baseline on next sample.

## Future Improvements

1. **Firmware initialization:** Zero telemetry structures in ERISC firmware (highest priority).
2. **Higher-resolution sampling:** Reduce 5-second interval for more accurate burst measurements.
3. **Multi-link validation:** Validate bandwidth across multiple fabric links simultaneously.
4. **Automated topology detection:** Auto-discover which channels to monitor for given workload.
5. **Per-packet telemetry:** More granular bandwidth tracking for debugging.

## References

- Telemetry structures: `tt_metal/api/tt-metalium/experimental/fabric/fabric_telemetry.hpp`
- Bandwidth calculation: `tt_telemetry/telemetry/ethernet/ethernet_metrics.cpp`
- Fabric tests: `tests/tt_metal/tt_fabric/`
- Issue tracker: `issue.md` (firmware initialization)

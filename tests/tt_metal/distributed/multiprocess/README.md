# Visible Devices Multi-Process Tests

This directory contains tests for validating the `TT_METAL_VISIBLE_DEVICES` environment variable functionality in a distributed multi-process context.

## Overview

This basic test suite validates when `TT_METAL_VISIBLE_DEVICES` is set, the process only sees the PCIe devices exposed to it.
This means on a T3000, you can effectively:
1) Emulate a single N300 process running tt-metal, for every N300 board independently
2) Expose multiple PCIe devices through `TT_METAL_VISIBLE_DEVICES` and test 2x2 mesh configuration
3) Emulate multi-host configuration by simultaneously launching multiple processes working on independent parts of the available system mesh.

## Test Script: `run_visible_devices_mp_tests.sh`

### Purpose
Runs the `distributed_multiprocess_tests` executable with various device configurations to validate visible devices functionality.

### Device Configurations Tested
- Single devices: `"0"`, `"1"`, `"2"`, `"3"`
- Device pairs: `"0,1"`, `"0,3"`, `"1,2"`, `"2,3"`

### Usage
```bash
./run_visible_devices_mp_tests.sh
```

### Prerequisites
- Build the tests first: `./build_metal.sh --debug --build-tests`
- Requires `mpirun` to be installed and available in PATH
- Requires tt-metal devices to be available on the system

### Output
The script will:
- Display progress for each device configuration
- Show ✓ for passed tests and ✗ for failed tests
- Exit with code 0 if all tests pass, 1 if any test fails

### Example Output
```
Testing TT_METAL_VISIBLE_DEVICES functionality with distributed_mp_unit_tests
============================================================================

Testing with TT_METAL_VISIBLE_DEVICES="0"
------------------------------------------------
✓ Test passed for configuration: 0

Testing with TT_METAL_VISIBLE_DEVICES="0,1"
------------------------------------------------
✓ Test passed for configuration: 0,1

...

All tests passed!
```

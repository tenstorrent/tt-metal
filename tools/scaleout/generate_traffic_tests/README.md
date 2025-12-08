# Generate Traffic Tests Tool

This tool automatically generates traffic test YAML files from cabling descriptors for fabric testing.

**Note:** Tests are generated from **easiest to hardest** (top to bottom in the YAML output), ensuring quick validation tests run first before more complex stress tests.

## Building

```bash
cmake --build build --target generate_traffic_tests
```

## Usage

```bash
./build/tools/scaleout/generate_traffic_tests --cabling-descriptor-path <path> [OPTIONS]
```

### Required Arguments

- `--cabling-descriptor-path PATH`, `-c PATH` - Path to the cabling descriptor textproto file

### Optional Arguments

- `--output-path PATH`, `-o PATH` - Path to output traffic test YAML file (default: `traffic_tests.yaml`)
- `--mgd-output-path PATH`, `-m PATH` - Path to auto-generate MGD file
- `--existing-mgd-path PATH`, `-e PATH` - Use existing MGD instead of generating
- `--profile PROFILE`, `-p PROFILE` - Test profile (see below)
- `--name-prefix PREFIX`, `-n PREFIX` - Prefix for generated test names
- `--flow-control` - Include flow control tests
- `--no-sync` - Disable sync for tests
- `--skip PLATFORMS` - Platforms to skip (comma-separated)
- `--verbose`, `-v` - Enable verbose output
- `--help`, `-h` - Print help message

### Test Profiles

| Profile | Description | Tests Generated |
|---------|-------------|-----------------|
| `sanity` (default) | Quick functional validation | SimpleUnicast, InterMesh, AllToAll |
| `stress` | High-volume stress testing | + RandomPairing, AllToOne, FlowControl, SequentialAllToAll |
| `benchmark` | Performance measurement | Optimized for throughput/latency measurement |
| `coverage` | Full test coverage | All patterns, all noc types, all combinations |

## Examples

### Generate sanity tests for Dual T3K

```bash
./build/tools/scaleout/generate_traffic_tests \
    --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/dual_t3k.textproto \
    --output-path tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_dual_t3k_auto.yaml \
    --mgd-output-path tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_auto_mgd.textproto \
    --verbose
```

### Generate stress tests for 16-node ClosetBox

```bash
./build/tools/scaleout/generate_traffic_tests \
    --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto \
    --output-path test_16_n300_stress.yaml \
    --mgd-output-path 16_n300_mgd.textproto \
    --profile stress \
    --name-prefix "ClosetBox_"
```

### Use existing MGD file

```bash
./build/tools/scaleout/generate_traffic_tests \
    --cabling-descriptor-path my_cluster.textproto \
    --existing-mgd-path existing_mgd.textproto \
    --profile coverage
```

### Skip certain platforms

```bash
./build/tools/scaleout/generate_traffic_tests \
    --cabling-descriptor-path my_cluster.textproto \
    --output-path tests.yaml \
    --skip "GALAXY,BLACKHOLE_GALAXY"
```

## Generated Test Order (Easiest to Hardest)

1. **SimpleUnicast** - Basic point-to-point within single mesh
2. **InterMeshUnicast** - Tests each inter-mesh connection
3. **AllToAllUnicast** - High-level all-to-all pattern with parametrization
4. **RandomPairing** - Random device pairing (stress/coverage only)
5. **AllToOne** - Convergence test, many senders to single receiver (stress/coverage only)
6. **FlowControlStress** - High packet count with flow control (stress/coverage only)
7. **SequentialAllToAll** - Sequential pair testing, max iterations (stress/coverage only)

## Programmatic API

The tool also provides a C++ API for programmatic usage:

```cpp
#include <generate_traffic_tests/generate_traffic_tests.hpp>

using namespace tt::scaleout_tools;

// Use preset profiles
auto config = get_stress_config();
config.mgd_output_path = "output.mgd.textproto";

generate_traffic_tests(
    "cabling_descriptor.textproto",
    "traffic_tests.yaml",
    config,
    /*verbose=*/true
);

// Or generate YAML string directly
auto yaml = generate_traffic_tests_yaml(cluster_desc, config);
```

## Integration with MGD Generator

This tool integrates with the `generate_mgd` tool:

1. Both use the same cabling descriptor as input
2. Traffic tests can auto-generate the MGD file (`--mgd-output-path`)
3. Or reference an existing MGD (`--existing-mgd-path`)

Typical workflow:
```bash
# Option 1: Generate both at once
./generate_traffic_tests -c cluster.textproto -o tests.yaml -m cluster_mgd.textproto

# Option 2: Generate separately
./generate_mgd -c cluster.textproto -o cluster_mgd.textproto
./generate_traffic_tests -c cluster.textproto -o tests.yaml -e cluster_mgd.textproto
```

## Supported Node Types

Same as `generate_mgd`:
- N300_LB_DEFAULT, N300_QB_DEFAULT (Wormhole)
- WH_GALAXY, WH_GALAXY_X_TORUS, WH_GALAXY_Y_TORUS, WH_GALAXY_XY_TORUS (Wormhole Galaxy)
- P150_LB, P150_QB_AE_DEFAULT, P300_QB_GE (Blackhole)
- BH_GALAXY, BH_GALAXY_X_TORUS, BH_GALAXY_Y_TORUS, BH_GALAXY_XY_TORUS (Blackhole Galaxy)

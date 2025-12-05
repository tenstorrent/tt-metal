# Generate Traffic Tests Tool

Generates traffic test YAML files from cabling descriptors for fabric testing.

## Building

```bash
cmake --build build --target generate_traffic_tests
```

## Usage

```bash
./build/tools/scaleout/generate_traffic_tests -c <cabling_descriptor> -o <output> [OPTIONS]
```

### Required

- `-c, --cabling-descriptor-path PATH` - Cabling descriptor textproto file

### Options

- `-o, --output-path PATH` - Output YAML file (default: `traffic_tests.yaml`)
- `-m, --mgd-output-path PATH` - Auto-generate MGD file
- `-e, --existing-mgd-path PATH` - Use existing MGD instead of generating
- `-p, --profile PROFILE` - Test profile: `sanity`, `stress`, `benchmark`
- `-n, --name-prefix PREFIX` - Prefix for test names
- `--packets N` - Packets per sender
- `--sizes S1,S2,...` - Packet sizes
- `--noc-types T1,T2,...` - NoC types (e.g., `unicast_write,atomic_inc`)
- `--no-sync` - Disable sync
- `--skip PLATFORMS` - Platforms to skip (comma-separated)
- `-v, --verbose` - Verbose output

### Test Profiles

| Profile | Description | Runtime |
|---------|-------------|---------|
| `sanity` | Quick validation | ~1 min |
| `stress` | Thorough testing with flow control | ~5-15 min |
| `benchmark` | Performance measurement, varied sizes | ~2-5 min |

## Examples

### Basic sanity tests

```bash
./build/tools/scaleout/generate_traffic_tests \
    -c tools/tests/scaleout/cabling_descriptors/dual_t3k.textproto \
    -o tests.yaml \
    -m mgd.textproto \
    -v
```

### Stress tests with custom config

```bash
./build/tools/scaleout/generate_traffic_tests \
    -c cluster.textproto \
    -o stress_tests.yaml \
    -m cluster_mgd.textproto \
    -p stress \
    --packets 5000 \
    --sizes 1024,2048,4096
```

### Use existing MGD

```bash
./build/tools/scaleout/generate_traffic_tests \
    -c cluster.textproto \
    -e existing_mgd.textproto \
    -o tests.yaml
```

## Generated Tests (by complexity)

1. **SimpleUnicast** - Basic point-to-point within single mesh
2. **InterMeshUnicast** - Tests each inter-mesh connection
3. **AllToAll** - All devices to all devices
4. **RandomPairing** - Random device pairing (stress only)
5. **AllToOne** - Many senders to one receiver (stress only)
6. **FlowControl** - High packet count with flow control (stress only)
7. **SequentialAllToAll** - Sequential pair testing (stress only)

## C++ API

```cpp
#include <generate_traffic_tests/generate_traffic_tests.hpp>

using namespace tt::scaleout_tools;

auto config = get_stress_config();
config.mgd_output_path = "output.mgd.textproto";

generate_traffic_tests("cabling.textproto", "tests.yaml", config, true);
```

## Supported Node Types

- N300_LB_DEFAULT, N300_QB_DEFAULT (Wormhole)
- WH_GALAXY, WH_GALAXY_X_TORUS, WH_GALAXY_Y_TORUS, WH_GALAXY_XY_TORUS
- P150_LB, P150_QB_AE_DEFAULT, P300_QB_GE (Blackhole)
- BH_GALAXY, BH_GALAXY_X_TORUS, BH_GALAXY_Y_TORUS, BH_GALAXY_XY_TORUS

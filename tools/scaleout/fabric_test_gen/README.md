

# Fabric Test Generator

Generates parametric fabric test configurations from Mesh Graph Descriptor (MGD) `.textproto` files. This tool parses topology information from MGD files and creates comprehensive test suites for validating fabric connectivity, latency, and bandwidth across device meshes.

## Quick Start

### Basic Usage
```bash
python3 main.py -f path/to/mgd.textproto
```

This generates `test_fabric_parametrized.yaml` with tests for the `[0,0]` → `[0,0]` core pair.

### Run Generated Tests
```bash
# Generate test config
python3 tools/scaleout/fabric_test_gen/main.py \
  -f tt_metal/fabric/mesh_graph_descriptors/n300_2x2_mesh_graph_descriptor.textproto \
  -o my_fabric_tests.yaml

# Run fabric tests with generated config
./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
  --test_config my_fabric_tests.yaml
```

## Command Reference

### Syntax
```bash
python3 main.py -f <MGD_FILE> [OPTIONS]
```

### Required Arguments
- `-f, --filename`: Path to the `.textproto` Mesh Graph Descriptor file

### Optional Arguments

#### Output
- `-o, --output`: Output YAML filename (default: `test_fabric_parametrized.yaml`)

#### Core Selection
Controls which worker cores are used as source and destination endpoints:

- `--src_core X Y`: Specific source core coordinates
- `--dst_core X Y`: Specific destination core coordinates
- `--sweep_cores`: Enable full sweep over all worker cores

**Core Selection Behavior:**

| Command | Behavior |
|---------|----------|
| _(no args)_ | `[0,0]` → `[0,0]` (single pair) |
| `--src_core 1 2` | `[1,2]` → `[0,0]` |
| `--dst_core 3 4` | `[0,0]` → `[3,4]` |
| `--src_core 1 2 --dst_core 3 4` | `[1,2]` → `[3,4]` |
| `--sweep_cores` | All cores → All cores ⚠️ |
| `--sweep_cores --src_core 1 2` | `[1,2]` → All dst cores |
| `--sweep_cores --dst_core 3 4` | All src cores → `[3,4]` |

⚠️ **Warning**: `--sweep_cores` generates all core pair combinations:
- Wormhole: 10×8 = 80 cores → 6,400 test pairs
- Blackhole: 10×14 = 140 cores → 19,600 test pairs

Multiply by (rows + cols) × test_modes × num_links for total test count.

⚠️ **Note**: Full sweep generates tests for ALL worker core coordinates, including cores that may be occupied by system functions (e.g., dispatch cores). Tests using occupied cores will fail. Use specific core coordinates (`--src_core`/`--dst_core`) to avoid this issue.

#### Test Configuration
- `--ntype`: NOC type (default: `unicast_write`)
  - Choices: `unicast_write`, `atomic_inc`, `fused_atomic_inc`

- `--test_modes`: Test modes to generate (default: both)
  - Choices: `latency`, `bandwidth`, or both (space-separated)
  - Latency tests always use 1 link

- `--num_links`: Link counts to test for bandwidth mode (default: `1 2 3 4`)
  - Only applies to bandwidth tests
  - Accepts space-separated integers

## Examples

```bash
# Default: test [0,0] -> [0,0]
python3 main.py -f cluster.textproto

# Test specific core pair
python3 main.py -f cluster.textproto --src_core 2 3 --dst_core 5 7

# Latency tests only
python3 main.py -f cluster.textproto --test_modes latency

# Bandwidth tests with specific link counts
python3 main.py -f cluster.textproto --test_modes bandwidth --num_links 2 4

# Full core sweep (WARNING: generates many tests)
python3 main.py -f cluster.textproto --sweep_cores

# Atomic increment operation
python3 main.py -f cluster.textproto --ntype atomic_inc --src_core 0 0 --dst_core 9 7
```

## Output Format

The generated YAML file contains a `Tests` array where each entry specifies:
- **name**: Unique test identifier (includes mode and link count)
- **topology**: `Linear` or `Ring` (from MGD)
- **device coordinates**: Source and destination devices in mesh
- **worker cores**: Source and destination core coordinates
- **test parameters**: Operation type, packet size, iteration count
- **link configuration**: Number of fabric links to use

### Example Output
```yaml
Tests:
  - name: x_0_0_1_latency_1links
    latency_test_mode: true
    benchmark_mode: false
    fabric_setup:
      topology: Linear
      num_links: 1
    defaults:
      size: 1024
      num_packets: 10
      ntype: unicast_write
      ftype: unicast
    senders:
      - device: [0, [0, 0]]
        core: [0, 0]
        patterns:
          - destination:
              device: [0, [0, 1]]
              core: [0, 0]
```

## Workflow

1. **Generate tests**: Run this tool with your MGD file and desired parameters
2. **Review output**: Check generated YAML for expected test coverage
3. **Run tests**: Pass YAML to fabric test executable as `--test_config`
4. **Iterate**: Adjust core selection and test modes based on results

## Supported Architectures

- **Wormhole B0**: 10×8 Tensix worker cores
- **Blackhole**: 10×14 Tensix worker cores

Architecture is auto-detected from the MGD `arch` field.

# Multi-Host Test Scripts

This directory contains scripts and configuration files for running multi-host tests on Tenstorrent Galaxy clusters.

## Quick Start

### Prerequisites
- SLURM allocation with 4 nodes
- `TT_METAL_HOME` environment variable set
- Passwordless SSH configured between hosts
- Rankfiles present in the current directory

### Running Tests

**1. Allocate SLURM resources (4 nodes required for both tests):**
```bash
srun --partition=wh_pod_8x16_2 -N 4 --pty /bin/bash
```

**2. Navigate to TT_METAL_HOME:**
```bash
cd $TT_METAL_HOME
```

**3. Run the desired test script:**

For **Dual Galaxy** tests (2 hosts):
```bash
./run_dual_galaxy_tests_commands.sh
```

For **Quad Galaxy** tests (4 hosts):
```bash
./run_quad_galaxy_tests_commands.sh
```

## Files

- `run_dual_galaxy_tests_commands.sh` - Script for 2-host dual galaxy tests
- `run_quad_galaxy_tests_commands.sh` - Script for 4-host quad galaxy tests
- `rankfile_dual_galaxy` - MPI rankfile for 2 hosts
- `rankfile_quad_galaxy` - MPI rankfile for 4 hosts
- `MULTIHOST_TEST_INSTRUCTIONS.md` - Detailed instructions and troubleshooting

## Detailed Documentation

For detailed instructions, troubleshooting, and configuration options, see:
**[MULTIHOST_TEST_INSTRUCTIONS.md](MULTIHOST_TEST_INSTRUCTIONS.md)**

## Test Coverage

### Dual Galaxy Tests
- `test_all_to_all_dispatch_8x8_dual_galaxy`
- `test_all_to_all_combine_8x8_dual_galaxy`

### Quad Galaxy Tests
- `test_all_to_all_dispatch_8x16_quad_galaxy`

## Host Configuration

**Dual Galaxy (2 hosts)**:
- wh-glx-a04u02
- wh-glx-a05u02

**Quad Galaxy (4 hosts)**:
- wh-glx-a04u02
- wh-glx-a05u02
- wh-glx-a05u08
- wh-glx-a05u14

To use different hosts, edit the rankfiles and update the host variables in the scripts.

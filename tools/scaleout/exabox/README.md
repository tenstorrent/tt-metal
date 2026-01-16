# BH Galaxy Exabox Validation

Scripts for validating Blackhole Galaxy Exabox clusters before running workloads.

## Full Hardware Qualification

When bringing up new clusters, operators must ensure that all hardware allocated for the system is stable and usable by software. The workflows outlined in this section are designed to do that.

These workflows also rely on Dockerized Metal Containers, allowing operators to validate the health of the cluster without having to build Metal.

The order matters: physical validation, then dispatch tests, then fabric tests. Physical validation hammers the Ethernet links to make sure they're stable. If links are flaky, fabric tests will just fail with routing errors - you'll waste time debugging software when it's actually a cable. Dispatch tests stress the compute/memory/data-movement blocks to verify chip stability before you try coordinating across the whole cluster.

### Prerequisites

- Clone tt-metal to `/data/<your-username>/` and run all commands from the repo root
- Passwordless SSH to all hosts
- `mpirun` available (Docker-based scripts also need `mpirun-ulfm`)
- FSD file for your cluster topology (on shared mount, should already exist):
  - 8x16: `/data/local-syseng-manual/5x8x16_fsd.textproto`
  - 4x32: `/data/local-syseng-manual/4x4x32_fsd.textproto`

**SSH Setup**

Start an ssh-agent and add your key:
```bash
eval $(ssh-agent)
ssh-add ~/.ssh/<your-key>
```

Verify you can connect without a password:
```bash
ssh <host> hostname
```

**MPI Check**

Verify MPI can reach all hosts (`<hosts>` = comma-separated list, e.g. `host1,host2,host3,host4`):
```bash
mpirun --host <hosts> hostname
```
This should print the hostname of each machine. If it hangs or prompts for a password, fix SSH first.

**Docker Image**

For Galaxy clusters, use:
```
ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:<tag>
```

Options for `<tag>`:
- `latest` - most recent passing build from main
- A specific version tag (e.g., `v0.66.0-dev20260115-28-g6eccf7061a`) - known-good as of Jan 2026

To build an image from your branch, run the [upstream-tests workflow](https://github.com/tenstorrent/tt-metal/actions/workflows/upstream-tests.yaml). The workflow summary shows the image tag once complete.

### Physical Validation

Discovers Ethernet connections, compares against expected topology (FSD), resets chips, sends traffic. Catches bad cables, DRAM failures, unstable links, CRC errors.

```bash
./tools/scaleout/exabox/run_validation_8x16.sh <hosts> <docker-image>
./tools/scaleout/exabox/run_validation_4x32.sh <hosts> <docker-image>
```

Runs 50 loops (reset, discovery, 10 traffic iterations each). Logs go to `validation_output/`.

Why 50? Some links only fail after a few resets - they train fine once but can't do it reliably. Running many iterations catches these. It also burns in the hardware a bit, so weak components show themselves early rather than failing in production.

Analyze results with:
```bash
./tools/scaleout/exabox/analyze_validation_results.sh
```

This tells you how many iterations passed vs failed, and breaks down failures by type (timeouts, missing connections, DRAM issues). If the same channel keeps failing, that's a bad cable. If failures are scattered randomly, might just be a flaky reset - try power cycling and running again.

### Dispatch Tests

Ensures all chips in the cluster are stable. Stress tests the Compute, Memory, and Data-Movement blocks on each chip.

```bash
./tools/scaleout/exabox/run_dispatch_tests.sh <hosts> <docker-image>
```

### Fabric Tests

Stress tests for the TT-Fabric layer. Ensures TT-Fabric SW and FW is compatible with the cluster topology. Also stresses the physical ethernet interconnect, verifying cluster stability. Only run after physical validation passes.

```bash
./tools/scaleout/exabox/run_fabric_tests_8x16.sh <hosts> <docker-image>
./tools/scaleout/exabox/run_fabric_tests_4x32.sh <hosts> <docker-image>
```

## Quick Health Check (For Developers)

For day-to-day use when you just need to verify a cluster is working. Unlike the Docker-based qualification scripts above, these run directly on the host, so you need a local build:

```bash
./create_venv.sh
source python_env/bin/activate
./build_metal.sh --build-metal-tests
```

Then run:
```bash
./tools/scaleout/exabox/recover_8x16.sh <hosts>
# or
./tools/scaleout/exabox/recover_4x32.sh <hosts>
```

Look for `All Detected Links are healthy` in the output.

## Troubleshooting

**Machine reboots during test**: If your machine reboots mid-test, the tests keep running on the other machines. You'll need to manually kill them or wait for them to timeout.

**Missing connections** (`Channel/Port Connections found in FSD but missing in GSD`): Check cables are seated, verify you're using the right FSD file.

**Timeouts** (`Timeout (10000 ms) waiting for physical cores`): Power cycle the cluster.

**DRAM Training Failed**: Hardware issue, contact syseng.

**Tests hanging**: Power cycle.

**Fabric tests failing**: Make sure physical validation passed first.

## Validation Output

Output from `recover_*.sh` and `run_validation_*.sh`:

Healthy:
```
| info     |     Distributed | All Detected Links are healthy.
```

Unhealthy:
```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              FAULTY LINKS REPORT                                                  ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝
Total Faulty Links: 1

Host                Tray  ASIC  Ch   Port ID  Port Type      Unique ID     Retrains    CRC Err       Corrected CW      Uncorrected CW    Mismatch Words  Failure Type                            Pkt Size    Data Size
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bh-glx-c02u02       4     7     0    13       TRACE          0x7dd4ab9...  0x0         0x2fb9872     0x85ad4           24                Retrain + Uncorrected CW + Data Mismatch64 B        367360 B
```

Data Mismatch usually means bad cable or port.

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `recover_*.sh` | Quick reset + 5 traffic iterations |
| `run_validation_*.sh` | Full 50-loop validation |
| `run_dispatch_tests.sh` | Chip stability stress tests |
| `run_fabric_tests_*.sh` | Fabric connectivity tests |
| `analyze_validation_results.sh` | Parse validation logs |
| `mpi-docker` | MPI+Docker wrapper (`--help` for usage) |

## Config Files

**MGDs** are in `tt_metal/fabric/mesh_graph_descriptors/`. Scripts pick the right one automatically.

**FSDs** are environment-specific. Generate with:
```bash
./build/tools/scaleout/run_cabling_generator \
    --cabling <cabling.textproto> \
    --deployment <deployment.textproto> \
    --output <suffix>
```
See `tools/scaleout/README.md` for details.

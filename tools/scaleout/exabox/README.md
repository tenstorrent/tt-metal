# BH Galaxy Exabox Validation

Scripts for validating Blackhole Galaxy Exabox clusters before running workloads.

## Quick Reference

**Last Known-Good Docker Image:**
```
ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:v0.66.0-dev20260115-28-g6eccf7061a
```

## Full Hardware Qualification

When bringing up new clusters, operators must ensure that all hardware allocated for the system is stable and usable by software. The workflows outlined in this section are designed to do that.

These workflows also rely on Dockerized Metal Containers, allowing operators to validate the health of the cluster without having to build Metal.

**Testing order matters:**

1. **Physical Validation** - Hammers the Ethernet links to ensure they're stable. Run this first to catch bad cables early.
2. **Dispatch Tests** - Stress the compute/memory/data-movement blocks to verify chip stability.
3. **Fabric Tests** - Test coordinated workloads across the whole cluster.

If links are flaky, fabric tests will fail with routing errors - you'll waste time debugging software when it's actually a cable. Always validate hardware health before running workloads.

### Prerequisites

- Clone tt-metal to a shared NFS mount accessible from all hosts (e.g., `/data/<your-username>/` on Exabox) and run all commands from the repo root
- Passwordless SSH to all hosts (using `ssh-add` for agent forwarding - see [SSH Setup](#ssh-setup) below)
- **Docker** installed on all hosts (required for Docker-based validation scripts)
- **Tenstorrent-modified OpenMPI v5.0.7** installed on all hosts
  - Install from: https://github.com/tenstorrent/ompi/releases/tag/v5.0.7
  - Provides `mpirun` and `mpirun-ulfm` (required for Docker-based scripts)
  - No manual PATH setup needed - installation handles symlinks automatically
- FSD file for your cluster topology (must be accessible on a shared filesystem)
  - **Note**: The paths below are specific to the current BH Exabox setup. Your system may have different descriptors depending on your topology and deployment.
  - 8x16: `/data/local-syseng-manual/5x8x16_fsd.textproto`
  - 4x32: `/data/local-syseng-manual/4x4x32_fsd.textproto`

**SSH Setup**

On your local host, start an ssh-agent and add your key:
```bash
eval $(ssh-agent)
ssh-add ~/.ssh/<your-key>
```

Log into an intermediate host (jump host) in the cluster:
```bash
ssh -A <jump-host>
```

From the jump host, verify you can connect to all peer hosts without a password:
```bash
ssh <peer-host> hostname
```

**MPI Check**

On the jump host, verify MPI can reach all hosts (`<hosts>` = comma-separated list, e.g. `host1,host2,host3,host4`):
```bash
mpirun --host <hosts> hostname
```
This should print the hostname of each machine. If it hangs or prompts for a password, fix SSH first.

**Docker Check**

Verify Docker is installed and running on all hosts:
```bash
mpirun --host <hosts> docker --version
```
This should print the Docker version on each machine. If any host fails, install Docker on that host.

**Docker Image**

For Galaxy clusters, use:
```
ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:<tag>
```

Options for `<tag>`:
- `latest` - most recent passing build from main (Note: Once quad systems are in CI, this will be consistently reliable. For now, use the known-good version below.)
- `v0.66.0-dev20260115-28-g6eccf7061a` - **Last known-good version** (see [Quick Reference](#quick-reference) at the top)

To build an image from a custom branch (your own branch or one requested from a Metal developer), run the [upstream-tests workflow](https://github.com/tenstorrent/tt-metal/actions/workflows/upstream-tests.yaml). The workflow summary shows the image tag once complete.

### Physical Validation

Discovers Ethernet connections, compares against expected topology (FSD), resets chips, sends traffic. Catches bad cables, DRAM failures, unstable links, CRC errors.

```bash
./tools/scaleout/exabox/run_validation.sh --hosts <hosts> --image <docker-image>
```

Runs 50 loops (reset, discovery, 10 traffic iterations each). Logs go to `validation_output/` in your current directory.

Why 50? Some links only fail after a few resets - they train fine once but can't do it reliably. Running many iterations catches these. It also burns in the hardware a bit, so weak components show themselves early rather than failing in production.

Analyze results with:
```bash
python3 tools/scaleout/exabox/analyze_validation_results.py validation_output/
```

The script parses all `*.log` files in the specified directory and provides:

- **Summary**: Pass/fail counts for each failure category with affected log files
- **Success Rate**: Percentage of healthy iterations (80%+ is typically a pass)
- **Recommendations**: Actionable next steps based on detected failure patterns
- **Cluster Info**: Detected hosts, chips per host, and traffic configuration

### Dispatch Tests

Ensures all chips in the cluster are stable. Stress tests the Compute, Memory, and Data-Movement blocks on each chip.

```bash
./tools/scaleout/exabox/run_dispatch_tests.sh --hosts <hosts> --image <docker-image>
```

Logs are saved to `dispatch_test_logs/` in your current directory.

Analyze results with:
```bash
python3 tools/scaleout/exabox/analyze_dispatch_results.py dispatch_test_logs/<log-file>.log
```

The script parses the test log and provides:

- **Test Results Summary**: MPI processes, total tests run, passed, failed, and skipped counts
- **Test Details**: Lists of passed, failed, and skipped tests with failure details
- **Warnings & Critical Errors**: Deduplicated runtime warnings and critical errors with occurrence counts
- **Recommendations**: Actionable next steps based on test results and detected issues
- **Final Test Result**: Overall pass/fail status with appropriate exit code

If these tests fail, raise the issue in the `#exabox-infra` Slack channel and tag the syseng and scaleout teams.

### Fabric Tests

Stress tests for the TT-Fabric layer. Ensures TT-Fabric SW and FW is compatible with the cluster topology. Also stresses the physical ethernet interconnect, verifying cluster stability. Only run after physical validation passes.

**Topology-Specific Scripts:**

The current BH Exabox has two different cluster topologies (both with 4 Galaxies, 128 chips):
- **8x16** (16×8 mesh topology) - Located in Aisle C - Used for DeepSeek-R1 implementation, machines in cluster:
  - `bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08`
  - `bh-glx-c03u02,bh-glx-c03u08,bh-glx-c04u02,bh-glx-c04u08`
- **4x32** (32×4 mesh topology) - Located in Aisle B - Used for Wan2.2 implementation, machines in cluster:
  - `bh-glx-c05u02,bh-glx-c05u08,bh-glx-c06u02,bh-glx-c06u08`

The mesh shape affects how workloads are distributed across chips. Choose the script matching your cluster topology:

```bash
./tools/scaleout/exabox/run_fabric_tests.sh --hosts <hosts> --image <docker-image> --config 4x32
./tools/scaleout/exabox/run_fabric_tests.sh --hosts <hosts> --image <docker-image> --config 8x16
```

Logs are saved to `fabric_test_logs/` in your current directory.

Analyze results with:
```bash
python3 tools/scaleout/exabox/analyze_fabric_results.py fabric_test_logs/<log-file>.log
```

The script parses the test log and provides:

- **Test Status**: Overall pass/fail status for fabric tests across all hosts
- **Warnings & Failures**: Critical errors and runtime warnings detected during testing
- **Recommendations**: Actionable next steps including escalation paths for persistent issues
- **Final Test Result**: Clear pass/fail indication with appropriate exit code


**Important: Host Ordering Matters for 4x32 Clusters**

For 4x32 topology, you **must** specify hosts in physical connectivity order. The 4 Galaxies are connected in a ring (`1 <-> 2 <-> 3 <-> 4 <-> 1`), and MPI rank assignment depends on the order you pass hosts.

If you see this error:
```
TT_FATAL: Graph specified in MGD could not fit in the discovered physical topology
```

This can mean one of two things:

1. **Physical validation reported missing connections** - Run physical validation first to check cluster health
2. **Hosts passed in wrong order** - Only if physical validation passed, fix by:
   - Identifying which host is "Host 1" in your pod
   - Logging onto that host
   - Passing hosts in ring order: `host1,host2,host3,host4`

See [Fabric Test Fails with "Graph could not fit in physical topology"](./TROUBLESHOOTING.md#fabric-test-fails-with-graph-could-not-fit-in-physical-topology) for details.

**Note:** These topology-specific scripts will eventually be replaced with a unified cluster-level descriptor approach that handles host ordering automatically.

If these tests fail, raise the issue in the `#exabox-infra` Slack channel and tag the syseng and scaleout teams.

**Exabox Physical Layout:**

![Exabox Diagram](https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/scaleout/exabox/images/exabox_diagram.png?raw=true)

## Quick Health Check (For Developers)

For day-to-day use when you just need to verify a cluster is working.

**Important: These scripts require a local build!**

Unlike the Docker-based qualification scripts above (`run_validation_*.sh`, `run_fabric_tests_*.sh`), the recovery scripts run binaries directly on the host and require you to build tt-metal first.

| Script Type | Requires Build? | Uses Docker? |
|-------------|----------------|--------------|
| `recover_*.sh` | **Yes** | No |
| `run_validation_*.sh` | No | Yes |
| `run_fabric_tests_*.sh` | No | Yes |

**Build first:**
```bash
./create_venv.sh
source python_env/bin/activate
./build_metal.sh --build-metal-tests
```

**Then run:**
```bash
./tools/scaleout/exabox/recover_8x16.sh <hosts>
# or
./tools/scaleout/exabox/recover_4x32.sh <hosts>
```

Look for `All Detected Links are healthy` in the output.

**If you see "could not access or execute an executable"**: You haven't built tt-metal. Either build it (see above) or use the Docker-based `run_validation_*.sh` scripts instead, which don't require a build.

## Troubleshooting

**Machine reboots during test**: If your machine reboots mid-test, the tests keep running on the other machines. You'll need to manually kill them or wait for them to timeout.

**Missing connections** (`Channel/Port Connections found in FSD but missing in GSD`): Check cables are seated, verify you're using the right FSD file.

**Timeouts** (`Timeout (10000 ms) waiting for physical cores`): Usually a transient issue. Issue a cluster-level reset (do NOT power cycle). If the issue persists, contact syseng in the `#exabox-infra` Slack channel. Power cycling should only be done in coordination with cluster managers (infra and cloud teams).

**DRAM Training Failed**: Usually a transient hardware issue that will go away with another cluster reset. If it persists after multiple resets, contact syseng in the `#exabox-infra` Slack channel.

**Tests hanging**: Could be a workload issue or cluster issue. First, run validation scripts to ensure the cluster is healthy. If the cluster is unhealthy, contact syseng in the `#exabox-infra` Slack channel. If the cluster is healthy but the workload hangs, this is an application problem - debug the workload. Do NOT power cycle.

**Fabric tests failing**: Make sure physical validation passed first.

## Validation Output

Output from `recover_*.sh` and `run_validation.sh`:

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

**About Data Mismatches:**

Non-deterministic data mismatches are currently being debugged by the Systems Engineering team and are seen across all BH Galaxy clusters. These usually go away with a second reset.

A missing cable or bad port/connection will show up as a **consistently missing link** during validation (not as a data mismatch).

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `recover_*.sh` | Quick reset + 5 traffic iterations |
| `run_validation.sh` | Full 50-loop validation |
| `run_dispatch_tests.sh` | Chip stability stress tests |
| `run_fabric_tests.sh` | Fabric connectivity tests |
| `analyze_validation_results.py` | Parse validation logs || `analyze_dispatch_results.py` | Parse dispatch test logs |
| `analyze_fabric_results.py` | Parse fabric test logs || `mpi-docker` | MPI+Docker wrapper (`--help` for usage) |

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

# BH Galaxy Exabox Validation

> **Important — for the latest recommendations and step-by-step guides, use Confluence: [Exabox DCAMP Bringup](https://tenstorrent.atlassian.net/wiki/spaces/Exabox/pages/2509373472).** Those pages are the source of truth and are kept current for the full workflow — preflight, validation, fabric tests, dispatch tests, `recover.sh`, and troubleshooting/escalation. This README is a summary and may lag.

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
- **Exit Codes**: Returns specific exit codes for automated workflows

**Exit Codes Description**
The script returns exit codes enabling automated troubleshooting (e.g., Ansible playbooks):
- `0` - Cluster healthy (≥80% success rate)
- `1` - Unhealthy links (repeated on same link)
- `2` - Unhealthy links (scattered across different links)
- `3` - DRAM training failures
- `4` - Missing connections (found in FSD but not in discovered topology)
- `5` - Extra connections (found in topology but not in FSD)
- `6` - Missing global connection
- `7` - FSD error
- `8` - MGD error
- `9` - Workload timeout
- `10` - ARC timeout
- `11` - AICLK timeout
- `12` - Network errors (MPI/SSH)
- `13` - Device init error (PCIe hang / ARC firmware startup failure)
- `50` - Inconclusive (manual review required)
- `66` - Input error (file/directory not found)

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

**Exit Codes for Automation:**
The script returns specific exit codes for automated workflows:
- `0` - All tests passed
- `1` - MGD error (topology mismatch)
- `2` - Firmware initialization failed
- `3` - Fabric router sync timeout
- `4` - Test hanging (incomplete log)
- `5` - NOC address conflict
- `6` - Ethernet core timeout
- `50` - Inconclusive (manual review required)
- `66` - Input error (log file not found)


**Important: Host Ordering Matters for 4x32 Clusters**

For 4x32 topology, you **must** specify hosts in physical connectivity order. The 4 Galaxies are connected in a ring (`1 <-> 2 <-> 3 <-> 4 <-> 1`). If you see `TT_FATAL: Graph specified in MGD could not fit in the discovered physical topology`, see [Fabric Test Fails with "Graph could not fit in physical topology"](./TROUBLESHOOTING.md#fabric-test-fails-with-graph-could-not-fit-in-physical-topology) for diagnosis and resolution.

If these tests fail, raise the issue in the `#exabox-infra` Slack channel and tag the syseng and scaleout teams.

**Exabox Physical Layout:**

![Exabox Diagram](https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/scaleout/exabox/images/exabox_diagram.png?raw=true)

## Quick Health Check (For Developers)

For day-to-day use when you just need to verify a cluster is working. `recover.sh` runs a distributed `tt-smi` reset, then cluster validation, then (on unrecoverable failure) auto-regenerates a degraded descriptor set — ~2 minutes end to end. Latest guidance: [Reference — Recover.sh](https://tenstorrent.atlassian.net/wiki/spaces/Exabox/pages/2509373523/Reference+Recover.sh+Quick+Reset+Validate) on Confluence.

**Before you run (prerequisites):**

- **Preflight must pass first** — chips visible (`tt-smi -l` shows 32 per host) and links up. Don't run `recover.sh` on a cluster that hasn't cleared preflight.
- **Passwordless SSH to every host** via agent forwarding (see [SSH Setup](#ssh-setup)).
- **MPI must reach all hosts.** Seed host keys and confirm reachability before running — this is the #1 cause of `recover.sh` hanging:
  ```bash
  export HOSTS=<comma-separated-hosts>
  echo "$HOSTS" | tr ',' '\n' | xargs -I{} sh -c 'ssh-keyscan -H {} >> ~/.ssh/known_hosts 2>/dev/null'
  mpirun --host "$HOSTS" hostname   # prints each hostname; if it hangs or prompts, fix SSH first
  ```

**On Exabox, always run from the vetted pre-built path — no image, nothing to compile:**
```bash
cd /data/local-syseng-manual/tt-metal-recover
./tools/scaleout/exabox/recover.sh --hosts <hosts> --mpi-if ens5f0np0
# or for 8x16 configuration:
./tools/scaleout/exabox/recover.sh --hosts <hosts> --config 8x16 --mpi-if ens5f0np0
```

This is the path everyone should use for a quick health check — it's already built and kept current, so you don't need to clone or build anything. (On another site the NIC name differs, so drop `--mpi-if` to auto-detect, or pin the right interface.)

Look for `All Detected Links are healthy` in the output. No banner means it isn't done.

**Rules — do / don't:**

DO:
- **Always run from the vetted path** `/data/local-syseng-manual/tt-metal-recover` — pre-built, no image needed.
- **Pin the MPI interface** with `--mpi-if ens5f0np0` on Exabox (other sites: check `ip link`, or omit `--mpi-if` to auto-detect).
- **If a run fails, power-cycle the affected hosts via `#bmc-bots` and re-run** before escalating — this clears most transient stalls and hangs.
- **Confirm `All Detected Links are healthy`** before calling it done.

DON'T:
- **Don't run from your own checkout** (`/data/<user>/tt-metal`) unless you specifically need to test a local change — use the vetted path.
- **Don't reach for image-based or hand-built runs** unless the vetted path can't do what you need.
- **Don't flash or update firmware** on cluster machines — ever. They run debug FW (see [Do NOT Update Firmware on Cluster Machines](./TROUBLESHOOTING.md#do-not-update-firmware-on-cluster-machines)).
- **Don't power-cycle on your own** — go through `#bmc-bots` (coordinate with the infra/cloud cluster managers).
- **Don't combine `--skip-reset` with `--skip-validation`** — `recover.sh` fails fast with `Error: cannot use both --skip-reset and --skip-validation` (you must keep at least one of reset or validation).
- **Don't declare success without the healthy-links banner.**

**These scripts run a local build, not Docker.** Unlike the Docker-based qualification scripts above (`run_validation.sh`, `run_fabric_tests.sh`), the recovery scripts run binaries directly on the host, so a build has to exist somewhere. The vetted path above already provides one; only build tt-metal yourself if you're running from your own checkout (e.g. to test a local change) or on a site without the vetted path.

| Script Type | Uses Docker? | Where the build comes from |
|-------------|--------------|----------------------------|
| `recover_*.sh` | No | Vetted pre-built path (Exabox), or your own `build_metal.sh` |
| `run_validation.sh` | Yes | Docker image |
| `run_fabric_tests.sh` | Yes | Docker image |

**Building your own (only if not using the vetted path):**
```bash
./create_venv.sh
source python_env/bin/activate
./build_metal.sh --build-metal-tests
```

If you see `could not access or execute an executable`, the build is missing — use the vetted path or build first. See [Recovery Script Fails](./TROUBLESHOOTING.md#recovery-script-fails-with-could-not-access-or-execute-an-executable).

**Tolerating missing cables:** by default, recovery fails if any expected cable is missing — either with `Encountered unrecoverable state` after 5 retrain attempts, or by early-exiting after a successful retrain without sending traffic. To validate the rest of the cluster when one or more cables are down, forward `--min-connections N` (relaxed mode, ASIC pair passes if it has at least N connections) via `--validation-args` and/or pass `--rerun-on-retrain` (rerun validation after a successful retrain so traffic actually runs).

```bash
./tools/scaleout/exabox/recover.sh --hosts <hosts> --validation-args "--min-connections 3" --rerun-on-retrain
```

Any other `run_cluster_validation` flag (e.g. `--hard-fail`, `--print-connectivity`, `--log-ethernet-metrics`) can be forwarded via `--validation-args`.

## Troubleshooting

**Machine reboots during test**: If your machine reboots mid-test, the tests keep running on the other machines. You'll need to manually kill them or wait for them to timeout.

**Missing connections** (`Channel/Port Connections found in FSD but missing in GSD`): Check cables are seated, verify you're using the right FSD file. If you only need to validate the rest of the cluster while a known cable is down, rerun with `--validation-args "--min-connections N"` and/or `--rerun-on-retrain` (see [Quick Health Check](#quick-health-check-for-developers)).

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
| `analyze_validation_results.py` | Parse validation logs |
| `analyze_dispatch_results.py` | Parse dispatch test logs |
| `analyze_fabric_results.py` | Parse fabric test logs |
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

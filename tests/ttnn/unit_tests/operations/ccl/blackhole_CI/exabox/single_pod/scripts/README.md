# single-pod test runner scripts

Helpers to run the 16-rank single-pod tests on the 4-host BH cluster.

## Cheat sheet

```bash
cd tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/single_pod/scripts
```

**One-time setup** (per session, and after any hang):

```bash
./reset_chips.sh
```

**Run a test** (each script picks one specific test — no args needed):

```bash
./run_1block.sh        # 1.  ReduceToOneB1 chain — single block (per-token wire workload)
./run_10blocks.sh      # 2.  ReduceToOneB1 chain — 10 blocks (10-token decode pattern)
./run_pipeline.sh      # 3.  Pipeline framework smoke (fake-MoE substitution)
```

**If hung, recover** (then re-run setup):

```bash
./recover_hung_run.sh && ./reset_chips.sh
```

That's it. Reset → run → (recover if needed) → reset → next test.

## Files

| Script | Purpose |
|---|---|
| `reset_chips.sh` | `tt-smi -glx_reset_auto` on all 4 hosts in parallel (~60s). |
| `recover_hung_run.sh` | `pkill -9` local + remote pytest/prte after a hang. |
| `run_1block.sh` | Test 1 — `ReduceToOneB1` × 1 (fast dispatch). |
| `run_10blocks.sh` | Test 2 — `ReduceToOneB1` × 10 (fast dispatch). |
| `run_pipeline.sh` | Test 3 — pipeline framework with fake MoE/LMHead (slow dispatch). |
| `ssh_ulimit_wrapper.sh` | PRRTE ssh agent that injects `ulimit -n 65536;` into remote commands. |
| `_hosts.sh` / `_run_common.sh` | Internal helpers (host list, shared launch logic). |

## What each test exercises

| # | Per-rank work | What it validates |
|---|---|---|
| 1 | one `ReduceToOneB1` (3-level tree, 8 → 1) on the (4, 2) submesh | the demo's actual reduce-to-one op under `FABRIC_2D_TORUS_Y` |
| 2 | ten back-to-back `ReduceToOneB1` calls | 10-token decode worth of MoE-end traffic; program cache + semaphore lifecycle |
| 3 | full 16-stage Blitz pipeline framework with MoE/LMHead replaced by no-compute stubs | sockets, fabric, tt-run, mesh-graph descriptor, slow dispatch |

## Pre-flight assumptions

1. `$TT_METAL_HOME/build/test/tt_metal/tt_fabric/test_physical_discovery` is built (`./build_metal.sh --build-tests` or `--build-tt-train`). Used by `bootstrap_pipeline_dir.sh` to discover per-host PCIe device IDs. The runner scripts auto-bootstrap on first launch.
2. `/opt/openmpi-v5.0.7-ulfm/bin` is reachable.
3. SSH to all 4 hosts works without a password prompt.

Override the host list per-shell if needed:

```bash
SINGLE_POD_HOSTS="hostA hostB hostC hostD" ./run_1block.sh
```

## Logs

Each invocation writes `/tmp/single_pod_<timestamp>_<test>.log`. The runner prints the last 40 lines on exit. To follow live in another shell:

```bash
LATEST=$(ls -t /tmp/single_pod_*.log | head -1)
tail -f "$LATEST" | sed 's/\x1b\[[0-9;]*m//g'   # strips ANSI color codes
```

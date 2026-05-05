# Runbook — single-pod 16-rank tests on QUAD_BH

End-to-end instructions for manually exercising the three single-pod tests
in this folder. Aligned with the parent `exabox/AGENTS.md` runbook style and
this folder's `scripts/README.md` cheat sheet — read both first if you've
never run an exabox test.

---

## 1. What's in this folder

The single-pod tests run on a **16-rank** layout (4 hosts × 4 MPI
ranks/host), each rank seeing a `(4, 2)` per-rank submesh of an 8-device
slice of one Galaxy host. This is a different layout from the 4-rank
training tests next door (`../training/`).

| # | Script | Pytest target | Dispatch | Wallclock |
|---|---|---|---|---|
| 1 | `scripts/run_1block.sh` | `test_fake_moe_traffic.py::test_fake_moe_chain_real_reduce_to_one_4x2_single_pod[1block-…]` | fast | ~60–90 s |
| 2 | `scripts/run_10blocks.sh` | `test_fake_moe_traffic.py::test_fake_moe_chain_real_reduce_to_one_4x2_single_pod[10blocks-…]` | fast | ~90–120 s |
| 3 | `scripts/run_pipeline.sh` | `test_single_pod_pipeline_fake_moe.py::test_single_pod_pipeline_fake_moe` | **slow** | ~5–10 min |

**What each test exercises**

| # | Per-rank work | Validates |
|---|---|---|
| 1 | one `ReduceToOneB1` tree (3-level, 8→1) on the (4, 2) submesh | the demo's actual reduce-to-one op under `FABRIC_2D_TORUS_Y` |
| 2 | ten back-to-back `ReduceToOneB1` calls | 10-token decode worth of MoE-end traffic; program cache + semaphore lifecycle across iterations |
| 3 | full 16-stage Blitz pipeline framework with MoE / LMHead replaced by no-compute stubs (see `_fake_moe_helpers.py`) | sockets, fabric, tt-run, mesh-graph descriptor, slow dispatch reachability — *not* MoE numerics |

`test_single_pod_pipeline_fake_moe.py` substitutes the broken MoE op's
synthetic-weight path with `PassthroughStage(ACTIVATION)` so the
framework runs end-to-end. The CCL traffic itself is validated separately
by tests 1 and 2.

---

## 2. Prerequisites

### 2.1 Build

Standard tt-metal build is enough — the single-pod tests don't depend on
tt-train. If you also have the sibling `../training/` tests in scope you
likely already built with `--build-tt-train`; that's fine here too.

### 2.2 Cluster

4-host BH Galaxy, default hostnames in `scripts/_hosts.sh`:

```
bh-glx-110-c07u02 bh-glx-110-c07u08 bh-glx-110-c08u02 bh-glx-110-c08u08
```

Override per-shell via:

```bash
export SINGLE_POD_HOSTS="hostA hostB hostC hostD"
```

### 2.3 SSH

Passwordless ssh from the launching host to every cluster host. The
single-pod runner specifically needs an ssh agent that injects
`ulimit -n 65536;` before each remote command — the default sshd
fd-limit (1024) is too low for tt-metal's fabric/socket allocations.
That wrapper is `scripts/ssh_ulimit_wrapper.sh` (PRRTE picks it up via
`--prtemca plm_rsh_agent`).

### 2.4 Generated pipeline config bundle (auto-bootstrapped)

Unlike the training tests (which use static rank-binding YAMLs in
`tests/tt_metal/distributed/config/`), the single-pod tests need a
**runtime-discovered** config bundle. Why discovered: the rank binding
needs `TT_VISIBLE_DEVICES` per rank, mapping each rank to the 8 specific
PCIe device IDs that form its (4,2) slice on its host. Those device IDs
are cluster-specific and can shift across reboots, so they have to be
probed at bring-up time on the actual cluster.

The bootstrap is **fully automatic** — the runner scripts
(`run_1block.sh`, `run_10blocks.sh`, `run_pipeline.sh`) call
`scripts/bootstrap_pipeline_dir.sh` if the bundle dir is missing or
incomplete. You should never need to invoke it by hand.

Default bundle path:

```
$TT_METAL_HOME/generated/single_pod_pipeline_dir/
```

Lives under `generated/` (already gitignored), so it's persistent across
sessions and doesn't depend on `/tmp` (which can age out, asymmetrically,
between hosts). Bootstrap wipes and recreates this directory on every
invocation, so stale bundles never accumulate. Override the path per-shell
with `SINGLE_POD_PIPELINE_DIR=...` if you need to.

Contents after bootstrap:

- `blitz_decode_pipeline_rank_binding_single_pod_ci.yaml` — 16-rank tt-run binding
- `blitz_decode_pipeline_rank_file_single_pod_ci` — `mpirun` rankfile (`slot=0-31` per host)
- `slice_to_pcie_device_mapping.yaml` — per-host slice→device-IDs map (probe output)
- Symlinks back to `$TT_METAL_HOME/{tt_metal,build,models,ttnn,runtime,tests,python_env}`

Under the hood the bootstrap chains `mpirun` →
`build/test/tt_metal/tt_fabric/test_physical_discovery` (gtest filter
`*Generate2x4SliceToPCIeDeviceMapping*`) → the Python generator at
`models/demos/deepseek_v3_b1/scaleout_configs/generate_blitz_decode_pipeline_configs.py`.
Wallclock ≈ 30–60 s.

To force a refresh manually (e.g. after changing the host list):

```bash
./scripts/bootstrap_pipeline_dir.sh
```

Prerequisite: `build/test/tt_metal/tt_fabric/test_physical_discovery` must
exist. It's built by `--build-tests` (which is implied by `--build-tt-train`).

### 2.5 Chips

Run a reset before the first launch and after any hung run:

```bash
./scripts/reset_chips.sh
```

Wallclock ≈ 60 s. Reset is complete when each host's stdout shows
`Re-initialized 32 boards`.

---

## 3. The single rule

**One `tt-run` invocation at a time. Wait for completion before the next.**

Same rule as the parent `AGENTS.md`. Two `tt-run` launches concurrently
will collide on chip locks and either deadlock or corrupt each other.

---

## 4. Quick path

```bash
cd $TT_METAL_HOME/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/single_pod/scripts

./reset_chips.sh                 # ~60s, parallel across 4 hosts
./run_1block.sh                  # test 1
./reset_chips.sh                 # always reset between tests
./run_10blocks.sh                # test 2
./reset_chips.sh
./run_pipeline.sh                # test 3 (slow dispatch — longer)
```

Each runner script accepts `-h` / `--help`:

```bash
./run_1block.sh --help
./run_10blocks.sh --help
./run_pipeline.sh --help
./reset_chips.sh --help
./recover_hung_run.sh --help
```

---

## 5. Manual per-test invocations

If you need to bypass the wrapper scripts and call `tt-run` directly,
mirror what `_run_common.sh` does. The full template is below — usually
only the test name and `EXTRA_ENV` change.

```bash
PIPELINE_DIR=$TT_METAL_HOME/generated/single_pod_pipeline_dir
SCRIPTS=$TT_METAL_HOME/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/single_pod/scripts
TESTS=$TT_METAL_HOME/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/single_pod
HOSTS_WITH_SLOTS="bh-glx-110-c07u02:4,bh-glx-110-c07u08:4,bh-glx-110-c08u02:4,bh-glx-110-c08u08:4"

cd "$PIPELINE_DIR"
ulimit -n 65536
export PATH=/opt/openmpi-v5.0.7-ulfm/bin:$PATH

tt-run --skip-executable-check \
  --rank-binding "$PIPELINE_DIR/blitz_decode_pipeline_rank_binding_single_pod_ci.yaml" \
  --mpi-args "--map-by rankfile:file=$PIPELINE_DIR/blitz_decode_pipeline_rank_file_single_pod_ci \
              --bind-to hwt:overload-allowed --host $HOSTS_WITH_SLOTS \
              --tag-output --mca btl_tcp_if_exclude docker0,lo \
              --prtemca plm_rsh_agent $SCRIPTS/ssh_ulimit_wrapper.sh \
              --prtemca plm_ssh_no_tree_spawn 1" \
  bash -c "ulimit -n 65536; cd $PIPELINE_DIR && \
           source $TT_METAL_HOME/python_env/bin/activate && \
           MESH_DEVICE=QUAD_BH TT_METAL_HOME=$TT_METAL_HOME \
           pytest -svv --timeout=240 \
           $TESTS/test_fake_moe_traffic.py::test_fake_moe_chain_real_reduce_to_one_4x2_single_pod[1block-h7168-root_0_1-1link-4x2_grid-fabric_2d_torus_y]"
```

For the pipeline test (test 3), add `TT_METAL_SLOW_DISPATCH_MODE=1` to the
inner env and bump `--timeout=600`.

---

## 6. Critical PRRTE / launcher quirks

These are non-obvious requirements that the wrapper scripts already handle.
If you launch manually you must reproduce them.

### 6.1 `--prtemca plm_ssh_no_tree_spawn 1`

Without this, custom `plm_rsh_agent` scripts cause silent rendezvous
failures: the HNP loses communication with daemons even though ssh
succeeds. Test 1 / 2 / 3 all need this flag.

### 6.2 No `-np` flag

Let the rankfile drive the process count. With `-np` PRRTE reports a
spurious *"Rank 16 missing slot"* error. Use `:4` per host in `--host`
instead so MPI knows each host has 4 slots.

### 6.3 ulimit raise for ssh

The default sshd `nofile` ulimit is 1024 — below what tt-metal needs for
fabric/socket allocations. The wrapper at `scripts/ssh_ulimit_wrapper.sh`
parses ssh args, locates the host arg, and prefixes the remote command
with `ulimit -n 65536;`. Configured via PRRTE's `--prtemca plm_rsh_agent`.

### 6.4 Slow vs fast dispatch

| Mode | Required for | Tests 1, 2 | Test 3 |
|---|---|---|---|
| Fast (default) | `mesh_device.create_sub_device_manager(...)` — `ttnn.broadcast`, `all_to_all_*`, etc. | ✅ | ❌ |
| Slow (`TT_METAL_SLOW_DISPATCH_MODE=1`) | the blitz_decode pipeline framework (sockets + kernel-driven I/O) | ❌ | ✅ |

Implication: a single test process cannot exercise both `ttnn.broadcast`
sub-device CCL **and** the pipeline framework. The fake-MoE design splits
these:

- Tests 1 & 2 (fast dispatch) validate the actual `all_reduce` / reduce
  tree on the single-pod fabric.
- Test 3 (slow dispatch) replaces MoEDecoderStage with
  `PassthroughStage(ACTIVATION)` (see `_fake_moe_helpers.py`) so no CCL
  ops run inside the pipeline framework.

---

## 7. Troubleshooting

### 7.1 `Device N init: failed to initialize FW`

Chips were left in a stale state by the previous run. Reset:

```bash
./scripts/reset_chips.sh
```

### 7.2 Test hangs with no log progress for several minutes

Most common after an aborted run leaves chip locks held. Recover then
reset:

```bash
./scripts/recover_hung_run.sh
./scripts/reset_chips.sh
```

Symptom in the next run if you skip reset: `Waiting for lock
'CHIP_IN_USE_*_PCIe' which is currently held by thread TID:<old_pid>`.

### 7.3 `Fabric Router Sync: Timeout after 10000 ms`

Hardware-layer transient. Reset chips and retry once. If it fires twice
in a row, escalate.

### 7.4 PRRTE: silent failure / no progress / "rank 16 missing slot"

You're missing one of the launcher quirks in §6 — most likely
`--prtemca plm_ssh_no_tree_spawn 1` (silent failure with custom ssh
agents) or the `:4` slot suffix per host (rank 16 missing slot).

### 7.5 Bundle dir missing or incomplete

The runner now auto-bootstraps via `scripts/bootstrap_pipeline_dir.sh` —
you should never see this as a runtime failure. If you do, the most
likely cause is the discovery binary not being built. Build it:

```bash
cd $TT_METAL_HOME
./build_metal.sh --build-tests   # or --build-tt-train (implies --build-tests)
```

Then re-run the test; the runner will bootstrap automatically.

To force a fresh bootstrap manually (e.g. after changing
`SINGLE_POD_HOSTS`):

```bash
./scripts/bootstrap_pipeline_dir.sh
```

### 7.6 `Using sub device managers is unsupported with slow dispatch`

You're trying to run a fast-dispatch CCL op (test 1 / 2) under
`TT_METAL_SLOW_DISPATCH_MODE=1`, or running the pipeline test (test 3)
without slow dispatch. Use the right wrapper script — `_run_common.sh`
sets `EXTRA_ENV` correctly per test.

### 7.7 `ttnn.broadcast` hangs at 99% CPU on (4, 2) submesh under
`FABRIC_2D_TORUS_Y` *(test 3 only)*

Known issue documented in the parent `AGENTS.md` §"Known issue: a2a hangs
on FABRIC_2D_TORUS_Y submesh" — applies similarly to broadcast in the
pipeline framework. Works on the unified 16×4/32×4 meshes under
FABRIC_1D. The pipeline test sidesteps this by using PassthroughStage
substitution (no CCL inside the pipeline) — if you remove that
substitution the test will hang.

### 7.8 PCC ≈ 0.5 with high ATOL on multi-host runs

`torch.rand` was called without `torch.manual_seed`. Each MPI rank gets
different random data so the per-rank goldens disagree across the rank
boundary. Seed the RNG inside the helper that builds inputs. (Same root
cause as the parent `AGENTS.md` §A.)

---

## 8. Sanity-check on a single process

The conftest skips cleanly if invoked without MPI:

```bash
source python_env/bin/activate
pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/single_pod/ -v --no-header
```

Expected: tests collected, all skipped with `requires QUAD_BH` or world-size
mismatches. Useful to verify the conftest, fixtures, and pytest collection
work without touching the cluster.

---

## 9. Logs and artifacts

Each invocation writes:

```
/tmp/single_pod_<timestamp>_<test>.log
```

`_run_common.sh` prints the last 40 lines on non-zero exit and a per-rank
PASSED/FAILED/SKIPPED/ERROR summary banner. To follow live in another shell:

```bash
LATEST=$(ls -t /tmp/single_pod_*.log | head -1)
tail -f "$LATEST" | sed 's/\x1b\[[0-9;]*m//g'   # strips ANSI
```

For per-rank timestamps in a completed log:

```bash
grep -E "^\[1,[0-9]+\]" /tmp/single_pod_*.log | tail -20
```

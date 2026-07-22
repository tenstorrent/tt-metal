# Runbook — deepseek_v3_b1 Blitz pipeline framework smoke on QUAD_BH single pod

End-to-end instructions for running the 16-rank framework-smoke test in
this folder. Read this first.

---

## 1. What this folder tests

A local fork of `test_passthrough_pipeline_block` — invoked across 16
ranks on a 4-host BH Galaxy single pod.

| Path | Role |
|---|---|
| `test_passthrough_pipeline.py` | Local fork of `models/demos/deepseek_v3_b1/tests/unit_tests/test_multi_host_pipeline.py::test_passthrough_pipeline_block` with two upstream-bug workarounds (see §6). |
| `conftest.py` | Folder-local `deepseek_pipeline_mesh_device` fixture that threads `fabric_router_config` / `worker_l1_size` through `set_fabric` (the parent `exabox/conftest.py`'s `mesh_device` doesn't). Deliberately *not* named `mesh_device` so it doesn't shadow the parent/root fixtures when running adjacent tests locally. |
| `scripts/bootstrap_pipeline_dir.sh` | Generates `TT_VISIBLE_DEVICES` rank-binding via the deepseek `generate_blitz_decode_pipeline_configs.py` probe + a Rev C slicer-correction post-step. |
| `scripts/run_pipeline_smoke.sh` | Reset-chips-aware runner. Auto-bootstraps if needed, then `tt-run`s the local test. |
| `scripts/_hosts.sh` | Parses the caller-set `HOSTS` env var (no default — fails fast if unset). |
| `scripts/reset_chips.sh` | `tt-smi -glx_reset_auto` on every host in `HOSTS`. |
| `scripts/recover_hung_run.sh` | `pkill -9` of `tt-run`/`prterun`/`prted`/`pytest` locally and on every host. |

The test exercises the deepseek Python pipeline framework
(`PipelineConfiguration` → `Pipeline.setup_and_run()` → `Pipeline.terminate()`)
end-to-end on real fabric. It is not run by any current CI workflow —
see §8.

---

## 2. Prerequisites

### 2.1 Build

Standard tt-metal build with the discovery binary:

```bash
cd $TT_METAL_HOME
./build_metal.sh --build-tests
```

Required binaries the runner expects to find:
- `$TT_METAL_HOME/build/test/tt_metal/tt_fabric/test_physical_discovery`
- `$TT_METAL_HOME/python_env/bin/{tt-smi, tt-run, pytest, python}`

### 2.2 Cluster

4-host BH Galaxy. **`HOSTS` must be set per-shell — there is no default.**
Cluster membership is operator-specific; baking a host list into the repo
would silently target the wrong machines on any other cluster, so the
scripts refuse to run if `HOSTS` is unset or empty.

```bash
export HOSTS="hostA hostB hostC hostD"          # space-separated
export HOSTS="hostA,hostB,hostC,hostD"          # comma-separated also OK
```

### 2.3 SSH

Passwordless ssh from the launching host to every cluster host (BatchMode-compatible).

### 2.4 Chip state

The chips must be in a fresh state before the first run **and after every
prior tt-metal run that used persistent kernels** (slow dispatch leaves
chip locks held even after pytest exits cleanly). Always run
`reset_chips.sh` before each launch unless you've just reset.

---

## 3. Quick path

```bash
export TT_METAL_HOME=/path/to/your/tt-metal              # adjust to your repo
export HOSTS="hostA hostB hostC hostD"                   # your 4-host BH Galaxy pod

# First time (or after changing HOSTS): bootstrap; it resets chips at the end.
bash $TT_METAL_HOME/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/deepseek_pipeline_tests/scripts/bootstrap_pipeline_dir.sh
bash $TT_METAL_HOME/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/deepseek_pipeline_tests/scripts/run_pipeline_smoke.sh

# Re-running the same test (no re-bootstrap): reset chips first.
bash $TT_METAL_HOME/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/deepseek_pipeline_tests/scripts/reset_chips.sh
bash $TT_METAL_HOME/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/deepseek_pipeline_tests/scripts/run_pipeline_smoke.sh
```

Wallclock ≈ 5–10 min (slow dispatch). The runner prints a per-rank
`PASSED=N/16` summary at the end and writes the full log to
`/tmp/deepseek_passthrough_<timestamp>.log`.

After any hung / aborted run:

```bash
bash .../deepseek_pipeline_tests/scripts/recover_hung_run.sh
bash .../deepseek_pipeline_tests/scripts/reset_chips.sh
```

---

## 4. The single rule

**One `tt-run` invocation at a time. Wait for completion before the
next.** Two `tt-run` launches concurrently will collide on chip locks
and either deadlock or corrupt each other.

---

## 5. Configuration knobs

| Env var | Default | Purpose |
|---|---|---|
| `TT_METAL_HOME` | *(required)* | Repo root. |
| `HOSTS` | *(required)* | Space-/comma-separated 4-host list. No default — set per-shell. |
| `PYTEST_TIMEOUT` | `900` | Per-test pytest timeout, seconds. |
| `BOOTSTRAP_SKIP_RESET` | `0` | Set to `1` to skip the auto chip-reset at the end of `bootstrap_pipeline_dir.sh`. |
| `SINGLE_POD_PIPELINE_DIR` | `$TT_METAL_HOME/generated/single_pod_pipeline_dir` | Bundle dir (rank-binding YAML, rankfile, slice mapping, symlinks). |
| `SINGLE_POD_RANKFILE_PATH` | `/var/tmp/single_pod_rankfile` | Hyphen-free rankfile copy that mpirun consumes. |

These three names use the `SINGLE_POD_` prefix because the upstream
`blitz_pipeline_config_single_pod_ci.yaml` hard-codes the file basenames;
renaming would require touching the deepseek pipeline-config (out of
scope for this folder).

---

## 6. Bootstrap details

`bootstrap_pipeline_dir.sh` runs once at first launch (or whenever the
bundle is missing/incomplete). It:

1. Wipes and recreates `$SINGLE_POD_PIPELINE_DIR` locally and on every
   remote host. Drops symlinks back to `$TT_METAL_HOME`.
2. Writes the hostfile.
3. Detects BH Galaxy revision from `tt-smi -ls` (`(board_id >> 32) & 0xF >= 3` ⇒ Rev C).
4. Runs `models/demos/deepseek_v3_b1/scaleout_configs/generate_blitz_decode_pipeline_configs.py`, which mpirun-spawns `test_physical_discovery` to emit `slice_to_pcie_device_mapping.yaml`. **On a Rev C cluster this output is currently incorrect** — the slicer hard-codes Rev A&B tray IDs (PR #41414 added the Rev C tray-ID swap; PR #41072 silently dropped it).
5. **Rev C only** — re-collects per-host tray maps from the discovery
   binary's `GenerateTrayToPCIeDeviceMapping` test (which produces
   `tray_id → list of UMD logical IDs` per host), then post-processes the
   slice yaml so each per-rank `(4, 2)` submesh contains the
   *physically* connected chips for the Rev C chassis layout.
6. Re-emits `blitz_decode_pipeline_rank_binding_single_pod_ci.yaml`
   from the corrected slice yaml.
7. Copies the rankfile to a hyphen-free path
   (`/var/tmp/single_pod_rankfile`) — OpenMPI 5.0.7's
   `--map-by rankfile:file=PATH` parser rejects paths containing `-`.
8. Resets chips on every host (calls `reset_chips.sh`). The discovery
   probes in steps 4 and 5 open devices and leave stale FW/lock state
   that would otherwise trip the next test run with `Device N init:
   failed to initialize FW`. Skip with `BOOTSTRAP_SKIP_RESET=1`.

**When to re-bootstrap manually:**
- You changed `HOSTS`.
- The cluster's PCIe device IDs shifted (rare; usually only after a
  full host reboot).
- You suspect the bundle is stale.

To force a fresh bootstrap, just run the script — it always wipes and
regenerates. The runner's auto-bootstrap is identical to manual, except
it only fires when the bundle is missing.

---

## 7. The test itself

`test_passthrough_pipeline.py::test_passthrough_pipeline_block` builds
a 16-stage ring with synthetic embedding weights:

| Rank | Stage |
|---|---|
| 0 | `EmbeddingStage(loopback_payload=ACTIVATION_W_TOKEN_META)` |
| 1–15 | `PassthroughStage(ACTIVATION_W_TOKEN_META)` |

Why `ACTIVATION_W_TOKEN_META` everywhere — and why we don't reuse the
shared `create_passthrough_pipeline_configuration` helper:

- `EmbeddingStage` hard-codes its **downstream** socket size to
  `ACTIVATION_W_TOKEN_META_FIFO_SIZE` (PR #43389; later confirmed by
  `9f8f765ca86`).
- The shared helper still hard-codes the chain payload to
  `PassthroughPayload.ACTIVATION` *and* the `EmbeddingStage`
  `loopback_payload` to `ACTIVATION`.
- That mismatch trips
  `tt_metal/distributed/mesh_socket_utils.cpp:153`'s socket-FIFO-size
  handshake check on stage 0→1 *and* stage 15→0 (loopback).

The local builder threads `ACTIVATION_W_TOKEN_META` through every D2D
socket pair, including the fabric-loopback wraparound.

The test also pre-generates `pipeline_config` and `stages_metadata` at
the top level (canonical pattern from
`test_lm_head_sampling.py::test_persistent_mode_spec_decode`) so each
stage's `PipelineBlock.__init__` skips its internal regen of
`generate_blitz_decode_pipeline()`. Without that, middle stages whose
`LoopbackConfig` is `no_loopback` re-run the generator with
`initialize_loopback=False`, producing a placeholder
`exit_node_coord == entry_node_coord` for the last stage that trips the
unique-fabric-node check at
`tt_metal/distributed/experimental/blitz_decode_pipeline.cpp:262` (added
by PR #42524, missing the `!initialize_loopback` skip the placeholder
needs).

The test stops at `setup_and_run() + barrier + terminate` — framework
reachability only. The upstream test additionally pushes vocab-size
tokens through and PCC-checks each one (a correctness gate, not a
framework gate); we drop it here.

---

## 8. Relationship to CI

This invocation is **not** in any CI workflow.

- `ops-post-commit.yaml:86-89` runs `models/demos/deepseek_v3_b1/tests/unit_tests/` via plain pytest (single-process). The upstream `test_passthrough_pipeline_block` runs there with `num_procs=1` — a degenerate single-stage pipeline, so the multi-host code paths in the test go untouched.
- `fabric-multihost-exabox.yaml:227-242` (the actual multi-host CI workflow) runs a different test: the C++ binary `build/test/tt_metal/multihost/socket_pipeline/multiprocess/unit_tests_pipeline`, which uses raw `MeshSocket` with hardcoded physical pipeline configs. It does not touch the deepseek pipeline framework.

Our setup invokes the multi-host *Python* path that exists in deepseek's
own test suite but isn't currently exercised by CI.

---

## 9. Troubleshooting

### 9.1 `Device N init: failed to initialize FW! Try resetting the board.`

Chips were left in a stale state by a prior run. Reset:

```bash
bash scripts/reset_chips.sh
```

You must reset between runs. Slow-dispatch persistent kernels can leave
chips half-initialized even after pytest exits cleanly.

### 9.2 Test hangs with no log progress for several minutes

Most common after an aborted run leaves chip locks held. Recover then
reset:

```bash
bash scripts/recover_hung_run.sh
bash scripts/reset_chips.sh
```

Symptom in the next run if you skip reset:
`Waiting for lock 'CHIP_IN_USE_*_PCIe' which is currently held by
thread TID:<old_pid>`.

### 9.3 `TypeError: open_mesh_device() got an unexpected keyword argument 'fabric_router_config'`

The test is resolving against the parent `exabox/conftest.py`'s
`mesh_device` instead of the folder-local
`deepseek_pipeline_mesh_device`. Verify the test's
`@pytest.mark.parametrize` and function signature both reference
`deepseek_pipeline_mesh_device`, and that `conftest.py` exists in this
folder.

### 9.4 `TT_FATAL: Mismatch in socket FIFO size during handshake.`
(at `tt_metal/distributed/mesh_socket_utils.cpp:153`)

A socket-pair has incompatible FIFO sizes. If the test ever regresses to
calling `create_passthrough_pipeline_configuration` directly (or someone
adds a stage with a different payload), this re-fires. Verify every D2D
socket pair along the ring uses the same `PassthroughPayload`. See §7.

### 9.5 `TT_FATAL: Stage [N] exit fabric node ... is reused across stages`
(at `tt_metal/distributed/experimental/blitz_decode_pipeline.cpp:262`)

The per-stage `PipelineBlock.__init__` regenerated `pipeline_config`
with `initialize_loopback=False` and produced an entry==exit placeholder
for the last stage. Verify the test passes
`stages_metadata` and `pipeline_config` to `build_pipeline()`. See §7.

### 9.6 Multi-rank chip-lock deadlock with a `Galaxy Rev C` warning trail

The bootstrap's Rev C slicer correction didn't run, or detected the
wrong revision, or the cluster has heterogeneous revs. Force a fresh
bootstrap:

```bash
rm -rf $TT_METAL_HOME/generated/single_pod_pipeline_dir /var/tmp/single_pod_rankfile
for h in $HOSTS; do
  ssh -o BatchMode=yes "$h" \
    "rm -rf $TT_METAL_HOME/generated/single_pod_pipeline_dir; rm -f /var/tmp/single_pod_rankfile"
done
bash scripts/bootstrap_pipeline_dir.sh
```

Inspect `bootstrap`'s output line `BH Galaxy revision = ...` to confirm
detection. If the cluster is mixed-rev (different boards report
different revisions), the four-equation correction is *not* valid and
will refuse to run.

### 9.7 `Fabric Router Sync: Timeout after 10000 ms`

Hardware-layer transient. Reset chips and retry once. If it fires twice
in a row, escalate.

### 9.8 `Mapping validation failed: N target node(s) are not mapped to any global node`

Inter-mesh topology mapping is rejecting the run because the physical
fabric doesn't form the assumed inter-mesh ring. The test currently uses
the deepseek-shared MGD with a 16-mesh ring; that ring requires real
cross-host fabric edges in the canonical snake order
(`low_u08 → low_u02 → high_u02 → high_u08 → wrap`). If your cluster's
fabric topology is different, the inter-mesh mapper has nothing to map
the ring onto. Verify physical connectivity with
`./build/test/tt_metal/tt_fabric/test_system_health`.

---

## 10. Logs and artifacts

Each invocation writes:

```
/tmp/deepseek_passthrough_<timestamp>.log
```

`run_pipeline_smoke.sh` prints the last 40 lines of this log on
non-zero exit, plus a per-rank PASSED/FAILED/SKIPPED/ERROR summary
banner.

Follow live in another shell:

```bash
LATEST=$(ls -t /tmp/deepseek_passthrough_*.log | head -1)
tail -f "$LATEST" | sed 's/\x1b\[[0-9;]*m//g'    # strip ANSI
```

Per-rank timestamps in a completed log:

```bash
grep -E "^\[1,[0-9]+\]" /tmp/deepseek_passthrough_*.log | tail -20
```

---

## 11. Sanity-check on a single process

The conftest skips cleanly if invoked without MPI:

```bash
source $TT_METAL_HOME/python_env/bin/activate
pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/deepseek_pipeline_tests/ -v --no-header
```

Expected: tests collected, all skipped because `num_procs != 16`.
Useful to verify the conftest, fixtures, and pytest collection work
without touching the cluster.

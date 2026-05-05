# Runbook — SocketManager send/recv exabox tests on QUAD_BH

End-to-end instructions for manually exercising the six tests in this folder.
Aligned with the parent `exabox/AGENTS.md` runbook style — read that first if
you've never run an exabox test.

---

## 1. What's in this folder

Two transport paths, each with three training-shaped tests.

| Test | Path | What it exercises |
|---|---|---|
| `test_send_recv_training.py::test_round_robin_send_recv_32x4` | byte-level (host MPI) | Adjacent-rank ring exchange in both directions |
| `test_send_recv_training.py::test_pipeline_activation_handoff_32x4` | byte-level | Stage-boundary fwd activation + bwd gradient |
| `test_send_recv_training.py::test_remote_optimizer_grad_exchange_32x4` | byte-level | All workers (1,2,3) → aggregator (0); aggregator → all workers |
| `test_socket_manager_fabric_training.py::test_round_robin_send_recv_fabric_32x4` | FABRIC | Same ring pattern, routed via SocketManager → MeshSocket → fabric |
| `test_socket_manager_fabric_training.py::test_pipeline_activation_handoff_fabric_32x4` | FABRIC | Same handoff, fabric-routed |
| `test_socket_manager_fabric_training.py::test_remote_optimizer_grad_exchange_fabric_32x4` | FABRIC | rank 0 ↔ adjacent workers `{1, 3}` only (see §6) |

Each FABRIC test takes ~50–60 s (most of which is fabric router init).
Byte-level tests run in seconds.

---

## 2. Prerequisites

### 2.1 Build

`_ttml*.so` must exist with the byte-level send/recv Python bindings:

```bash
cd $TT_METAL_HOME
./build_metal.sh --build-tt-train
```

The bindings live in `tt-train/sources/ttml/nanobind/nb_core.cpp`
(see `DistributedContext::send`/`recv`). The conftest auto-symlinks
`build_Release/ttml/_ttml*.so` into the source package on first import.

### 2.2 Cluster

4-host BH Galaxy (`bh-glx-*`), each host a full Galaxy of 32 devices,
inter-host fabric wired as a 4-node ring (verified by TopologyMapper at
launch time). Default host list:

```
bh-glx-b06u02, bh-glx-b06u08, bh-glx-b07u02, bh-glx-b07u08
```

Override via `HOSTS=...` env in the launcher script, or substitute on the
command line.

### 2.3 SSH

Passwordless ssh from the launching host to every cluster host, including
to itself. Test once before launching:

```bash
for h in bh-glx-b06u02 bh-glx-b06u08 bh-glx-b07u02 bh-glx-b07u08; do
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "hostname"
done
```

### 2.4 Chips

Run a reset before the first launch and after any hung run:

```bash
for h in bh-glx-b06u02 bh-glx-b06u08 bh-glx-b07u02 bh-glx-b07u08; do
  ssh -o BatchMode=yes "$h" "$TT_METAL_HOME/python_env/bin/tt-smi -glx_reset_auto" &
done
wait
```

Each reset takes ~60 s. Reset is complete when each host's stdout shows
`Re-initialized 32 boards`.

---

## 3. The single rule

**One `tt-run` invocation at a time. Wait for completion before the next.**

The cluster's chips are shared. Two `tt-run` launches will collide and
deadlock or corrupt each other.

---

## 4. Quick path — run everything

```bash
cd $TT_METAL_HOME
bash tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/scripts/run_training_tests.sh
```

This launches `pytest` once for the byte-level tests under the standard
single-mesh binding. **It does not currently run the FABRIC tests** — those
need a different binding (see §5). For full coverage, run §5.1 then §5.2
manually one at a time.

---

## 5. Manual per-test invocations

Both invocations use the same `--mpi-args` host list. Two different
`--rank-binding` YAMLs depending on transport.

### 5.1 Byte-level (single-mesh binding)

Binding: `tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml`
(stock — all 4 ranks share `mesh_id: 0`).

```bash
tt-run \
  --rank-binding tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \
  --mpi-args "--host bh-glx-b06u02,bh-glx-b06u08,bh-glx-b07u02,bh-glx-b07u08" \
  bash -c "source python_env/bin/activate && \
           MESH_DEVICE=QUAD_BH pytest --timeout=300 -v \
           tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/test_send_recv_training.py::test_round_robin_send_recv_32x4"
```

Substitute the test name to run the other byte-level tests:

- `::test_round_robin_send_recv_32x4`
- `::test_pipeline_activation_handoff_32x4`
- `::test_remote_optimizer_grad_exchange_32x4`

You can also run all three byte-level tests in one invocation by dropping
the `::test_*` suffix:

```bash
tt-run ... pytest --timeout=300 -v \
  tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/test_send_recv_training.py
```

**Expected output** (per test): `1 passed in <≈3-4s>`. Wall-time per test
~30 s (dominated by mesh init).

### 5.2 SocketManager-FABRIC (multi-mesh binding)

Binding: `tests/tt_metal/distributed/config/quad_bh_galaxy_split_4x2_multi_mesh_rank_bindings.yaml`
(authored — 4 ranks → 4 distinct `mesh_id`s).

Descriptor: `tests/tt_metal/tt_fabric/custom_mesh_descriptors/quad_bh_galaxy_4mesh_ring_8ch.textproto`
(authored — 4 × (8,4) Galaxy meshes, 8-channel ring inter-mesh edges).

```bash
tt-run \
  --rank-binding tests/tt_metal/distributed/config/quad_bh_galaxy_split_4x2_multi_mesh_rank_bindings.yaml \
  --mpi-args "--host bh-glx-b06u02,bh-glx-b06u08,bh-glx-b07u02,bh-glx-b07u08" \
  bash -c "source python_env/bin/activate && \
           MESH_DEVICE=QUAD_BH pytest --timeout=600 -v \
           tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/test_socket_manager_fabric_training.py::test_round_robin_send_recv_fabric_32x4"
```

Substitute test name for the other two FABRIC tests:

- `::test_round_robin_send_recv_fabric_32x4`             (~64 s)
- `::test_pipeline_activation_handoff_fabric_32x4`       (~50 s)
- `::test_remote_optimizer_grad_exchange_fabric_32x4`    (~50 s)

Note the longer per-test `--timeout=600` here; FABRIC setup takes 30–40 s
before the first send.

**Expected output** (per test): `1 passed in <≈45-65s>`. Each rank prints
INFO log lines like `[rank=N] sent tag=… to rank M` and `[rank=N] recv ok
from rank M`. After each test passes, kill any lingering processes (see §7)
before the next launch — pytest sometimes hangs in teardown even on success.

### 5.3 Reset between FABRIC test runs

Recommended (not strictly required) before each FABRIC launch:

```bash
for h in bh-glx-b06u02 bh-glx-b06u08 bh-glx-b07u02 bh-glx-b07u08; do
  ssh "$h" "$TT_METAL_HOME/python_env/bin/tt-smi -glx_reset_auto" &
done; wait
```

Skipping this is fine if the previous test exited cleanly. Mandatory after
any failure or hang.

---

## 6. Topology constraint baked into FABRIC test 3

The cluster's inter-mesh fabric is a 4-node ring with edges
`(0-1, 1-2, 2-3, 0-3)`. There is **no direct rank 0 ↔ rank 2 edge**.
`MeshSocket` setup between non-adjacent meshes consistently times out
on this hardware.

`test_remote_optimizer_grad_exchange_fabric_32x4` therefore only uses
`WORKERS = [1, 3]` (rank 0's direct neighbors) and lets rank 2 sit at
barriers. The byte-level variant of the same test
(`test_remote_optimizer_grad_exchange_32x4`) covers all-to-1 with all
three workers — host MPI is independent of fabric topology.

If you ever extend the FABRIC tests to non-adjacent rank pairs you'll
need to either (a) author a different mesh descriptor with an additional
edge that matches the physical wiring, or (b) implement multi-hop fabric
sockets at the C++ layer.

---

## 7. Troubleshooting

### 7.1 `pytest` hangs at 99% CPU after the test reports `passed`

Common — pytest's teardown sometimes can't tear down the fabric cleanly.
Kill it manually and proceed:

```bash
# locally
ps -ef | grep -E 'tt-run|prterun|pytest|prted' | grep -v grep | awk '{print $2}' | xargs -r kill -9

# on every host
for h in bh-glx-b06u02 bh-glx-b06u08 bh-glx-b07u02 bh-glx-b07u08; do
  ssh "$h" "ps -ef | grep -E 'pytest|prted' | grep -v grep | awk '{print \$2}' | xargs -r kill -9"
done
```

Confirm clean before the next launch:

```bash
ps -ef | grep -E 'tt-run|prterun|pytest|prted' | grep -v grep | wc -l
# → 0
for h in bh-glx-b06u02 bh-glx-b06u08 bh-glx-b07u02 bh-glx-b07u08; do
  echo -n "$h: "; ssh "$h" "ps -ef | grep -E 'pytest|prted' | grep -v grep | wc -l"
done
```

### 7.2 `Fabric Router Sync: Timeout after 10000 ms`

Hardware-layer transient. Reset chips and retry once. If it fires twice in
a row, escalate — likely a real fabric / wiring issue.

### 7.3 `Device N init: failed to initialize FW`

Chips were left in a stale state by the previous run. Run the parallel
reset from §2.4 and retry.

### 7.4 `MeshSocket must only be used for communication between different
host ranks` *(FABRIC tests only)*

You're using the **wrong rank-binding** for the FABRIC tests. They require
the multi-mesh binding from §5.2 (one `mesh_id` per rank), not the stock
single-mesh galaxy binding.

### 7.5 `Cannot create N receiver-sender pairs with only 0 available fabric
links between devices` *(FABRIC tests only)*

The mesh-graph descriptor doesn't declare inter-mesh `connections`, or
declares them between meshes that aren't physically wired. The descriptor
in §5.2 is correct for this cluster; if you change it, ensure each
declared `connections {}` block matches a real physical edge.

### 7.6 `Strict mode validation failed: target graph edge from node X to Y
requires N channels, but physical edge has fewer`

The descriptor demands more channels than the cluster has. Reduce
`channels { count: ... }` to match the physical capacity (8 on this
cluster).

### 7.7 `Graph specified in MGD could not fit in the discovered physical
topology … All mapping possibilities exhausted`

Either `device_topology` (per-mesh device shape) doesn't match the per-host
physical layout, or the inter-mesh edges in the descriptor don't match the
physical adjacency map. Check the log for the *Physical Mesh-Level Graph
Adjacency Map* and *Internal Graph Adjacency Map* sections — they show what
the topology mapper actually discovered. The current descriptor uses
`(8, 4)` per-mesh which matches the BH Galaxy single-host layout.

### 7.8 `Can't get a single buffer from host storage distributed over mesh
shape MeshShape([8, 4])` *(FABRIC tests only)*

You're calling something that expects a single-shard tensor on a
multi-shard one. Two common causes:

- **Tensor creation**: forgot the replicate mapper.
  Use `ttnn._ttnn.multi_device.replicate_tensor_to_mesh_mapper(device)`
  (the C++ factory — *not* the Python wrapper
  `ttnn.distributed.ReplicateTensorToMesh`, which is incompatible with
  `from_numpy`'s nanobind type check).
- **Verification**: `to_numpy()` without a composer.
  Use `tensor.to_numpy(composer=ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0))`
  and then read the first replica.

### 7.9 `Timed out trying to establish a socket connection` *(FABRIC)*

You're trying to send/recv between two ranks whose meshes have no direct
fabric edge. Check §6 — the cluster's fabric is a ring, not all-to-all.
Either use direct-neighbor pairs, or extend the test to barrier rank 2.

### 7.10 Test hangs with no log progress for 10+ minutes

Likely a partial-bring-up hang. Kill (§7.1) and reset chips (§2.4). If
it reproduces twice in a row, capture the per-rank log and escalate.

---

## 8. Sanity-check on a single process

The conftest skips cleanly if invoked without MPI:

```bash
source python_env/bin/activate
pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/ -v --no-header
```

Expected: 6 tests collected, 6 skipped with reason
`send/recv tests require multi-process launch (world_size=1); use tt-run.`

This confirms the conftest, `_ttml` bootstrap, and pytest collection all
work without touching the cluster. Useful to verify after a fresh build.

---

## 9. Logs and artifacts

Per-run logs from the iteration that landed these tests are at
`/tmp/training_send_recv_runs/`:

- `run3_*` — byte-level tests (passing)
- `run10_*` — FABRIC tests 1 & 2 (passing)
- `run12_03_*` — FABRIC test 3 with `WORKERS=[1,3]` (passing)
- `SUMMARY.txt` — full iteration history including the descriptor failures
  that led to the current `quad_bh_galaxy_4mesh_ring_8ch.textproto`

When you launch a new run, redirect into a sibling file so previous runs
remain available:

```bash
tt-run … > /tmp/training_send_recv_runs/run_$(date +%Y%m%d_%H%M%S).log 2>&1
```

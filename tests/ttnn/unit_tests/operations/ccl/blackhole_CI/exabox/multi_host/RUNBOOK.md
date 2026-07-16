# Runbook — general multi-host CCL exabox tests

End-to-end instructions for running the CCL collective tests in this folder
on the BH Galaxy cluster. These are the "everyday" CCL exabox tests
(`all_gather`, `all_reduce`, `reduce_scatter`, `broadcast`, `all_to_all_*`)
parametrized over the SINGLE_BH / DUAL_BH / QUAD_BH mesh sizes. Sister
folders cover specialized scenarios:

- `../deepseek_pipeline_tests/` — deepseek-v3 pipeline framework smoke test.
- `../training/` — byte-level `DistributedContext.send/recv` tests for tt-train.

---

## 1. What's in this folder

| File | What it tests |
|---|---|
| `test_all_gather_exabox.py` | `ttnn.all_gather` |
| `test_all_reduce_exabox.py` | `ttnn.all_reduce` |
| `test_reduce_scatter_exabox.py` | `ttnn.reduce_scatter` |
| `test_all_broadcast_exabox.py` | `ttnn.all_broadcast` |
| `test_broadcast_exabox.py` | `ttnn.broadcast` |
| `test_all_to_all_dispatch_exabox.py` | `ttnn.experimental.all_to_all_dispatch` (MoE dispatch) |
| `test_all_to_all_combine_exabox.py` | `ttnn.experimental.all_to_all_combine` (MoE combine) |
| `_a2a_moe_helpers.py` | shared `run_all_to_all_*_test` helpers used by both A2A tests |

Each test file has variants per mesh size — `test_*_8x4`, `test_*_16x4`,
`test_*_32x4` — selected by the mesh size you're running on.

---

## 2. Prerequisites

### 2.1 Build (bare-metal mode only)

Standard `./build_metal.sh` is sufficient (no `--build-tt-train`
needed for these tests). Existing build directory works.

When using `--image <docker-image>`, no local build is required — tests
run inside the Docker container. Only a tt-metal checkout (for `mpi-docker`
and launcher scripts) is needed.

### 2.2 Cluster

| Mesh | Devices | Hosts | `MESH_DEVICE` env |
|---|---|---|---|
| SINGLE_BH | 8×4 = 32 | 1 | `SINGLE_BH` |
| DUAL_BH | 16×4 = 64 | 2 | `DUAL_BH` |
| QUAD_BH | 32×4 = 128 | 4 | `QUAD_BH` |

Rank-bindings (used for DUAL/QUAD only):

- `tests/tt_metal/distributed/config/16x4_dual_bh_galaxy_rank_bindings.yaml`
- `tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml`

Default host lists in `scripts/run_multi_host_ccl_tests.sh`:

- DUAL_BH: `bh-glx-b06u02,bh-glx-b06u08`
- QUAD_BH: `bh-glx-b06u02,bh-glx-b06u08,bh-glx-b07u02,bh-glx-b07u08`

Override per-shell with `HOSTS="hostA,hostB,hostC,hostD"` if your cluster
differs.

### 2.3 SSH

Passwordless SSH from the launching host to every host in `HOSTS`.

---

### 2.4 Docker mode prerequisites

When using `--image`, the launching host needs:

- Docker installed on all target hosts
- Passwordless SSH between hosts (same as bare-metal)
- The Docker image pulled on each host (or `docker pull` access)
- `mpi-docker` script (comes with the tt-metal checkout at
  `tools/scaleout/exabox/mpi-docker`)

No `python_env`, no `tt-run`, no local build required.

---

## 3. The single rule

**One `tt-run` / `mpi-docker` invocation at a time.** Wait for completion
before launching the next one. Two concurrent runs collide on chip locks.

---

## 4. Quick path

### Reset chips (run once at the start of a session and after any hang)

```bash
cd $TT_METAL_HOME
bash tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/scripts/reset_chips.sh
```

Wallclock ≈ 60 s. Defaults to QUAD_BH host list — override with
`MESH_DEVICE=...` and/or `HOSTS=...` for other configs. Use
`--help` for full usage.

### QUAD_BH (4 hosts — default, bare-metal)

```bash
cd $TT_METAL_HOME
bash tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/scripts/run_multi_host_ccl_tests.sh
```

### QUAD_BH with Docker (no local build needed)

```bash
cd $TT_METAL_HOME
bash tests/.../multi_host/scripts/run_multi_host_ccl_tests.sh \
  --image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:latest
```

### DUAL_BH (2 hosts)

```bash
MESH_DEVICE=DUAL_BH HOSTS="hostA,hostB" \
  bash tests/.../multi_host/scripts/run_multi_host_ccl_tests.sh
```

### DUAL_BH with Docker

```bash
MESH_DEVICE=DUAL_BH HOSTS="hostA,hostB" \
  bash tests/.../multi_host/scripts/run_multi_host_ccl_tests.sh \
  --image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:latest
```

### SINGLE_BH (1 host, no MPI)

```bash
MESH_DEVICE=SINGLE_BH \
  bash tests/.../multi_host/scripts/run_multi_host_ccl_tests.sh
```

### SINGLE_BH with Docker

```bash
MESH_DEVICE=SINGLE_BH \
  bash tests/.../multi_host/scripts/run_multi_host_ccl_tests.sh \
  --image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:latest
```

### Run a single test

```bash
bash tests/.../multi_host/scripts/run_multi_host_ccl_tests.sh \
  test_all_gather_exabox.py::test_all_gather_32x4
```

### Get help

```bash
bash tests/.../multi_host/scripts/run_multi_host_ccl_tests.sh --help
```

---

## 5. Manual `tt-run` invocation (bypass the wrapper)

### QUAD_BH

```bash
tt-run \
  --rank-binding tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \
  --mpi-args "--host bh-glx-b06u02,bh-glx-b06u08,bh-glx-b07u02,bh-glx-b07u08" \
  bash -c "source python_env/bin/activate && MESH_DEVICE=QUAD_BH \
    pytest --timeout=240 -v -k _32x4 \
    tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/"
```

### DUAL_BH

Substitute the binding (`16x4_dual_bh_…`), the host list, `MESH_DEVICE=DUAL_BH`,
and `-k _16x4`.

### SINGLE_BH

No `tt-run`:

```bash
source python_env/bin/activate
MESH_DEVICE=SINGLE_BH pytest --timeout=240 -v -k _8x4 \
  tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/
```

---

## 5b. Manual Docker invocation (bypass the wrapper)

### QUAD_BH via mpi-docker

```bash
cd $TT_METAL_HOME
./tools/scaleout/exabox/mpi-docker \
  --image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:latest \
  --empty-entrypoint \
  --host bh-glx-b06u02,bh-glx-b06u08,bh-glx-b07u02,bh-glx-b07u08 \
  -x TT_MESH_ID=0 \
  -x TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto \
  -x MESH_DEVICE=QUAD_BH \
  bash -c 'cd "$TT_METAL_HOME" && pytest --timeout=240 -v -k _32x4 \
    tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/'
```

Substitute the appropriate MGD path and host list for DUAL_BH.

---

## 6. Per-op mesh / fabric notes

| Op | FABRIC_1D works? | Mapper used by test | Cluster axis support |
|---|---|---|---|
| `all_gather` | ✅ | `ShardTensor2dMesh` | both axes |
| `all_reduce` | ✅ | `ShardTensor2dMesh` | both axes (PCC threshold 0.998 for 32-device axis=0) |
| `reduce_scatter` | ✅ | `ShardTensor2dMesh` | both axes (same threshold caveat) |
| `all_broadcast` | ✅ | `[Replicate, Shard(-1)]` w/ full `MeshShape` | both axes; needs sub-device |
| `broadcast` | ❌ — **requires `FABRIC_2D`** | 1D `MeshShape(1, N)` w/ `[Replicate, Shard(0)]` | needs sub-device + `subdevice_id` |
| `all_to_all_dispatch` | ✅ | helper-driven | `cluster_axis=1` only on multi-host |
| `all_to_all_combine` | ✅ | helper-driven | `cluster_axis=1` only on multi-host |

---

## 7. Troubleshooting

The most common failures:

### 7.1 `Device N init: failed to initialize FW`

Chips were left in a stale state. Reset:

```bash
bash tests/.../multi_host/scripts/reset_chips.sh
```

The script runs `tt-smi -glx_reset_auto` in parallel on every host in
your `HOSTS` (or locally for `SINGLE_BH`). Honors the same
`MESH_DEVICE`/`HOSTS` env vars as the runner.

### 7.2 PCC ≈ 0.5 with high ATOL on multi-host runs

`torch.rand(...)` was called without `torch.manual_seed(...)`. Each MPI
rank generates *different* random data, so per-rank goldens disagree.
Seed before the random generator inside the helper that builds inputs.

### 7.3 `pytensor.cpp:256: buffers.size() == 1`

Test was built with the legacy 1D mapper `ttnn.ShardTensorToMesh(device, dim=0)`.
Switch to `ShardTensor2dMesh` with `dims=(0, None)` (axis 0) or
`(None, 0)` (axis 1) plus `mesh_shape=mesh_shape`.

### 7.4 `MeshDevice has no devices`

You created a submesh that lives entirely on rank 0 (e.g. `(1, 4)` on a
16×4 mesh, where rank 0 owns rows 0–7). Don't create per-axis submeshes;
pass `cluster_axis` to the op directly.

### 7.5 Op hangs at 99% CPU with no log progress

Most common causes:

- `ttnn.broadcast` on a 2D mesh with `FABRIC_1D` — use `FABRIC_2D`.
- `ttnn.broadcast` without sub-device setup — missing `SubDevice` /
  `subdevice_id` argument.
- Prior run's chip locks not released — kill stragglers (`pkill -9 -f
  pytest`, ssh and same on every host) then reset chips.

### 7.6 PCC just below threshold (≈0.998 vs 0.999)

Real bfloat16 accumulation noise on 32-device reductions. Relax the
threshold to 0.998 for 32-device (axis=0) cases; keep 16-device strict.

### 7.7 L1 OOM on full-mesh MoE tests

Reduce per-device counts. Working values for 16×4 / 32×4:
`batches_per_device=2, experts_per_device=2, select_experts_k=2,
hidden_size=1024`. See `test_all_to_all_*_exabox.py`.

---

## 8. Logs and artifacts

Tests use pytest's stdout. To save and review:

```bash
LOG=/tmp/multi_host_$(date +%Y%m%d_%H%M%S).log
bash scripts/run_multi_host_ccl_tests.sh 2>&1 | tee "$LOG"

# strip ANSI for grepping
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "PASSED|FAILED|ERROR" | head
```

For per-rank inspection on multi-host runs (where `--tag-output` prefixes
each rank's output with `[1,N]`):

```bash
grep -E '^\[1,0\]<stdout>' "$LOG"   # rank 0
grep -aE 'PASSED|FAILED' "$LOG" | sort -u
```

---

## 9. Files in this folder

```
multi_host/
├── RUNBOOK.md                            ← this file
├── __init__.py                           ← marks the dir as a Python package
├── _a2a_moe_helpers.py                   ← shared all_to_all helpers
├── test_all_broadcast_exabox.py          ← all_broadcast
├── test_all_gather_exabox.py             ← all_gather
├── test_all_reduce_exabox.py             ← all_reduce
├── test_all_to_all_combine_exabox.py     ← all_to_all_combine (MoE)
├── test_all_to_all_dispatch_exabox.py    ← all_to_all_dispatch (MoE)
├── test_broadcast_exabox.py              ← broadcast
├── test_reduce_scatter_exabox.py         ← reduce_scatter
└── scripts/
    ├── run_multi_host_ccl_tests.sh       ← launcher (--help for usage)
    └── reset_chips.sh                    ← parallel chip reset (--help for usage)
```

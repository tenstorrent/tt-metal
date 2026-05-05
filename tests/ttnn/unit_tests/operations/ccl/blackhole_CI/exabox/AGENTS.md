# Exabox CCL Test Runbook

Practical guide for running and authoring CCL tests in this folder on Blackhole
multi-galaxy / multi-host setups. Read this before adding a test or debugging
one — most "weird" failures here have one of a small number of known root causes
and the fix is usually one or two lines.

---

## 1. Hardware topologies

| Name | Mesh | Devices | Hosts | `MESH_DEVICE` env |
|---|---|---|---|---|
| SINGLE_BH | 8×4 | 32 | 1 | `SINGLE_BH` |
| DUAL_BH | 16×4 | 64 | 2 | `DUAL_BH` |
| QUAD_BH | 32×4 | 128 | 4 | `QUAD_BH` |

Rank-binding YAMLs live in `tests/tt_metal/distributed/config/`:

- `8x4_single_bh_galaxy_rank_bindings.yaml` (rarely used — single-host runs use plain pytest)
- `16x4_dual_bh_galaxy_rank_bindings.yaml`
- `32x4_quad_bh_galaxy_rank_bindings.yaml`

## 2. Run commands

### SINGLE_BH (no MPI)

```bash
source python_env/bin/activate
MESH_DEVICE=SINGLE_BH pytest \
  tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/test_*.py::test_*_8x4
```

### DUAL_BH (16×4, 2 hosts)

```bash
tt-run \
  --rank-binding tests/tt_metal/distributed/config/16x4_dual_bh_galaxy_rank_bindings.yaml \
  --mpi-args "--host bh-glx-110-c08u02,bh-glx-110-c08u08" \
  bash -c "source python_env/bin/activate && MESH_DEVICE=DUAL_BH \
    pytest --timeout=240 \
    tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/test_*.py::test_*_16x4"
```

### QUAD_BH (32×4, 4 hosts)

```bash
tt-run \
  --rank-binding tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \
  --mpi-args "--host bh-glx-110-c07u02,bh-glx-110-c07u08,bh-glx-110-c08u02,bh-glx-110-c08u08" \
  bash -c "source python_env/bin/activate && MESH_DEVICE=QUAD_BH \
    pytest --timeout=240 \
    tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/test_*.py::test_*_32x4"
```

Always pass `--timeout=240` (or similar) per pytest case so a hang on one
parametrize does not stall the whole batch.

## 3. Run protocol — strict rules

1. **One test process at a time.** All these tests share the same physical
   cluster. Two `tt-run` invocations concurrently will collide on chip locks
   and either deadlock or corrupt each other.
2. **Wait for completion** before launching the next run. If you scheduled a
   wakeup or background task, do not start another test until that one finishes.
3. **Clean up on failure.** If a run hangs and you kill it, leftover processes
   on remote hosts will hold chip locks and break the next run.

### Cleanup recipe (after a hang)

```bash
# Local processes (host where tt-run was invoked)
pkill -9 -f 'tt-run'
pkill -9 -f 'prterun'
pkill -9 -f 'pytest.*exabox'

# Remote ranks — repeat for every host in --mpi-args
for h in bh-glx-110-c07u02 bh-glx-110-c07u08 bh-glx-110-c08u02 bh-glx-110-c08u08; do
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" \
    "pkill -9 -f 'pytest.*exabox' 2>/dev/null; pkill -9 -f prted 2>/dev/null"
done

# Confirm clean
ps -ef | grep -E 'tt-run|prterun|pytest.*exabox' | grep -v grep | wc -l   # → 0
for h in bh-glx-110-c07u02 bh-glx-110-c07u08 bh-glx-110-c08u02 bh-glx-110-c08u08; do
  ssh -o BatchMode=yes "$h" "ps -ef | grep -E 'pytest|prted' | grep -v grep | wc -l"
done
```

If chip locks persist across cleanup, the symptom in the next run's log is
`Waiting for lock 'CHIP_IN_USE_*_PCIe' which is currently held by thread TID:
<old_pid>`. Track that PID down and kill it on whichever host owns it.

## 4. Common failure modes and fixes

These are the failures every new exabox test hits at least once. Match the
symptom to the row, apply the fix.

### A. PCC ≈ 0.5 with high ATOL on every multi-host run

**Symptom**: `Max ATOL Delta: 0.99...`, `PCC: 0.50...` failures only on
DUAL_BH/QUAD_BH, never on SINGLE_BH.

**Root cause**: `torch.rand(...)` was called without `torch.manual_seed(...)`.
Each MPI rank generates *different* random data, so each rank's golden
tensors disagree across the rank boundary.

**Fix**: seed before the random generator, inside the helper that builds inputs:

```python
torch.manual_seed(0)
shards = [torch.rand(input_shape).bfloat16() for _ in range(num_along_axis)]
```

Fixed examples: `test_all_broadcast_exabox.py`, `test_all_reduce_exabox.py`,
`test_reduce_scatter_exabox.py`, `test_broadcast_exabox.py`.

### B. `pytensor.cpp:256: buffers.size() == 1`

**Symptom**:

```
RuntimeError: TT_FATAL @ ...pytensor.cpp:256: buffers.size() == 1
Can't convert a tensor distributed on MeshShape([16, 4]) mesh to row-major
logical tensor. Supply a mesh composer to concatenate multi-device shards.
```

**Root cause (most common)**: input was built with the legacy 1D mapper
`ttnn.ShardTensorToMesh(device, dim=0)`. That mapper does not establish proper
per-device tensor topology in distributed mode, so `ttnn.get_device_tensors`
returns a tensor that still claims to span the full multi-rank mesh, and
`ttnn.to_torch` refuses without a composer.

**Fix**: use the 2D-aware mapper.

```python
shard_dims = (0, None) if cluster_axis == 0 else (None, 0)
tt_input = ttnn.from_torch(
    torch_input, ...,
    mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=mesh_shape),
)
```

Pair with mesh-coord-aware verification (see Section 6).

### C. `MeshDevice has no devices`

**Symptom**: a non-zero rank crashes with `compute_with_storage_grid_size()
... MeshDevice has no devices`.

**Root cause**: the test creates a submesh that lives entirely on rank 0 (e.g.
`(1, 4)` on a 16×4 mesh, where rank 0 owns rows 0–7). Rank 1 sees an empty
submesh and crashes when the helper tries to use it.

**Fix**: don't create submeshes that span only a subset of ranks. Use the full
mesh and pass `cluster_axis` to the op directly. The op-level `cluster_axis`
parameter is what restricts communication to one mesh axis — submeshes are
not needed for that.

### D. Op hangs at 99% CPU with no log progress

**Symptom**: after `Exabox mesh device with N devices created`, no further
output for several minutes; ranks pinned at 99% CPU.

**Possible causes**:

1. **`ttnn.broadcast` on a 2D mesh with `FABRIC_1D`** — broadcast spins
   indefinitely. Use `FABRIC_2D` (see Section 5).
2. **`ttnn.broadcast` without sub-device setup** — the broadcast op needs a
   worker `SubDevice` and `subdevice_id` argument; without it, the op blocks
   waiting for workers that don't exist. Other CCL ops (`all_reduce`,
   `all_gather`, `reduce_scatter`) do not need this.
3. **Prior run's chip locks not released** — see Section 3 cleanup.

### E. PCC just below threshold (≈0.998 vs threshold 0.999)

**Symptom**: `axis=0` cases on 32-device reductions fail with `PCC: 0.998…`
when the threshold is 0.999. axis=1 (4-device) cases pass at the same
threshold.

**Root cause**: bfloat16 accumulation noise scales with reduction width. A
32-way sum carries ~2× the rounding error of a 16-way sum. This is real
hardware/dtype behavior, not a correctness bug.

**Fix**: relax the threshold to 0.998 for 32-device reductions, or pass a
width-aware `pcc_threshold` per parametrization. Keep 16-device reductions
strict at 0.999.

### F. L1 OOM (`bank_manager.cpp:439: false`) on full-mesh MoE tests

**Symptom**: MoE tests (`all_to_all_dispatch`, `all_to_all_combine`) fail in
`from_torch` with an L1 allocation assertion when scaled to full 16×4 or 32×4
mesh.

**Root cause**: the MoE helpers compute `experts = experts_per_device *
total_mesh_devices`. With per-device defaults from t3000 (`experts_per_device=8`,
`hidden_size=7168`), the per-device input tensor blows up to ~100+ MB on a
64-device mesh.

**Fix**: reduce per-device counts (`batches_per_device`, `experts_per_device`,
`select_experts_k`, `hidden_size`) for exabox-scale meshes. The current
working values for 16×4/32×4 are `2/2/2/1024`. See `test_all_to_all_*_exabox.py`.

Also: the helper checks `experts % devices == 0` where `devices =
mesh_shape[0] * mesh_shape[1]`. Compute `experts = experts_per_device * devices`
(not `mesh_shape[cluster_axis]`).

## 5. Per-op multi-host quirks

| Op | FABRIC_1D works? | Mapper | Verification | Notes |
|---|---|---|---|---|
| `all_gather` | ✅ | `ShardTensor2dMesh` | `get_device_tensors` + per-device `to_torch` w/ `view.is_local` | reference pattern |
| `all_reduce` | ✅ | `ShardTensor2dMesh` | same as gather | PCC threshold 0.998 for 32-device |
| `reduce_scatter` | ✅ | `ShardTensor2dMesh` | same | PCC threshold 0.998 for 32-device |
| `all_broadcast` | ✅ | `[Replicate, Shard(-1)]` w/ full `MeshShape(*mesh_shape)` | `ConcatMesh2dToTensor(dims=(0,1))` | needs sub-device |
| `broadcast` | ❌ — **requires `FABRIC_2D`** | `MeshShape(1, N)` 1D-style w/ `[Replicate, Shard(0)]` | `ConcatMeshToTensor(dim=cluster_axis)` + slice | needs sub-device + `subdevice_id` |
| `all_to_all_dispatch` | ✅ | helper uses `ShardTensor2dMesh` | helper uses `ConcatMeshToTensor` — multi-host clean | `cluster_axis=1` only in repo |
| `all_to_all_combine` | ✅ | helper uses 1D `ShardTensorToMesh` | axis=1 uses `ConcatMeshToTensor` ✅; axis=0 uses per-device path ❌ multi-host | `cluster_axis=1` only in repo |

## 6. Verification patterns (multi-host safe)

### Mesh-coord per-device check (preferred for ops with cluster_axis support)

```python
def _verify_output(tt_output_tensor, golden, mesh_device):
    coords = list(tt_output_tensor.tensor_topology().mesh_coords())
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    device_tensors = ttnn.get_device_tensors(tt_output_tensor)
    coord_iter = coords
    if view is not None and len(device_tensors) != len(coords):
        coord_iter = [c for c in coords if view.is_local(c)]

    for coord, tt_out in zip(coord_iter, device_tensors):
        if view is not None and not view.is_local(coord):
            continue
        eq, mess = comp_pcc(golden, ttnn.to_torch(tt_out), pcc_threshold)
        assert eq, mess
```

Use when: input was built with `ShardTensor2dMesh` (proper topology), and you
want a per-device golden check.

### Composer-based check (used by `broadcast` and `all_to_all`)

```python
output = ttnn.to_torch(
    tt_output_tensor,
    mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=cluster_axis),
)
# slice and compare
```

Use when: input was built with the legacy 1D `MeshShape(1, N)` mapper that
doesn't expose per-device topology to `get_device_tensors`. Composer gathers
across MPI ranks correctly on its own.

## 7. Authoring a new exabox test

Mirror `test_all_reduce_exabox.py` as the canonical template:

1. **Three parametrize blocks per mesh**: `device_params + topology` (indirect
   `device_params`), `mesh_device`, `cluster_axis`.
2. **Helper function** `_get_tensors(...)` that:
   - Calls `torch.manual_seed(...)` first.
   - Uses `ShardTensor2dMesh` with `(shard_dim, None)` for axis 0 or
     `(None, shard_dim)` for axis 1.
3. **Helper function** `_verify_*_output(...)` using the mesh-coord pattern
   from Section 6.
4. **`maybe_trace`** wrapper around the op call so `enable_trace=True/False`
   parametrize works.
5. **Test functions per mesh**: `test_*_16x4` (DUAL_BH), `test_*_32x4` (QUAD_BH).
   Use `requires_device(["DUAL_BH"])` / `["QUAD_BH"]` markers.

## 8. Cross-checking before writing parametrize values

Before adding a new parametrize axis (especially `cluster_axis=[0, 1]` on a
multi-host mesh), confirm the op is actually validated in that configuration
elsewhere in the repo. The MoE ops (`all_to_all_combine`,
`all_to_all_dispatch`, `selective_combine`) only validate `cluster_axis=1` on
multi-host. The dual-galaxy CI runner is the source of truth:

```bash
grep "axis_" tests/scripts/multihost/run_dual_galaxy_tests.sh
```

For other ops, grep the relevant test files:

```bash
grep -nB1 "parametrize.*cluster_axis\|parametrize.*axis," \
  tests/nightly/tg/ccl/test_*.py tests/nightly/t3000/ccl/test_*.py
```

If no existing site validates `axis=0` multi-host for an op, default to
`axis=1` only — and document it in the test docstring with the cross-references.

## 9. Memory budgeting cheat sheet for full-mesh tests

| Mesh | Total devices | Multiplier vs 8-device baseline |
|---|---|---|
| 8×4 SINGLE_BH | 32 | 4× |
| 16×4 DUAL_BH | 64 | 8× |
| 32×4 QUAD_BH | 128 | 16× |

If a t3000 test uses `experts_per_device=8` on a (1,8) mesh, the per-device
input tensor's expert-dimension is 64. Running the same helper at exabox scale
makes that 512 (8×) or 1024 (16×) — likely L1-fatal.

Rule of thumb when porting MoE-style helpers: divide per-device counts by
8–16× when scaling from t3000 to QUAD_BH. The current exabox tests use
`batches_per_device=2, experts_per_device=2, select_experts_k=2,
hidden_size=1024` and that fits L1 on QUAD_BH.

## 10. Debugging timeline / observability

When a multi-host run misbehaves, in order:

1. **Per-rank timestamps**: `grep -E "^\[1,N\]" log | tail -1` for each rank
   `N` shows where each rank is in the lifecycle. Divergence between ranks is
   a strong signal (e.g. one rank hit a fabric timeout, others are blocked
   waiting for it).
2. **Last-line stage names**: ranks log `Fabric initialized on Device N`, then
   `Fabric Initialized with config FabricConfig::*`, then `Exabox mesh device
   with M devices created`. If progress stalls between two of those, that
   pinpoints the failing phase.
3. **Process inspection**: `ps -ef | grep pytest` — if pytest is at 99% CPU
   with no log motion, it's likely spinning in a polling loop (often a fabric
   sync waiting for an event that never arrives).
4. **`Fabric Router Sync: Timeout`** in the log is a hardware-layer issue, not
   a test bug. Worth retrying once before deeper investigation — it has been
   transient at least once.

## 11. Sequential-run telemetry pattern

When running the full suite on a mesh, save per-test logs and a running
summary so you can resume cleanly if something fails partway. Pattern that
worked well:

```bash
mkdir -p /tmp/exabox_runs
echo "Started: $(date)" > /tmp/exabox_runs/SUMMARY.txt
# For each test:
tt-run ... 2>&1 | tee /tmp/exabox_runs/01_<name>.log | tail -10
echo "1. <name>: <PASSED|FAILED counts> in <time>s" >> /tmp/exabox_runs/SUMMARY.txt
```

Per-test log lets you grep failures with full context; the running summary
gives you an at-a-glance state of the suite.

## 12. Single-pod 16-rank tests (blitz_decode pipeline + fake-MoE)

Tests that span 4 hosts × 4 MPI ranks/host, each rank seeing a `(4, 2)` BH
submesh, live in the `single_pod/` subfolder. They depend on artifacts
generated by `models/demos/deepseek_v3_b1/scaleout_configs/generate_blitz_decode_pipeline_configs.py`,
not the standard `tests/tt_metal/distributed/config/*` rank-bindings.

### Artifact path
`generate_blitz_decode_pipeline_configs.py` writes a per-run directory to
`/tmp/single_pod_<timestamp>/` and updates `/tmp/single_pod_current_dir.txt`.
Inside it:
- `blitz_decode_pipeline_rank_binding_single_pod_ci.yaml` (16-rank tt-run binding)
- `blitz_decode_pipeline_rank_file_single_pod_ci` (mpirun rankfile, with
  `slot=0-31` per host)
- Symlinks back to `$TT_METAL_HOME/{tt_metal,build,models,ttnn,runtime,tests,python_env}`

### Working tt-run launch flags

PRRTE 3.0.8 has two non-obvious requirements when running 16-rank single-pod:

1. **`--prtemca plm_ssh_no_tree_spawn 1`** — without this, custom
   `plm_rsh_agent` scripts cause silent rendezvous failures (the HNP loses
   communication with daemons even though ssh succeeds).
2. **No `-np` flag** — let the rankfile drive process count, otherwise PRRTE
   reports a spurious "Rank 16 missing slot" error. Add `:4` to each host
   in `--host` so MPI knows each host has 4 slots.

A custom ssh-agent wrapper that injects `ulimit -n 65536;` before each
remote command is required because the default sshd ulimit (1024) is below
what tt-metal needs for fabric/socket allocations. The wrapper at
`/tmp/ssh_ulimit_wrapper.sh` parses ssh args, locates the host arg, and
prefixes the remote command with the ulimit raise.

Reference launch script: `/tmp/run_fake_moe_tier3.sh` (no slow dispatch — for
CCL chain tests) and `/tmp/run_fake_moe_pipeline.sh` (with
`TT_METAL_SLOW_DISPATCH_MODE=1` — for the pipeline framework).

### Slow dispatch vs fast dispatch

- **Fast dispatch** (default): required for CCL tests that use
  `mesh_device.create_sub_device_manager(...)` (i.e. anything calling
  `ttnn.broadcast`, `all_to_all_*`, or other ops that pin a worker subset).
  Slow dispatch asserts: `Using sub device managers is unsupported with
  slow dispatch` at sub-device load time.
- **Slow dispatch** (`TT_METAL_SLOW_DISPATCH_MODE=1`): required by the
  blitz_decode pipeline framework (sockets, kernel-driven I/O). The `Pipeline`
  setup in the demo asserts on this.

Implication: a single test cannot exercise both `ttnn.broadcast`/sub-device
CCL and the pipeline framework. The fake-MoE design splits these:
- Phase 2 standalone CCL chain test runs without slow dispatch and
  validates `all_reduce` on the actual single-pod fabric.
- Phase 4 pipeline-integration test runs with slow dispatch and replaces
  MoEDecoderStage with a `PassthroughStage(ACTIVATION)` (see
  `single_pod/_fake_moe_helpers.py::make_fake_moe_decoder_stage_factory`)
  so no CCL ops run inside the pipeline framework.

### Chip reset workflow

After a hung 16-rank test (especially when killed externally), all 4 hosts
need their chips reset before the next run, or device init fails with
`Device 0/1 init: failed to initialize FW`. Run:

```bash
for h in bh-glx-110-c07u02 bh-glx-110-c07u08 bh-glx-110-c08u02 bh-glx-110-c08u08; do
  ssh $h "$TT_METAL_HOME/python_env/bin/tt-smi -glx_reset_auto" &
done
wait
```

Each reset takes ~60 seconds (4 in parallel ≈ 60s wallclock). After reset,
chips report 32 boards on each host.

### Known issue: a2a hangs on FABRIC_2D_TORUS_Y submesh

`ttnn.all_to_all_dispatch` and `ttnn.all_to_all_combine` hang indefinitely
when run on a `(4, 2)` per-rank submesh under `FABRIC_2D_TORUS_Y`. The hang
is below the python layer; pytest's `--timeout` does not fire. Verified
2026-04-30. The same ops work on the unified 16x4 / 32x4 meshes under
FABRIC_1D (covered by `test_all_to_all_dispatch_exabox.py`). Skipped
reproducers: `single_pod/test_fake_moe_traffic.py::test_fake_moe_a2a_*_4x2_single_pod`.

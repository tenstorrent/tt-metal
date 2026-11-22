# Cross-Host Communication

How to get chips on two (or more) hosts talking to each other with TT-Fabric, and
how to verify it actually works.

This is a practical guide derived from bringing up a 2-host Blackhole Galaxy
fabric. Every command, error message and caveat below was reproduced on real
hardware.

---

## 1. The two planes

Cross-host communication is **two independent planes**, and both must work:

| Plane | Carries | Transport | Where in code |
|---|---|---|---|
| **Host plane** | Control + metadata only (descriptors, handshake, rank agreement) | MPI over TCP/RDMA between hosts | `DistributedContext` (`tt_metal/api/tt-metalium/distributed_context.hpp:153`) |
| **Device plane** | Actual tensor data | QSFP-DD Ethernet cables, chip-to-chip, driven by TT-Fabric | `PhysicalSystemDescriptor`, `ControlPlane`, `MeshSocket` |

The host plane never carries tensor payloads. Chips on different hosts are cabled
**directly to each other** — data does not go through the host NIC.

```
        HOST A                                        HOST B
   ┌──────────────┐                              ┌──────────────┐
   │ 32 chips     │   QSFP-DD cables (fabric)    │ 32 chips     │
   │              │◄════════════════════════════►│              │
   └──────┬───────┘        device plane          └──────┬───────┘
          │                                            │
          └──────────► MPI over TCP (host plane) ◄──────┘
```

On Blackhole UBB (Galaxy) the external ports are **QSFP-DD on Ethernet channels
8 and 9**. Port-to-channel mapping is defined in `tools/scaleout/board/board.cpp`.

---

## 2. How cross-host links are discovered

You do not declare cables. Each rank discovers what its own chips can see, then
the ranks exchange and merge descriptors.

1. **Local discovery.** UMD reads each local chip's exit nodes — the *remote chip
   unique ID* read from the link partner's registers. See
   `cluster_desc.get_ethernet_connections_to_remote_devices()` at
   `tt_metal/fabric/physical_system_discovery.cpp:721`, and exit-node population
   at `:795`.

2. **Exchange.** Every rank serializes its `PhysicalSystemDescriptor` and swaps it
   with peers over MPI (`exchange_metadata`, `:615`), then merges (`:691`).

3. **Pairing.** Rank 0 walks the merged exit-node table and pairs each cable by
   matching **reciprocal chip ID and reciprocal channel** — the link must be
   visible from both ends (`generate_cross_host_connections`, `:364`).

4. **Pruning.** Anything that fails to pair is discarded with
   `"Dropping one-sided link."` (`:566`, `:587`). Only surviving links exist as
   far as fabric is concerned.

Each ASIC records the `host_name` and `mpi_rank` that discovered it, so multiple
ranks (or containers) on one machine are distinguished.

### Why this matters in practice

A link is only usable if **both endpoints agree**. If host A sees a cable to host
B's chip 42, but host B reports that cable connects to something local, the pair
never forms and the link is silently dropped. This is the single most common
failure mode — see the troubleshooting table.

---

## 3. Prerequisites

### 3.1 Hardware

- Both hosts have chips visible: `ls /dev/tenstorrent/ | grep -c '^[0-9]'`
- Matching firmware bundles across hosts. Verify with `tt-smi -s` on each and
  compare `firmwares.fw_bundle_version` and `eth_fw`.
- Matching `TT-KMD` driver version.

### 3.2 Host networking

SSH from the launching host to every other host, passwordless, for the `ttuser`
account (PRRTE spawns daemons over SSH).

> **If `/etc/hosts` points at a dead subnet**, alias the live IPs in
> `~/.ssh/config`. On the reference cluster `/etc/hosts` mapped names to a dead
> `10.81.14.0/24` while the working network was `10.82.97.0/24`:
>
> ```
> Host DC16-2-BG-0202-u13-41
>   HostName 10.82.97.41
> ```

Identify the correct TCP interface for MPI:
`ip -br a` → pick the address in use, e.g. `ens5f0np0.1`.

### 3.3 A filesystem that is actually shared — or don't assume it is

**`/big_home` is NOT shared on the reference cluster.** It is a separate local
ext4 volume on *each* host. Same path, different disks:

| Host | Device | Used |
|---|---|---|
| 40 | `/dev/mapper/vg_home-lv_home` ext4 | 2.0 T |
| 41 | `/dev/mapper/vg_home-lv_home` ext4 | 2.2 T |

Consequences:

- Every file you create/edit is **host-local by default**.
- If host 41 lacks your file, `python` exits 2 and `prterun` kills the whole job.
- Worse: a **stale** copy can still run and give misleading results.

Check once, then decide how to sync:

```bash
df -hT /big_home            # same device name + same Used  => shared
mount | grep -E "nfs|cifs|9p|fuse"
```

If not shared, either `scp` your changed files before every run, or keep the
workload on genuinely shared storage (e.g. an NFS mount).

Quick sync helper for a two-host setup:

```bash
for f in \
  tests/tt_metal/tt_fabric/custom_mesh_descriptors/my_mgd.textproto \
  tests/tt_metal/distributed/config/my_rank_bindings.yaml \
  tests/tt_metal/distributed/config/my.rankfile \
  tests/tt_metal/multihost/my_script.py ; do
  scp -q "$f" "HOSTB:$PWD/$f"
done
```

Verify with checksums, not with hope:

```bash
diff <(md5sum < "$f") <(ssh HOSTB "md5sum < $PWD/$f")
```

### 3.4 Both hosts must have build and venv

Each host runs the binaries and Python itself. Confirm on every host:

```bash
ls build/lib/_ttnncpp.so build/lib/libtt_metal.so build/tools/scaleout/run_cluster_validation
source python_env/bin/activate && python -c "import ttnn; print('ok')"
```

---

## 4. Configuration: three files

Cross-host workloads need a mesh graph descriptor, rank bindings, and a rankfile.

### 4.1 Mesh Graph Descriptor (MGD)

Defines the logical mesh and how it is partitioned across hosts. Schema:
`tt_metal/fabric/protobuf/mesh_graph_descriptor.proto`. See also
[`MGD_README.md`](MGD_README.md).

For **two 8×4 Galaxies joined into one 16×4 mesh**, one rank per host:

```proto
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [ 16, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 2, 1 ] }
  channels { count: 2 policy: RELAXED }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
```

Key points:

- `host_topology { dims: [2, 1] }` says the mesh is split across **2 hosts along
  dim 0**. That is what makes `cluster_axis=0` cross the host boundary.
- **One mesh spanning both hosts**, not two separate meshes. Collectives like
  `ttnn.all_reduce` shard along an axis *within* a single mesh — two independent
  meshes cannot all-reduce with each other.
- `policy: RELAXED` tolerates fewer usable lanes than declared. `STRICT` (the
  default when `policy` is omitted) hard-fails if the count is not met.

**Validate the descriptor offline** before running anything — this catches typos
with no hardware access:

```bash
protoc --experimental_allow_proto3_optional \
  --proto_path=tt_metal/fabric/protobuf \
  --encode=tt.tt_fabric.proto.MeshGraphDescriptor \
  tt_metal/fabric/protobuf/mesh_graph_descriptor.proto \
  < my_mgd.textproto > /dev/null && echo "MGD parses OK"
```

### 4.2 Rank bindings

Maps MPI ranks to mesh/host ranks and per-rank environment.

```yaml
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,...,31"

  - rank: 1
    mesh_id: 0
    mesh_host_rank: 1
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1,2,...,31"

mesh_graph_desc_path: "tests/tt_metal/tt_fabric/custom_mesh_descriptors/my_mgd.textproto"
```

- Both ranks use **`mesh_id: 0`** — the mesh is split by host, not duplicated.
- `mesh_host_rank` must increment per host.
- `TT_VISIBLE_DEVICES` must expose **all local chips** (32 for a Galaxy). A
  too-small list is a common silent error: an 8×4 mesh needs 32 chips, not 16.

### 4.3 Rankfile

Pins each MPI rank to a host. Must agree with the rank bindings.

```text
rank 0=HOSTA slot=0
rank 1=HOSTB slot=0
```

---

## 5. Launching

`tt-run` wraps `mpirun`, exports per-rank env (`TT_MESH_ID`,
`TT_MESH_HOST_RANK`, `TT_MESH_GRAPH_DESC_PATH`, `TT_VISIBLE_DEVICES`), and adds
the multi-host MPI options. Docs: [`ttnn/ttnn/distributed/README_ttrun.md`](../../ttnn/ttnn/distributed/README_ttrun.md).

### 5.1 Two modes

| Mode | How | Needs hardware? |
|---|---|---|
| **Auto allocation** | `tt-run -m <mgd> --hosts A,B ./app` | **Yes** — Phase 1 runs `generate_rank_bindings`, which does live discovery |
| **Legacy / explicit** | `tt-run --rank-binding <yaml> --mpi-args "..." ./app` | No — skips Phase 1 |

Use **legacy mode** when you want a reproducible config or cannot run discovery.
Use auto-allocation when you want the tool to derive and self-check bindings.

### 5.2 Full command (legacy mode)

```bash
cd $TT_METAL_HOME
source python_env/bin/activate

tt-run \
  --rank-binding tests/tt_metal/distributed/config/my_rank_bindings.yaml \
  --mpi-args "--host HOSTA,HOSTB \
              --map-by rankfile:file=$PWD/tests/tt_metal/distributed/config/my.rankfile \
              --mca btl self,tcp --mca btl_tcp_if_include ens5f0np0.1 \
              --bind-to none --tag-output" \
  python tests/tt_metal/multihost/my_script.py
```

- `--tag-output` prefixes each line with `[world,rank]`; essential for multi-host
  debugging.
- Always add `--dry-run` first. It prints the exact `mpirun` command and per-rank
  env without executing anything — the fastest way to catch a bindings mistake.

```bash
tt-run --dry-run --rank-binding ... --mpi-args "..." true
```

---

## 6. Verifying it works

Do this in two stages. **Do not skip stage 1** — it isolates cabling problems
from workload problems.

### Stage 1 — Does discovery see the links?

`run_cluster_validation` performs live discovery and prints the cross-host map.
It needs **no golden descriptor** for a basic run.

```bash
tt-run \
  --rank-binding tests/tt_metal/distributed/config/my_rank_bindings.yaml \
  --mpi-args "--host HOSTA,HOSTB --map-by rankfile:file=$PWD/config/my.rankfile \
              --mca btl self,tcp --mca btl_tcp_if_include ens5f0np0.1 \
              --bind-to none --tag-output" \
  ./build/tools/scaleout/run_cluster_validation \
    --print-connectivity --send-traffic --num-iterations 20 --log-ethernet-metrics
```

Read these lines specifically:

| Look for | Good | Bad |
|---|---|---|
| `Detected Hosts:` | all expected hosts | missing a host |
| `CROSS-HOST CONNECTIONS` section | present, lists cables | **absent entirely** |
| `Dropping N one-sided connection(s)` | absent, or `0` | `N > 0` |
| `remote chip ids {…}` | non-empty (in UMD logs) | `{}` on both ranks |

`--send-traffic` sends real packets across every detected link and reports CRC
errors and retrains — this proves the cables carry data, not just that they were
enumerated.

> **Warning: "✓ All Detected Links are healthy" is not a pass.**
> It validates *the links it discovered*. A completely disconnected cluster can
> report this while printing `Dropping 15 one-sided connection(s)`. Always check
> the drop count and the presence of a `CROSS-HOST CONNECTIONS` section.

### Stage 2 — Does data cross correctly?

Then run the workload. A minimal, self-verifying check shards a tensor along the
host-spanning axis, reduces over fabric, and compares every device shard to a
torch reference. See
`tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/test_all_reduce_exabox.py`
for the canonical pattern.

```python
import ttnn, torch

# 1. MPI world is NOT initialized just by running under tt-run.
if not ttnn.distributed_context_is_initialized():
    ttnn.init_distributed_context()

# 2. Fabric config MUST be set before opening the mesh.
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

# 3. COL dispatch is required on Blackhole unless fabric_tensix_config is also set.
dispatch = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL, None)

mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(16, 4), dispatch_core_config=dispatch)

# 4. Shard along the axis that crosses the host boundary.
t = ttnn.from_torch(
    torch.cat([torch.rand(1, 1, 32, 224).bfloat16() for _ in range(16)], dim=0),
    layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh,
    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, None), mesh_shape=(16, 4)),
)
out = ttnn.all_reduce(t, cluster_axis=0, topology=ttnn.Topology.Linear, num_links=2)
```

Two API traps worth knowing:

- **The distributed context is not created by `tt-run`.** It is normally created
  as a side effect of `open_mesh_device()`. If you need rank/size *first*, call
  `ttnn.init_distributed_context()` explicitly. See
  `tests/ttnn/distributed/test_multi_mesh.py:28`.
- **`mesh_device.get_num_devices()` returns the GLOBAL mesh size**, not the local
  count. For a 16×4 mesh on 2 hosts it returns 64. Per-rank count comes from
  `mesh_device.get_view().get_local_mesh_coord_range()`.

---

## 7. Troubleshooting

Every entry below is a real failure observed during bring-up.

### 7.1 `Dropping N one-sided connection(s)`, `remote chip ids {}`

**Symptom**
```
Physical Discovery Warning: Global Connection between host A and host B
  is not found in the host connectivity graph. ASIC … (Tray 1 Location 1 chan 8)
  -> ASIC … (Tray 3 Location 1 chan 8). Dropping one-sided link.
Physical Discovery: Dropping 15 one-sided connection(s).
```

**Cause** — the two endpoints disagree about what a cable connects to. Usual
reasons, in order of likelihood:

1. **Chips were physically moved between hosts** since the cabling was recorded.
   Cables stay in fixed tray/ASIC positions; if the ASIC behind a port changed,
   a cable that was cross-host becomes local at one end.
2. **Firmware reflash left links un-retrained.** See 7.2 — this is the big one.
3. Physically unseated or failed QSFP cable/transceiver.

**Diagnose** — compare `remote chip ids {}` per rank. If one rank emits all the
warnings and the other emits none, the problem is asymmetric and points at that
host's links.

**Fix** — reset the chips on **both** hosts (7.2). If that does not restore
links, inspect cabling physically.

### 7.2 After a firmware reflash: run `glx_reset_auto`

This is the highest-value item in this document.

A `tt-flash` of a new firmware bundle (e.g. `19.11.0 → 19.17.1`) can leave QSFP
links un-retrained. Discovery then finds **zero** remote chips, and no amount of
MGD/rank-binding tuning helps.

**Symptom** — the cluster looks healthy (all chips present, firmware matching on
both hosts) but there is no cross-host section and `remote chip ids {}` on every
rank.

**Fix**

```bash
# All hosts in parallel — takes ~60 s
MESH_DEVICE=DUAL_BH HOSTS="HOSTA,HOSTB" \
  bash tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/scripts/reset_chips.sh
```

Or per host: `tt-smi -glx_reset_auto`.

**Rule:** after any reflash, reset **every** host before expecting fabric to work.
A reset on only one host can fix an asymmetric subset of links and leave you
chasing phantom problems.

### 7.3 `python: can't open file … No such file or directory` on one rank

**Cause** — `/big_home` (or the repo) is **not shared** between hosts; the peer
lacks the file. Exit code 2 from one rank aborts the whole job.

**Fix** — `scp` the file to each host, and verify checksums (section 3.3).

### 7.4 `Distributed context not initialized`

```
RuntimeError: Distributed context not initialized. Call init_distributed_context() first.
```

**Cause** — reading `distributed_context_get_rank()`/`_get_size()` before any
device has been opened. `tt-run` starts MPI but does not create ttnn's context
object.

**Fix**

```python
if not ttnn.distributed_context_is_initialized():
    ttnn.init_distributed_context()
```

### 7.5 `SAT placement: mesh 'M0' has no PGD grouping and no embeddable MGD fallback`

```
Physical groupings: Mesh graph descriptor 'M0': no PGD grouping could be placed on the PSD
TT_FATAL: Internal error: SAT placement: mesh 'M0' (mesh 0) has no PGD grouping and no embeddable MGD fallback
```

**Cause** — almost always **downstream of 7.1/7.2**, not a descriptor problem. If
discovery found no cross-host links, there is no valid way to place a mesh whose
`host_topology` requires two hosts. The mapper is correctly refusing an
impossible layout.

**Fix** — fix discovery first. Do not edit the MGD to work around this; verify
with stage 1 that `CROSS-HOST CONNECTIONS` is present before touching the
descriptor. `RELAXED` policy, pinnings and channel counts cannot conjure links
that were already pruned.

### 7.6 `has 4 eth channels, but only 2 routing planes are available`

```
phys 7 M0D46 in direction 0 has 4 eth channels, but only 2 routing planes are available
```

**Benign.** `control_plane.cpp:1577` trims extra lanes to the number of routing
planes (`directional_eth_chans.resize(num_available_routing_planes)`). The
topology remains valid; you simply use fewer lanes per direction.

**Only act** if you want more per-direction bandwidth: raise `num_planes` in
`set_fabric_config(..., num_planes=2)`, or move to `FABRIC_2D`.

### 7.7 Fabric init succeeds but collectives give wrong results / PCC ≈ 0.5

**Cause** — each rank generated *different* random input. Multi-host tests need a
shared seed, or the reference and the device data diverge for reasons unrelated
to fabric.

**Fix** — `torch.manual_seed(0)` before generating inputs, on every rank.

### 7.8 Hang at 99 % CPU, no progress

**Cause** — stale chip locks from a previous killed run.

**Fix** — kill stragglers on **every** host, then reset:

```bash
pkill -9 -f pytest     # on each host
tt-smi -glx_reset_auto # on each host
```

---

## 8. Checklist

Before blaming your code, confirm in order:

1. [ ] `/dev/tenstorrent` has all expected chips on every host.
2. [ ] `fw_bundle_version` and `eth_fw` match across hosts (`tt-smi -s`).
3. [ ] SSH works from the launcher to every host.
4. [ ] Your files exist on **every** host, with matching checksums.
5. [ ] MGD parses offline (`protoc --encode`).
6. [ ] `tt-run --dry-run` shows the expected `TT_MESH_ID`, `TT_MESH_HOST_RANK`,
       `TT_VISIBLE_DEVICES` per rank.
7. [ ] Stage 1 discovery shows `CROSS-HOST CONNECTIONS` with **0** one-sided drops.
8. [ ] Only then: run the workload, and compare against a shared-seed reference.

---

## 9. Reference configuration

A complete, verified 2-host setup lives in this repo:

| File | Purpose |
|---|---|
| `tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_8x4_hosts_40_41_mesh_graph_descriptor.textproto` | One 16×4 mesh, `host_topology [2,1]` |
| `tests/tt_metal/distributed/config/dual_8x4_hosts_40_41_rank_bindings.yaml` | 2 ranks, both `mesh_id 0`, host ranks 0/1 |
| `tests/tt_metal/distributed/config/dual_8x4_hosts_40_41.rankfile` | Rank → host pinning |
| `tests/tt_metal/multihost/test_fabric_40_41.py` | `all_reduce` across the boundary, DRAM + L1, verified per shard |

Measured on that configuration: 32 cross-host channels on ch8/ch9, 8 exit chips
per host (trays 1–2), `all_reduce` passing across 64 devices.

---

## See also

- [`MGD_README.md`](MGD_README.md) — writing mesh graph descriptors
- [`README_fabric_reliability_mode.md`](README_fabric_reliability_mode.md) — STRICT vs RELAXED
- [`PHYSICAL_GROUPING_DESCRIPTOR_README.md`](PHYSICAL_GROUPING_DESCRIPTOR_README.md) — pinning/placement
- [`../../ttnn/ttnn/distributed/README_ttrun.md`](../../ttnn/ttnn/distributed/README_ttrun.md) — `tt-run` launcher
- [`../../tools/scaleout/validation/README.md`](../../tools/scaleout/validation/README.md) — cluster validation tool
- [`../../tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/RUNBOOK.md`](../../tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/RUNBOOK.md) — multi-host CCL test runbook

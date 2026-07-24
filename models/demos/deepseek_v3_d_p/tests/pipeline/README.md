# Cross-mesh socket smoke + perf

A 2-rank test that drives `H2D socket → cross-mesh D2D socket → D2H socket` on
either:

- **One Blackhole galaxy** carved into two `(2, 4)` meshes (whole UBB trays).
- **Two Blackhole galaxies** as two `(4, 8)` meshes connected by inter-galaxy
  ethernet.

Same Python test runs in both modes; only the rank-binding YAML and the
launcher differ.

The test verifies (a) cross-mesh `MeshSocket` data exchange is bit-exact and
(b) measures end-to-end bandwidth for a configurable tensor shape (default
`640×1792 bf16` = 2,293,760 bytes).

It exists to validate the cross-mesh fabric path the production decode demo
relies on, and as a building block for prefill→decode KV migration tests.


## What's in here

```
test_cross_mesh_socket_smoke.py        — pytest (smoke + perf variants)
runme_cross_mesh_smoke.sh              — single-galaxy launcher
runme_2galaxy_cross_mesh_smoke.sh      — 2-galaxy launcher
run_2galaxy_end_to_end.sh              — 2-galaxy launcher with recovery
dual_tray_2x4_rank_bindings.yaml       — single-galaxy 2-rank binding (trays 1+2)
dual_galaxy_rank_bindings.yaml         — 2-galaxy 2-rank binding (1 rank per host)
bh_galaxy_dual_2x1_minimal.textproto   — (kept for reference; not viable)
dual_2x1_minimal_rank_bindings.yaml    — (kept for reference; not viable)
```

The python test parametrizes on:

- `mesh_device`: `(2, 4)` ids=`1galaxy`, `(4, 8)` ids=`2galaxy` — selected via `-k 1galaxy` or `-k 2galaxy`.
- `page_size_bytes` (perf only): `256, 1024, 2048, 4096, 8192, 16384` — sweep
  to map the bandwidth-vs-page-size curve.


## Prerequisites

### 1. tt-metal built with the ETH HAL bump

Commit `7e1579142a3` on `jjovicic/socket-experiment` bumps
`MEM_ERISC_KERNEL_CONFIG_SIZE` from 25 KB to 32 KB in
`tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h`. The upstream 25 KB
limit doesn't fit the current `FABRIC_2D` router program (~26000 B) for
intra-galaxy cross-mesh paths with `assign_z_direction: true`.

Verify the build picked it up:

```bash
grep MEM_ERISC_KERNEL_CONFIG_SIZE \
    tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h
# Expect: #define MEM_ERISC_KERNEL_CONFIG_SIZE (32 * 1024)
```

If you see `(25 * 1024)`, `git checkout jjovicic/socket-experiment`, clear
the kernel cache, and rebuild:

```bash
rm -rf ~/.cache/tt-metal-cache/
./build_metal.sh --build-type RelWithDebInfo
```

### 2. (2-galaxy only) Inter-galaxy ETH PHYs trained

A fresh galaxy reset via `tt-smi -glx_reset_auto` alone is **not** enough on
this hardware — the inter-galaxy ethernet PHYs need to be trained by either a
simultaneous-via-mpirun reset across all hosts and/or a traffic-sending
cluster-validation pass.

You can do this manually:

```bash
cd .../tt-metal
./tools/scaleout/exabox/recover.sh --hosts <HOST_A>,<HOST_B> --num-iterations 10
# Repeat 1-2 times if first run reports failures.
```

…or use the wrapper script:

```bash
./models/demos/deepseek_v3_d_p/tests/pipeline/run_2galaxy_end_to_end.sh \
    --host-a bh-glx-d04u02 --host-b bh-glx-d04u08
```

`run_cluster_validation` (which `recover.sh` invokes) must be built. If it's
missing under `build/tools/scaleout/`, rebuild tt-metal with the scaleout
tools target.

### 3. `TT_TCP_INTERFACE`

The NIC name carrying inter-host traffic on your cluster. On the d04 hosts
it's `ens5f0np0`. Find via:

```bash
ip -br addr | grep UP   # pick the one with a cluster-fabric IP
```


## Running


### Single galaxy

```bash
cd /path/to/tt-blaze
source env.sh
bash tt-metal/models/demos/deepseek_v3_d_p/tests/pipeline/runme_cross_mesh_smoke.sh
```

`tt-run --bare --mpi-args "--oversubscribe"` packs both ranks onto the
single allocated host. Rank 0 (mesh_id=0) gets tray 1 (chips 0–7); rank 1
(mesh_id=1) gets tray 2 (chips 8–15). MGD is the in-tree
`bh_galaxy_dual_2x4_intermesh.textproto` (4 intermesh channels with
`assign_z_direction: true`, RELAXED policy).

Smoke verifies 4 H2D→D2D→D2H iterations of a 64-byte `int32` tensor with
bit-exact roundtrip. Perf sweep transfers a 640×1792 bf16 tensor 20× per
page-size point, reports wall time + GB/s.


### Two galaxies

Easiest path is the wrapper:

```bash
cd /path/to/tt-blaze
source env.sh
bash tt-metal/models/demos/deepseek_v3_d_p/tests/pipeline/run_2galaxy_end_to_end.sh \
    --host-a bh-glx-d04u02 --host-b bh-glx-d04u08
```

Or run the two stages manually:

```bash
# 1. Train inter-galaxy ETH (repeat if needed)
cd .../tt-metal
./tools/scaleout/exabox/recover.sh --hosts bh-glx-d04u02,bh-glx-d04u08 --num-iterations 10

# 2. Run the test (do NOT reset between training and this)
cd /path/to/tt-blaze
HOST_A=bh-glx-d04u02 HOST_B=bh-glx-d04u08 TT_TCP_INTERFACE=ens5f0np0 \
    bash tt-metal/models/demos/deepseek_v3_d_p/tests/pipeline/runme_2galaxy_cross_mesh_smoke.sh
```

Both hosts must have:
- `/data/jjovicic/...` on shared NFS (verified with `df -T`)
- Passwordless SSH between them (`ssh-copy-id` once, both directions)
- HAL-bumped tt-metal build (see prereq 1)

The MGD is `dual_bh_galaxy_experimental_mesh_graph_descriptor.textproto`
(2 `(4, 8)` meshes, 4 intermesh channels STRICT, `assign_z_direction: true`).


## Expected results

### Topology + setup

```
Inter-mesh mapping succeeded, found 2 mesh pair(s)
[mesh_id=0] iter 0: write_tensor
[mesh_id=1] iter 0: roundtrip OK
...
[rank with mesh_id={0,1}] smoke OK
```

### Perf (640×1792 bf16 = 2,293,760 bytes per logical tensor; 20 iters/point)

Both single-galaxy and 2-galaxy land at the same numbers (±1%):

```
page_size  | per-tensor  | bandwidth
-----------+-------------+--------------
   256 B   | 10.7 ms     | 0.22 GB/s
  1024 B   |  8.0 ms     | 0.29 GB/s
  2048 B   |  7.8 ms     | 0.29 GB/s
  4096 B   |  7.7 ms     | 0.30 GB/s
  8192 B   |  7.6 ms     | 0.30 GB/s   ← plateau
 16384 B   |  7.6 ms     | 0.30 GB/s
```

The 0.30 GB/s plateau is the **single H2D writer / single D2H reader** ceiling
for this socket path. Inter-galaxy DAC is fast enough that the bottleneck is
elsewhere (per-page setup + single-channel pipeline depth, not the cable).


## Troubleshooting


### `TT_FATAL: Program size (X) too large for kernel config buffer (25600) on ACTIVE_ETH`

You're on a build without the HAL bump (see prereq 1). Rebuild from
`jjovicic/socket-experiment`.

### `Inter-mesh mapping succeeded` then `Device N: Timeout (10000 ms) waiting for physical cores ... `

Inter-galaxy ETH PHYs aren't trained. Run `recover.sh` (with validation, not
`--skip-validation`) — see prereq 2. If one run isn't enough, run 2–3 times
back-to-back with `--num-iterations 10`. Don't re-`tt-smi -glx_reset_auto`
between recovery and the test.

### `ETH_LIVE_STATUS: 0x0` on every chip via `tt-smi -s`

Not a definitive "no link" signal — runtime fabric init is what trains the
link, and the telemetry is misleading. Try running the test anyway and see
whether device init succeeds.

### `Permission denied (publickey)` from mpirun

You need passwordless SSH between the two hosts.

```bash
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
# Repeat on the peer if ~/.ssh isn't on shared NFS.
```

### `All nodes which are allocated for this job are already filled.`

SLURM allocated 1 task; you're trying to run 2 ranks. Already handled in the
launchers via `--mpi-args "--oversubscribe"` (single-galaxy) or
`--mpi-args "--host ${HOST_A}:1,${HOST_B}:1 ..."` (2-galaxy). If you write
your own launcher, include `--oversubscribe`.


## Why each MGD knob

- **`assign_z_direction: true`** on the inter-mesh connection — instructs the
  topology mapper to use **Z-direction physical ethernet ports** (the
  inter-tray and inter-galaxy cables). Without this, the mapper would try
  X/Y direction ports which don't reach another mesh.
- **`channels { count: N policy: RELAXED|STRICT }`** — number of inter-mesh
  ethernet channels the mapping requires. RELAXED accepts fewer if the
  hardware doesn't provide all N (logged as a warning, not fatal); STRICT
  must match exactly. The single-galaxy MGD asks for 8 in RELAXED mode and
  uses the 4 the hardware actually has between trays.
- **`device_topology { dims: [...] }`** must match the per-rank mesh shape:
  `(2, 4)` for one tray, `(4, 8)` for one galaxy.


## What it builds toward

The cross-mesh socket path here is the same `SocketInterface` + cross-mesh
`MeshSocket` machinery the production sp4 decode pipeline uses for inter-stage
data. Confirming it works at the 2-rank level is the unit-test prerequisite
for KV-cache migration between a prefill galaxy and a decode galaxy (step 3
in the broader plan).

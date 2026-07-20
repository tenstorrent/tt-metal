# Running ttnn ops (add / matmul / CCL) across two T3Ks

A hands-on, reproducible guide. Everything here was run on the two cabled T3K boxes
(`t3k-node-b` .243 = launcher, `t3k-node-a` .247 = remote) and **passes** — see
`scripts/PASS_output.txt` for the exact captured output.

This guide assumes the hardware/software setup in
[`../Dual_T3K_test.md`](../Dual_T3K_test.md) is already done (identical checkout+build on
both boxes, `mpirun-ulfm`, passwordless SSH, `/etc/mpirun/hostfile`, `ens18`). That doc is
about the C++ fabric test suite; **this** doc runs real `ttnn` Python tensor ops.

---

## 0. The mental model: two ways to use two T3Ks

Each T3K = 8 Wormhole chips (a 2×4 mesh). Cabled together you have 16 chips over 2 hosts.
`ttnn` exposes them via **one MPI process per host**, all running the *same* Python script
under `tt-run` (a wrapper over `mpirun-ulfm`). There are two distributed models:

| | **Big-Mesh** (scale-up) | **Multi-Mesh + sockets** (scale-out) |
|---|---|---|
| Logical view | **one** 1×16 MeshDevice across both hosts | **two** 2×4 MeshDevices (one per host) |
| A single op | `ttnn.add(a,b)` runs SPMD across all 16 chips | runs on one host's 8 chips |
| Cross-host data | implicit (runtime shards/gathers over fabric) | explicit `MeshSocket` `send_async`/`recv_async` |
| Use it for | tensor/data parallelism, one big model | pipeline parallelism, multi-model |
| MGD | `dual_t3k_1x16_experimental_bigmesh_mgd.textproto` | `dual_t3k_mesh_graph_descriptor.textproto` |
| Script | `scripts/bigmesh_ops.py` | `scripts/multimesh_pipeline.py` |

Both are demonstrated below with **add + matmul + an all_gather (CCL)**, each checked
against a PyTorch golden by PCC.

---

## 1. Where the common files live (shared mount)

`tt-run` starts rank 1 on the **remote** host, so the Python file must exist at the **same
path on both boxes**. This whole `claude_job` folder is kept at:

```
/home/namvu/dual-t3k/tt-metal/claude_job/
```

on the launcher (`t3k-node-b` .243), and the remote (`t3k-node-a` .247) **NFS-mounts
`/home/namvu/dual-t3k/tt-metal` at the same path**, so both hosts see the scripts at
`/home/namvu/dual-t3k/tt-metal/claude_job/dual_t3k_ops/scripts/`. On the launcher, the old
repo path `…/tenstorrent/tt-metal/claude_job` is a symlink to this location, so
`claude_job/…` still resolves there too.

**Mount setup.** The NFS packages are already installed on both boxes
(`nfs-kernel-server` on the launcher, `nfs-common` on the remote), the server is `enabled`,
the export line is in `/etc/exports`, and the remote auto-mounts via a systemd-automount
`/etc/fstab` line — so **the mount now survives reboots on its own** (see "Persistent across
reboots" below). Steps 1–2 are the manual bring-up if you ever need to (re)create it from
scratch or the automount is missing.

**Step 1 — launcher (.243):** add the export line, then re-export.
```bash
echo '/home/namvu/dual-t3k/tt-metal  192.168.1.247(rw,sync,no_subtree_check,no_root_squash)' | sudo tee -a /etc/exports
sudo exportfs -ra
sudo exportfs -v          # should now list the line
```

**Step 2 — remote (t3k-node-a, .247):** mount it. The mountpoint dir already exists, so:
```bash
ssh namvu@t3k-node-a 'sudo mount -t nfs 192.168.1.243:/home/namvu/dual-t3k/tt-metal /home/namvu/dual-t3k/tt-metal'
```

**Step 3 — verify both hosts see the same files:**
```bash
ls /home/namvu/dual-t3k/tt-metal/claude_job/dual_t3k_ops/scripts/          # launcher
ssh namvu@t3k-node-a 'ls /home/namvu/dual-t3k/tt-metal/claude_job/dual_t3k_ops/scripts/'  # remote
```
Both should list `bigmesh_ops.py`, `multimesh_pipeline.py`, etc.

> **How to tell it's down:** the remote's `ls` returns `No such file or directory` and
> `findmnt /home/namvu/dual-t3k/tt-metal` on the remote prints nothing. That means the mount
> dropped (usually a reboot) — redo Steps 1–2.
>
> Because both hosts share ONE copy, editing a script needs no sync — just edit in place.
> **The whole tt-metal framework now lives on this mount too** (`/home/namvu/dual-t3k/tt-metal`,
> commit `3632044cfd6`, `build_Release` + `python_env` + MGDs) — physically on the launcher's
> local disk, read by the remote over NFS. So both ranks run one identical tree. See §2 for the
> env vars this requires. (The old per-host checkout under `/home/namvu/tenstorrent` is no longer
> used for these runs.)

**Persistent across reboots (already configured).** The server side survives reboots because
`nfs-server` is `enabled` and the export line lives in `/etc/exports` on the launcher. The remote
auto-mounts via an `/etc/fstab` line.

> **Don't use a plain `defaults,_netdev` fstab line** — it was tried and it did NOT survive a
> reboot. If the remote boots while the launcher's NFS export isn't ready yet (both boxes reboot,
> or .247 comes up first), the one-shot mount attempt fails and systemd never retries, so the
> folder shows up empty. Use a **systemd automount** line instead: it installs an `autofs`
> placeholder at boot and mounts NFS lazily on first access, so server-vs-client boot ordering no
> longer matters, and `nofail` keeps a slow/absent server from blocking boot.

The line now in the remote's `/etc/fstab`:
```
192.168.1.243:/home/namvu/dual-t3k/tt-metal  /home/namvu/dual-t3k/tt-metal  nfs  _netdev,x-systemd.automount,x-systemd.mount-timeout=30,nofail,retry=5,hard,timeo=600  0  0
```
To (re)install it and activate without a reboot:
```bash
ssh namvu@t3k-node-a 'grep -q "x-systemd.automount.*dual-t3k" /etc/fstab || \
  echo "192.168.1.243:/home/namvu/dual-t3k/tt-metal  /home/namvu/dual-t3k/tt-metal  nfs  _netdev,x-systemd.automount,x-systemd.mount-timeout=30,nofail,retry=5,hard,timeo=600  0  0" | sudo tee -a /etc/fstab'
ssh namvu@t3k-node-a 'sudo systemctl daemon-reload && sudo systemctl start "$(systemd-escape -p --suffix=automount /home/namvu/dual-t3k/tt-metal)"'
# verify: the mountpoint auto-mounts the moment you touch it
ssh namvu@t3k-node-a 'ls /home/namvu/dual-t3k/tt-metal/claude_job/dual_t3k_ops/scripts/ && findmnt /home/namvu/dual-t3k/tt-metal'
```
If it ever regresses again, `findmnt /home/namvu/dual-t3k/tt-metal` on the remote showing an
`autofs` row means the automount is armed; a second `nfs4` row appears after first access.

---

## 2. Every run: shell env + fabric preflight

> **⚠️ FIRST: these T3Ks are SHARED. Always hold `moreh-lock` around anything that touches the
> devices (reset, `test_system_health`, `tt-run`).** Other people/sessions (e.g. a `moreh-vllm`
> container, another Claude session) run dual-T3K jobs on the *same 16 chips*. Two jobs opening the
> same fabric at once wedges the ETH heartbeat and neither recovers — this caused every mysterious
> "Stuck at 0xabcd…" wedge we hit. `moreh-lock` serializes access via `/dev/shm/MOREH_LOCK.*`.
>
> ```bash
> moreh-lock status                         # "Lock is free" or who holds it
> # wrap the WHOLE hardware sequence (reset + health + tt-run) in ONE lock hold:
> moreh-lock run --wait-timeout 600 --max-hold 1800 -m "bigmesh verify" -- bash -lc '
>   <reset + test_system_health + tt-run ...>'
> ```
> `moreh-lock run` queues if the lock is held, ties the lock to the command lifetime, forwards
> signals, and frees it on exit. Do the reset **inside** the lock so nothing grabs the fabric
> between reset and run. (`moreh-lock -h` for options.)

Run from the **mounted framework** at `/home/namvu/dual-t3k/tt-metal` (commit `3632044cfd6`),
which both hosts see over NFS. Set **all four** env vars — the two root vars must BOTH point at
the mounted tree, and the cache must be LOCAL per host:

```bash
cd /home/namvu/dual-t3k/tt-metal
source python_env/bin/activate                              # puts tt-run on PATH
export ARCH_NAME=wormhole_b0                                # else: "Error: ARCH_NAME is not set"
export TT_METAL_HOME=/home/namvu/dual-t3k/tt-metal          # mounted tree, NOT ~/tenstorrent
export TT_METAL_RUNTIME_ROOT=/home/namvu/dual-t3k/tt-metal  # C++ kernel-build root (see below)
export TT_METAL_CACHE=/home/namvu/.cache/tt_metal_local     # LOCAL per-host kernel cache

# Verify the inter-host QSFP fabric is trained BEFORE launching:
./build_Release/test/tt_metal/tt_fabric/test_system_health
```

`test_system_health` must end with `[  PASSED  ] 3 tests.`. If it reports
`Timed out waiting for ETH heartbeat ... Stuck at 0xabcd0ff2`, the links are not trained —
see Troubleshooting.

**Why each var matters (all four were required to make the mounted framework work):**
- **`TT_METAL_RUNTIME_ROOT` (not just `TT_METAL_HOME`).** The C++ runtime derives its
  kernel-build root — sfpi compiler, framework `-I` includes, kernel sources — from
  `TT_METAL_RUNTIME_ROOT` (`rtoptions.cpp`), **not** `TT_METAL_HOME`. Both `~/.bashrc` files here
  still export the OLD repo (`TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT=/home/namvu/tenstorrent/tt-metal`).
  If you set only `TT_METAL_HOME`, the run loads the mounted `libtt_metal.so` (commit `3632044`) but
  compiles kernels from the old repo (commit `f50eb740`) → header/source skew →
  `idle_erisc compile failure` → **SIGSEGV on a remote rank**. Set BOTH to the mounted path.
- **`TT_METAL_CACHE` must be LOCAL.** With one shared `TT_METAL_HOME` on NFS, both hosts would
  otherwise JIT-compile the same kernels into the same shared `built/` dir at once → write race.
  Pointing it at `/home/namvu/.cache/tt_metal_local` (on each host's local `sda2`, NOT the mount)
  gives each host its own cache. `/tmp/ttnn` is already local.
- **tt-run forwards all of these automatically.** `tt-run` propagates every `TT_*`/`ARCH_*` var to
  BOTH ranks via `mpirun -x KEY=value` (`ttrun.py` `ENV_PASSTHROUGH_PREFIXES`), and explicitly
  overrides `TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT` with the launcher's values — so the remote's
  stale `.bashrc` does NOT win. Exporting them once on the launcher is enough. Because the cache
  path string is identical on both hosts but resolves to each host's own local disk, the local-cache
  trick needs no per-host difference.

> **One-time build fix (already applied): `ttnn/ttnn/_ttnn.so` symlink.** A fresh mounted checkout
> may be missing the Python-package copy of the native extension, so `import ttnn` dies with
> `ModuleNotFoundError: No module named 'ttnn._ttnn'`. The `.so` exists under `build_Release/ttnn/`;
> link it into the package dir (its RUNPATH is absolute, so a symlink resolves its deps fine, and
> both hosts see it over NFS):
> ```bash
> ln -sf ../../build_Release/ttnn/_ttnn.so /home/namvu/dual-t3k/tt-metal/ttnn/ttnn/_ttnn.so
> ```

---

## 3. Big-Mesh: add + matmul + all_gather across all 16 chips

```bash
tt-run --tcp-interface ens18 \
  --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_1x16_experimental_bigmesh_mgd.textproto \
  --hosts t3k-node-a,t3k-node-b \
  python3 /home/namvu/dual-t3k/tt-metal/claude_job/dual_t3k_ops/scripts/bigmesh_ops.py
```

Expected (rank prefixes `[1,0]`/`[1,1]` are the two hosts):

```
[rank 0] opened Big-Mesh (1, 16) = 16 chips across 2 host(s)
[rank 0] ADD: checked 8 local shard(s), worst PCC=1.00000 -> PASS
[rank 0] MATMUL: checked 8 local shard(s), worst PCC=0.99998 -> PASS
[rank 0] ALL_GATHER: checked 8 local device(s), worst PCC=1.00000 -> PASS
[rank 0] OVERALL: PASS
```

First launch runs a ~1–2 min Phase-1 discovery (`generate_rank_bindings`) and caches it under
`generated/ttrun/<id>/`; later launches reuse the cache (`Phase 1 cache hit`). Total wall time
~3–4 min including fabric bring-up on 16 chips.

---

## 4. Multi-Mesh + sockets: a host→host pipeline

```bash
tt-run --tcp-interface ens18 \
  --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto \
  --hosts t3k-node-a,t3k-node-b \
  python3 /home/namvu/dual-t3k/tt-metal/claude_job/dual_t3k_ops/scripts/multimesh_pipeline.py
```

Pipeline: rank 0 does `C = add(a,b)` on mesh 0 and `send_async`s it over a `MeshSocket`;
rank 1 `recv_async`s C, does `matmul(C,W)`, and verifies. Expected:

```
[rank 0] Multi-Mesh: each host owns a (2, 4) mesh; 2 meshes total
[rank 0] ADD done on mesh 0; C sent -> mesh 1
[rank 1] recv C, MATMUL(C,W) done; 8 local shard(s), worst PCC=0.99998 -> PASS
[rank 1] intra-mesh ALL_GATHER worst PCC=1.00000 -> PASS
[rank 1] Multi-Mesh pipeline OVERALL: PASS
```

---

## 5. Cautions & gotchas (read before editing the scripts)

- **`FABRIC_2D`, not `FABRIC_1D`.** The 1×16 big mesh can't route under `FABRIC_1D`
  (`Could not find any forwarding direction ...`). Set fabric **before** `open_mesh_device`;
  disable it on teardown.
- **Never hard-kill a `tt-run` job mid-init.** `pkill`/SIGKILL **or a wrapper's SIGTERM at a fixed
  timeout** during fabric init wedges the ethernet cores and the next run fails the fabric
  handshake. A first cold run recompiles all kernels on 16 chips and can exceed 10 min, so launch
  it in the **background** (not under a foreground timeout) and poll the log — do not let a harness
  timeout SIGTERM it. Recover a wedge per §6 (concurrent `tt-smi -r` + settle).
- **SPMD = every rank runs the whole script.** `torch.manual_seed(0)` **before** building inputs
  (so both ranks' goldens match); branch on `ttnn.distributed_context_get_rank()`; only rank 0/1
  should print/assert. Call `torch.set_num_threads(os.cpu_count())` after opening the mesh —
  `MPI_Init` pins torch to 1 thread.
- **Verify per LOCAL shard, not a global concat.** Each rank only owns its half of the mesh, so
  `ttnn.to_torch(mesh_composer=Concat...)` does not gather across hosts. Use
  `ttnn.get_device_tensors(t)` filtered by `mesh_device.get_view().is_local(coord)` and compare
  each local shard to its golden slice. (Both scripts have a `local_shards()` helper.)
- **all_gather axis must match the shard axis.** To gather back to the full tensor, shard with
  `ttnn.ShardTensor2dMesh(dims=(None, d), mesh_shape)` on the same axis you pass as
  `cluster_axis`, and use `topology=ttnn.Topology.Linear`.
- **Socket recv spec must match the sent tensor** exactly: receiver does
  `ttnn.allocate_tensor_on_device(sender_tensor.spec, device)`.
- **Benign noise to ignore:** `Unknown motherboard '' ... falling back to bus_id`, and
  `Failed to create shared memory /tt_device_..._memory: Permission denied`.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| rank 1 `can't open file '.../script.py'` | remote NFS mount of `/home/namvu/dual-t3k/tt-metal` not up / path mismatch (§1) |
| `ModuleNotFoundError: No module named 'ttnn._ttnn'` | mounted checkout missing the package-dir native ext → `ln -sf ../../build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so` (§2 one-time fix) |
| `idle_erisc compile failure` then a rank **SIGSEGVs** during device init | `TT_METAL_RUNTIME_ROOT` still points at the old repo (`~/.bashrc`) while running the mounted lib → export `TT_METAL_RUNTIME_ROOT=/home/namvu/dual-t3k/tt-metal` (§2). Clear the poisoned cache: `rm -rf /home/namvu/.cache/tt_metal_local/*` on both hosts |
| `Error: ARCH_NAME is not set` | `export ARCH_NAME=wormhole_b0` (§2) |
| `tt-run: command not found` | `source python_env/bin/activate` (§2) |
| foreground run killed at a fixed timeout (SIGTERM) mid-init → wedges fabric | run the launch in the **background** so a wrapper timeout can't SIGTERM it mid-fabric-init; poll the log for `OVERALL`/`EXIT CODE` |
| fabric wedges repeatedly; `tt-smi -r` "succeeds" but health still `Stuck at 0xabcd…` even after settling | **another session is using the same 16 chips** (concurrent dual-T3K job). Check: `moreh-lock status`, and on both hosts `pgrep -af "run_bigmesh_smoke\|ttrun\|tt-smi"` / `docker ps` (look for `moreh-vllm*`). Fix: run everything under **`moreh-lock`** (§2); reset+run inside one lock hold so the peer can't re-grab the fabric between reset and launch |
| `Could not find any forwarding direction from src (M0,D3) to dst (M0,D0)` | using `FABRIC_1D` on the big mesh → use `FABRIC_2D` |
| `open_mesh_device` → `Fabric Router Sync: Timeout ... Ethernet handshake likely failed` | ethernet cores wedged (usually after a killed run) → `tt-smi -r` on **both** hosts, then wait for `test_system_health` PASS |
| `test_system_health`: `Timed out waiting for ETH heartbeat ... Stuck at 0xabcd0ff2` | QSFP links not trained (e.g. just after a reset). Wait ~1 min and re-run; it auto-retrains |
| Hang on socket `send_async`/`recv_async` | wrong `sender_rank`/`receiver_rank`, or recv spec ≠ sent spec |
| `Phase 1 cache hit` but topology changed | add `--force-rediscovery` to re-run discovery |
| MPI wire/version errors | a stray `/usr/bin/mpirun` 4.1.2 on PATH; tt-run must use `mpirun-ulfm` 5.0.7 |

Recover a wedged cluster (both hosts) — **hold `moreh-lock`, then reset both CONCURRENTLY, then
wait ~90 s untouched**. The lock keeps a peer session from re-grabbing the fabric mid-recovery
(without it the reset won't hold — see Troubleshooting). Sequential resets also mistime the retrain
(one host trains against a still-resetting peer) and the cross-host ETH heartbeat stays `Stuck`
(`0xabcd...`); lock + concurrent reset + settle clears it on the first health check:
```bash
moreh-lock run --wait-timeout 600 --max-hold 1800 -m "recover+verify" -- bash -lc '
  ssh namvu@t3k-node-a "tt-smi -r" &         # remote, backgrounded
  tt-smi -r                                  # launcher, at the same time
  wait
  sleep 90                                   # settle while QSFP links retrain
  ./build_Release/test/tt_metal/tt_fabric/test_system_health   # expect PASSED 3 tests
  # ... then tt-run your workload here, still under the same lock ...
'
```

---

## 7. Writing your own op

Copy `scripts/bigmesh_ops.py` (Big-Mesh) or `scripts/multimesh_pipeline.py` (Multi-Mesh) and
change the middle. The skeleton is always:

```python
import os, torch, ttnn
torch.manual_seed(0)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)     # BEFORE open
dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 16))   # or (2,4) per host
torch.set_num_threads(max(1, os.cpu_count() or 1))
rank = int(ttnn.distributed_context_get_rank())

t = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                    mesh_mapper=ttnn.ShardTensorToMesh(dev, dim=3))   # or Replicate/Shard2d
out = ttnn.add(t, t)                                    # any ttnn op: matmul, all_gather, ...
ttnn.synchronize_device(dev)
# verify per local shard via ttnn.get_device_tensors(out) + get_view().is_local(coord)

ttnn.distributed_context_barrier()                      # BEFORE close
ttnn.close_mesh_device(dev)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
```

Launch with the matching MGD + `--hosts t3k-node-a,t3k-node-b` under `tt-run` (§3/§4).
Put new scripts under `/home/namvu/dual-t3k/tt-metal/claude_job/dual_t3k_ops/scripts/` so both
hosts see them via the shared mount (§1) — no copy step needed.

---

## 8. Inspect the hardware topology (Fabric Manager)

To *see* the physical 16-chip wiring (which chip connects to which, over which cables), use
**Fabric Manager** — a controller + per-host agents running in Docker. It's read-only and does
**not** need `moreh-lock` (it queries cached discovery, it doesn't train the fabric).

**Bring up all 16 chips in FM (both agents must be registered).** The controller + node-b agent
run on the launcher; register node-a's agent so FM sees the second host:
```bash
ssh namvu@t3k-node-a 'docker rm -f fabric-agent 2>/dev/null; docker run -d --name fabric-agent \
  --network host --privileged --restart unless-stopped \
  -v /dev/tenstorrent:/dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v ~/ttfm/cluster.yaml:/etc/fabric-manager/config.yaml:ro \
  ghcr.io/tenstorrent/tt-fabric-manager:latest \
  tt-fabric-manager-agent --config /etc/fabric-manager/config.yaml \
  --host-id t3k-node-a --advertise-address 192.168.1.247:50053'
ssh namvu@t3k-node-a 'docker logs --tail 5 fabric-agent'   # expect: "Registration successful"
```
`--restart unless-stopped` makes it survive reboots (as long as the controller is up).

**Query the topology** (`ttfm` helper):
```bash
ttfm(){ docker run --rm --net=host -e FABRIC_MANAGER_ENDPOINT=192.168.1.243:50051 --entrypoint="" \
  ghcr.io/tenstorrent/tt-fabric-manager:latest tt-fabric-manager-cli "$@"; }
ttfm show-topology              # ASCII summary -> "2 hosts | 16 ASICs | cross-host links"
ttfm query-cluster-descriptors  # per-host: chip index -> ASIC unique_id, ethernet_connections, n300 boards
ttfm query-physical-topology    # PSD JSON; exit_node_connection_table = the cross-host cables
```
Or the **web UI**: <http://192.168.1.243:8080> (served by the controller).

**What the real topology is** (from the live descriptors):
- Each host = one **T3K = 8 chips = 4× n300 boards**, wired as a **2×4 mesh** (each link = 2 eth
  channels). Per host the n300 board pairs are `{0,7} {1,5} {2,6} {3,4}`; the 4 MMIO/PCIe chips
  `0–3` form a ring `0-1-2-3-0`.
- The two hosts are bridged by **4 QSFP cross-host cables** (8 eth channels), one from each MMIO
  chip: `b0↔a0, b1↔a1, b2↔a2, b3↔a3` (2 channels each).
- `ttnn.open_mesh_device(MeshShape(1,16))` presents all this as a **logical 1×16 line** (not a
  ring/torus — `Topology.Linear`); `FABRIC_2D` routes over the real 2-D grid + the QSFP cables.

> **Gotcha:** if only one host's agent is registered, `show-topology` reports `1 host | 8 ASICs |
> 0 cross-host links` — it can't confirm a cross-host link one-sided (the raw descriptor still
> lists it). Register **both** agents for the full `2 hosts | 16 ASICs`.
>
> A rendered diagram of this topology (real ASIC ids + cables) lives at
> `claude_job/dual_t3k_ops/topology.html`.

### 8b. Web UI graph shows only 4 chips / "1 tray, 2 devices" — configure an FSD

**Symptom.** `ttfm query-physical-topology` and the CLI correctly report **16 ASICs / 2
hosts**, but the web UI graph (default **Source = PSD (discovered)**) collapses each T3K to
**one tray with 2 devices → 4 nodes, ~3 links**.

**Why.** The UI draws one node per *physical slot* = `(hostName, trayId, asicLocation)`
(`web/app.js` `slotKeyFor`). A T3K (LoudBox = 4× n300 PCIe cards) has no chassis "tray", so
UMD discovery reports **`tray_id = 0` for all 4 boards**, and `asic_location` is only 0/1 (the
two chips of one n300). So all 8 chips per host map to just **2 slots**. The per-board identity
(tray 0..3) is only carried by a **Factory System Descriptor (FSD)**, which was not configured
(controller logged `FSD aggregation unavailable … falling back to the PSD-only host set`; the
UI's **FSD (factory)** source returned `Factory system descriptor not configured`). This is a
UI limitation for tray-less systems, **not** a discovery/hardware fault — the data is all there.

**Fix (done, reboot-persistent). Everything lives in THIS repo — no need to touch the
`~/tt-fabric-manager` tree.** All the FM controller needs (its config + the FSD) is kept under
`claude_job/dual_t3k_ops/fabric_manager/` and mounted straight into the container:

```
claude_job/dual_t3k_ops/
├─ fabric_manager/
│  ├─ controller.yaml                 # FM controller config (has the FSD search path)
│  ├─ fsd/dual_t3k_fsd.textproto      # the generated FSD for THIS dual-T3K
│  └─ apply.sh                        # recreate the controller to read the two above
└─ scripts/gen_fsd.py                 # regenerates the FSD from a live PSD dump
```

The FM *agents* keep using their own `~/tt-fabric-manager/configs/cluster.yaml` (they only read
`controller.listen_address`); that file is left untouched. Only the **controller** is pointed at
this repo.

**To (re)apply — one command on the launcher (t3k-node-b / .243):**
```bash
bash claude_job/dual_t3k_ops/fabric_manager/apply.sh
```
`apply.sh` recreates the `fabric-controller` container mounting `controller.yaml` →
`/etc/fabric-manager/config.yaml` and `fsd/` → `/etc/fabric-manager/fsd`, waits for both agents
to re-register, and prints the ASIC count (expect 16). It's **read-only / observability only** —
does **not** touch the fabric, so no `moreh-lock` needed. `--restart unless-stopped` makes it
survive reboots.

**Then view it:** open <http://192.168.1.243:8080>, set the **Source** dropdown to **FSD
(factory)** → 16 devices across 4 trays/host, all links. Tick **Validate** to diff the
discovered (PSD) topology against this factory (FSD) view.

**To regenerate the FSD** (only needed if the hardware/cabling changes — see the ⚠️ note below):
```bash
ttfm query-physical-topology > psd_dualt3k.json
python3 claude_job/dual_t3k_ops/scripts/gen_fsd.py psd_dualt3k.json \
  claude_job/dual_t3k_ops/fabric_manager/fsd/dual_t3k_fsd.textproto
# added/edited *.textproto is re-read per query — no restart needed unless you edit controller.yaml
```
The generator assigns each n300 board (found via its loc0↔loc1 internal link) a distinct
`tray_id` 1..4 and faithfully reproduces every eth channel + the 4 QSFP cross-host cables.

Verify from the CLI (PSD stays collapsed by design; FSD is the full view):
```bash
curl -s 'http://192.168.1.243:8080/api/physical-topology?source=fsd' \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print(len(d["asicDescriptors"]),"ASICs")'
# -> 16 ASICs
```

> **Notes.** The default **PSD** graph *still* shows 4 collapsed nodes — that's inherent to
> tray-less T3K discovery; use the **FSD** source for the real 16-chip picture. The FSD's
> synthesized ASIC ids only remap to real hardware `unique_id`s when the discovered slot
> `(host, tray_id, asic_location)` matches — since discovery reports `tray_id=0`, the FSD nodes
> keep synthetic ids and node tooltips omit the PCIe BDF (cosmetic only; connectivity is exact).

> **⚠️ This FSD is specific to THIS dual-T3K cluster — regenerate it for any other topology.**
> `dual_t3k_fsd.textproto` hard-codes exactly these two hosts (`t3k-node-a`, `t3k-node-b`), 4
> n300 trays each, and the exact cross-host cabling — it is **not** a generic descriptor. If you
> run a different system (a single T3K, 4 T3Ks, more/other hosts, non-n300 boards, or the cables
> get re-plugged), this file no longer matches and the **FSD (factory)** view will be wrong or
> the controller will report `No FSD covers all hosts`. For any other layout, **re-run the
> generator against that system's live PSD** and drop the new `.textproto` into
> `claude_job/dual_t3k_ops/fabric_manager/fsd/` (the controller loads *every* `*.textproto` in
> that dir and auto-selects the one whose `hosts` match the discovered set, so you can keep
> several side by side):
> ```bash
> ttfm query-physical-topology > psd_<name>.json
> python3 claude_job/dual_t3k_ops/scripts/gen_fsd.py psd_<name>.json \
>   claude_job/dual_t3k_ops/fabric_manager/fsd/<name>_fsd.textproto
> # no controller restart needed for an added file — it's re-read per query;
> # (restart only if you edit controller.yaml)
> ```
> The generator itself is topology-agnostic *for n300-based systems*: it reads hosts from
> `host_to_rank`, finds each n300 board via its loc0↔loc1 internal link, and reproduces every
> discovered eth channel + cross-host cable. It asserts on non-n300 layouts (4 loc0 chips/host,
> one loc1 partner per board) — for Galaxy/Blackhole/other boards, adjust the pairing logic in
> `gen_fsd.py` or author the FSD by hand (schema:
> `tt-fabric-manager/.../factory_system_descriptor/schemas/factory_system_descriptor.proto`).

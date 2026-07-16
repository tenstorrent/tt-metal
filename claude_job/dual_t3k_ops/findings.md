# Findings — ttnn ops across two T3Ks (keep-forever)

Verified working on 2026-07-15 from the **mounted framework** `/home/namvu/dual-t3k/tt-metal`,
commit `3632044cfd6`, both boxes (Big-Mesh + Multi-Mesh, exit 0). First verified 2026-07-14 on the
old local repo `f50eb740b26`; re-verified here from the NFS-mounted tree both ranks share.

## The two working launch commands

Preamble (launcher = `t3k-node-b`), run from the **mounted framework** so both hosts share one tree:
```bash
cd /home/namvu/dual-t3k/tt-metal                            # mounted tree (commit 3632044cfd6)
source python_env/bin/activate
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=/home/namvu/dual-t3k/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/namvu/dual-t3k/tt-metal  # MUST also be mounted (see gotcha #10)
export TT_METAL_CACHE=/home/namvu/.cache/tt_metal_local     # LOCAL per host (see gotcha #11)
```

**Big-Mesh** — one logical 1×16 MeshDevice across both hosts, ops SPMD over all 16 chips:
```bash
tt-run --tcp-interface ens18 \
  --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_1x16_experimental_bigmesh_mgd.textproto \
  --hosts t3k-node-a,t3k-node-b \
  python3 /home/namvu/dual-t3k/tt-metal/claude_job/dual_t3k_ops/scripts/bigmesh_ops.py
```

**Multi-Mesh + sockets** — each host = its own 2×4 mesh, tensors cross hosts via MeshSocket:
```bash
tt-run --tcp-interface ens18 \
  --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto \
  --hosts t3k-node-a,t3k-node-b \
  python3 /home/namvu/dual-t3k/tt-metal/claude_job/dual_t3k_ops/scripts/multimesh_pipeline.py
```

Both exit 0 with rank PCC PASS (add=1.0, matmul=0.99998, all_gather=1.0). See `scripts/PASS_output.txt`.

## Gotchas (each cost real debugging time)

1. **Script must live at the SAME path on both hosts.** rank 1 runs on the remote. Scripts live at
   `/home/namvu/dual-t3k/tt-metal/claude_job/dual_t3k_ops/scripts/` on the launcher; the remote
   NFS-mounts `/home/namvu/dual-t3k/tt-metal` at the same path (see GUIDE §1), so one copy serves both
   hosts — no rsync, edit in place. (Fallback if no shared mount: `rsync -az script.py
   namvu@t3k-node-a:<same-path>` on every edit.) The repo checkout + MGDs remain identical per-host
   copies under `/home/namvu/tenstorrent`.
2. **Big-Mesh needs `FABRIC_2D`, not `FABRIC_1D`.** `FABRIC_1D` can't build a route on the 1×16 mesh
   spanning two hosts → `TT_FATAL: Could not find any forwarding direction from src (M0,D3) to dst (M0,D0)`.
3. **Never hard-kill (`pkill`/Ctrl-C SIGKILL) a `tt-run` job mid-fabric-init.** It wedges the ethernet
   cores; the next `open_mesh_device` dies with `Fabric Router Sync: Timeout ... Ethernet handshake ...`.
   Let jobs finish or hit their timeout cleanly.
4. **Recovering from a wedge:** `tt-smi -r` on **both** hosts, then re-run `test_system_health` until it
   PASSES. After a reset the inter-host QSFP links re-train in ~1 min; during that window health shows a
   transient `Timed out waiting for ETH heartbeat ... Stuck at 0xabcd0ff2`. Wait it out; do not launch
   until health is green.
5. **SPMD seed + rank-0 reporting.** Every rank runs the whole script. `torch.manual_seed(0)` BEFORE
   building inputs so goldens match; guard prints/asserts on rank. `torch.set_num_threads(os.cpu_count())`
   after init (MPI_Init pins it to 1).
6. **Multi-host verification is per-LOCAL-shard.** Each rank only physically owns its half of the mesh, so
   `ttnn.to_torch(..., mesh_composer=Concat...)` does NOT gather across hosts. Verify with
   `ttnn.get_device_tensors(t)` filtered by `mesh_device.get_view().is_local(coord)`, comparing each local
   shard to the matching golden slice.
7. **CCL sharding axis must match the gather axis.** 1-D `ShardTensorToMesh(dim)` shards across all flat
   devices; `all_gather(cluster_axis=k)` gathers only mesh axis k. To all_gather back to the full tensor,
   shard with 2-D `ShardTensor2dMesh(dims=(None, d), mesh_shape)` on the axis you gather, and use
   `topology=ttnn.Topology.Linear`.
8. **Socket recv spec must match the sent tensor exactly.** Receiver does
   `ttnn.allocate_tensor_on_device(sender_spec, device)`; identical seed on both ranks makes the sent and
   allocated specs match (shape/dtype/layout/sharding).
9. **`set_fabric_config(...FABRIC_2D)` BEFORE `open_mesh_device`**; `distributed_context_barrier()` BEFORE
   `close_mesh_device`; `set_fabric_config(...DISABLED)` on teardown.
10. **Running from a shared/mounted framework needs `TT_METAL_RUNTIME_ROOT`, not just `TT_METAL_HOME`.**
    The C++ runtime picks its kernel-build root (sfpi compiler, framework `-I` includes, kernel sources)
    from `TT_METAL_RUNTIME_ROOT` (`tt_metal/llrt/rtoptions.cpp`), **not** `TT_METAL_HOME`. Both boxes'
    `~/.bashrc` here still export the OLD repo path for both vars. Setting only `TT_METAL_HOME=<mounted>`
    loads the mounted `libtt_metal.so` (commit 3632044) but compiles kernels from the old repo
    (f50eb740) → version skew → `idle_erisc compile failure` → **SIGSEGV on a rank** during device init.
    Export BOTH root vars to the mounted path. `tt-run` forwards every `TT_*`/`ARCH_*` var to both ranks
    via `mpirun -x KEY=value` and overrides the two root vars with the launcher's values, so setting them
    once on the launcher wins over the remote's stale `.bashrc`. After such a crash, clear the poisoned
    cache (`rm -rf /home/namvu/.cache/tt_metal_local/*` on both hosts).
11. **`TT_METAL_CACHE` must be a LOCAL per-host dir** (`/home/namvu/.cache/tt_metal_local`, on each host's
    own `sda2` — NOT the NFS mount). With one shared `TT_METAL_HOME` on NFS, both hosts would JIT-compile
    the same kernels into the same shared `built/` dir concurrently → write race. The same path string on
    both hosts resolves to different local disks, so no per-host difference is needed. `/tmp/ttnn` is local.
12. **A mounted checkout may lack the package-dir native ext** → `import ttnn` fails with
    `ModuleNotFoundError: No module named 'ttnn._ttnn'`. One-time fix (both hosts see it over NFS):
    `ln -sf ../../build_Release/ttnn/_ttnn.so /home/namvu/dual-t3k/tt-metal/ttnn/ttnn/_ttnn.so`. The `.so`
    RUNPATH is absolute, so a symlink resolves its `_ttnncpp.so`/`libtt_metal.so` deps fine.
13. **Don't run the launch under a foreground wrapper timeout; use a background run + log poll.** A first
    cold run recompiles all kernels on 16 chips and can exceed 10 min. A wrapper SIGTERM at timeout kills
    it mid-fabric-init and wedges the ethernet exactly like `pkill` (gotcha #3). To recover a wedge, reset
    **both hosts concurrently** (`ssh remote 'tt-smi -r' & ; tt-smi -r ; wait`), then wait ~90 s untouched
    before polling `test_system_health` — sequential resets mistime the QSFP retrain and stay `Stuck`.

## Benign noise (ignore)

- `warning ... Unknown motherboard '' for chip_id=N ... falling back to bus_id as tray_id` — cosmetic.
- `warning ... Failed to create shared memory /tt_device_..._memory: Permission denied` — cosmetic.
- tt-run "Phase 1 cache hit, skipping generate_rank_bindings (generated/ttrun/<id>)" — expected reuse;
  add `--force-rediscovery` only if the topology changed.

## Environment (verified)

- Launcher `t3k-node-b` .243 / remote `t3k-node-a` .247, NIC `ens18` both.
- `mpirun-ulfm` = OpenMPI 5.0.7 ULFM (`/usr/local/bin`); tt-run uses it internally. Never `/usr/bin/mpirun` 4.1.2.
- `/etc/mpirun/hostfile` lists both hosts (used by the C++ suite; tt-run uses `--hosts`).
- Passwordless SSH launcher → remote. Identical checkout + `build → build_Release` on both.

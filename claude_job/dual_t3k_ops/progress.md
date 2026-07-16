# Progress journal

## 2026-07-14 — Session 1: setup + preflight

- Created job folder `claude_job/dual_t3k_ops/` (prompt/progress/findings + scripts/).
- **Preflight PASS** on launcher `t3k-node-b`:
  - `tt-run` on PATH via `python_env/bin/activate`; `ARCH_NAME=wormhole_b0`.
  - `/etc/mpirun/hostfile` lists `t3k-node-a slots=1`, `t3k-node-b slots=1`.
  - Passwordless SSH launcher → `t3k-node-a` works.
  - Remote parity: same commit `f50eb740b26`, `build → build_Release`, binaries + `mpirun-ulfm` present.
  - QSFP fabric link UP: cross-host pairs chip0 ch0/1, chip3 ch0/1, chip1 ch6/7, chip2 ch6/7 all
    `link UP (QSFP)` to remote Unique IDs (`2618320xx` = t3k-node-a). (`test_system_health`.)

## 2026-07-14 — Session 1: Big-Mesh working

- **File sync**: `/shared` is a common NFS mount on both hosts at the same path (verified: file
  written on .243 readable on .247). Staged scripts to `/shared/dual_t3k_ops/scripts/` and launch
  from there → no per-edit rsync. (rsync-to-remote is the fallback.)
- **Gotcha A**: script must exist at same path on both hosts (rank 1 = remote). `/shared` solves it.
- **Gotcha B**: `FABRIC_1D` on the 1×16 big mesh fails at fabric-route build:
  `Could not find any forwarding direction from src (M0,D3) to dst (M0,D0)`. **Use `FABRIC_2D`.**
- **Gotcha C**: hard-killing (`pkill`) a `tt-run` job **mid-fabric-init wedges the ethernet cores**;
  next run fails `open_mesh_device` with `Fabric Router Sync: Timeout ... Ethernet handshake ...`.
  Recovery = `tt-smi -r` on **both** hosts, then wait for `test_system_health` to PASS (ETH links
  re-train ~1 min after reset; they show a transient `ETH heartbeat ... Stuck at 0xabcd0ff2` first).
- **Big-Mesh RESULT (run4, FABRIC_2D, exit 0)** — `python3 /shared/.../bigmesh_ops.py` under tt-run
  with `dual_t3k_1x16_experimental_bigmesh_mgd.textproto`, `--hosts t3k-node-a,t3k-node-b`:
  - ADD worst PCC=1.00000, MATMUL worst PCC=0.99998, ALL_GATHER worst PCC=1.00000 → **OVERALL PASS**
  - Each rank verifies its 8 LOCAL shards (`get_device_tensors` + `view.is_local`), not a global concat.

- **Multi-Mesh RESULT (run1, FABRIC_2D, exit 0)** — `python3 /shared/.../multimesh_pipeline.py` under
  tt-run with `dual_t3k_mesh_graph_descriptor.textproto`, `--hosts t3k-node-a,t3k-node-b`:
  - rank 0: ADD on mesh 0 → `send_async` over MeshSocket → rank 1: `recv_async` → MATMUL PCC=0.99998;
    intra-mesh ALL_GATHER PCC=1.00000 → **OVERALL PASS**.
  - Gotcha D: 1-D `ShardTensorToMesh(dim)` shards across all 8 flat chips; `all_gather(cluster_axis=1)`
    only gathers one mesh axis. For an all_gather that recomposes the full tensor, shard with the
    **2-D** `ShardTensor2dMesh(dims=(None, d), mesh_shape)` on the same axis you gather.
  - Gotcha E: socket receiver must `allocate_tensor_on_device(sender_spec, device)` with a spec that
    matches the sender's sent tensor exactly (same shape/dtype/layout/sharding) → both ranks build
    identical goldens under one `torch.manual_seed(0)`.

Evidence: `scripts/PASS_output.txt`; full logs `scripts/_bigmesh_run4.log`, `scripts/_multimesh_run1.log`.

**STATUS: both models PASS. Wrote `GUIDE.md` + `findings.md`.**

## 2026-07-15 — Session 2: relocate to dedicated shared mount

- Stopped using `/shared` (NAS NFS). Moved the whole `claude_job/` to
  **`/home/namvu/dual-t3k/tt-metal/claude_job/`** on the launcher (.243). The remote (.247) will
  NFS-mount `/home/namvu/dual-t3k/tt-metal` at the **same path**, so both hosts share one copy —
  scripts run from `/home/namvu/dual-t3k/tt-metal/claude_job/dual_t3k_ops/scripts/`.
- Left a symlink `…/tenstorrent/tt-metal/claude_job -> /home/namvu/dual-t3k/tt-metal/claude_job` on
  the launcher so repo-relative `claude_job/…` still resolves. Removed `/shared/dual_t3k_ops`.
- Updated `GUIDE.md` §1 (mount setup + verify) and all launch paths; updated `findings.md` gotcha #1.
- Pending: user sets up the NFS export/mount on .247; then re-verify both hosts `ls` the same script
  path before the next distributed run. (Did NOT move the tt-metal repo/build — stays local per host.)

## 2026-07-15 — Session 2b: NFS mount UP + framework moved onto the shared mount

Situation report (state verified on disk):

- **NFS mount is LIVE.** Launcher (.243) exports `/home/namvu/dual-t3k/tt-metal` to .247 (`exportfs -v`
  shows the line); remote (.247) has it mounted (`findmnt` = nfs4, `clientaddr=192.168.1.247`). Both
  hosts see the same tree at the same path.
- **The FRAMEWORK now lives on the shared mount** (this is the "framework on NFS" / GLM5-style solution
  the user asked about, now chosen). `/home/namvu/dual-t3k/tt-metal` is a **full tt-metal checkout**:
  commit **`3632044cfd6`** (note: newer/different from the earlier local `f50eb740b26`), `build ->
  build_Release`, `python_env/bin/tt-run` present, test binaries present. The **remote sees the identical
  tree+binaries via the mount** (same commit, same `build` symlink, `test_system_health` present).
  Physically it's on the launcher's local `sda2`; .247 reads it over NFS. `claude_job/` lives inside it.
- **Discussion resolved:** mounting the framework is valid (user's GLM5-on-4-galaxy rig mounts the same
  `$HOME` on all hosts). My earlier "keep it local" was over-cautious. See findings + GUIDE.

### OPEN / MUST DO NEXT (important)
1. **Set `TT_METAL_CACHE` to a LOCAL per-host dir before any multi-host run.** It is currently **unset**.
   With one shared `TT_METAL_HOME` on NFS, both hosts JIT-compile the *same* kernels into the *same*
   `built/` dir at once → write race / corruption. Point each host's kernel cache at local disk:
   `export TT_METAL_CACHE=/home/namvu/.cache/tt_metal_local` (env var confirmed in
   `tt_metal/llrt/rtoptions.cpp`). Alternatively warm the cache single-host first. `/tmp/ttnn` is already local.
2. **Re-verify both models from the MOUNTED framework.** The earlier PASS was from the OLD local repo
   (`~/tenstorrent/tt-metal`, `f50eb740b26`). Not yet re-run from `~/dual-t3k/tt-metal` (`3632044cfd6`).
   Run Big-Mesh + Multi-Mesh with `TT_METAL_HOME=/home/namvu/dual-t3k/tt-metal` and `TT_METAL_CACHE` local;
   confirm rank-0 PASS + exit 0.
3. Then update `GUIDE.md` (§2 env: use the mounted `TT_METAL_HOME` + `TT_METAL_CACHE` local) and
   `findings.md` to make the mounted-framework the documented path.

### Still-true gotchas from Session 1 (unchanged)
FABRIC_2D for big mesh; never pkill mid-init (wedges ETH → `tt-smi -r` both hosts, wait for
`test_system_health` PASS); SPMD seed + rank-0 asserts; verify per-local-shard; all_gather axis must
match shard axis; socket recv spec must match sent spec.

## 2026-07-15 — Session 2b (cont.): BOTH models re-verified from the MOUNTED framework — PASS

Goal met: both models now run and PASS from `/home/namvu/dual-t3k/tt-metal` (commit `3632044cfd6`),
the tree both hosts share over NFS — not the old local `~/tenstorrent` (`f50eb740b26`).

**Result (both exit 0), evidence `scripts/PASS_output.txt` + `_bigmesh_mounted.log` / `_multimesh_mounted.log`:**
- Big-Mesh (1×16): ADD PCC=1.00000, MATMUL=0.99998, ALL_GATHER=1.00000 → OVERALL PASS (both ranks).
- Multi-Mesh (2×(2,4) + MeshSocket): rank0 ADD→send C; rank1 recv→MATMUL=0.99998, ALL_GATHER=1.00000 → PASS.

**Env now required (documented in GUIDE §2 + findings preamble):**
```
cd /home/namvu/dual-t3k/tt-metal && source python_env/bin/activate
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=/home/namvu/dual-t3k/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/namvu/dual-t3k/tt-metal   # both root vars, mounted
export TT_METAL_CACHE=/home/namvu/.cache/tt_metal_local      # LOCAL per host
```

**Three blockers hit and fixed this session (all new gotchas #10–13 in findings.md):**
1. `import ttnn` → `ModuleNotFoundError: ttnn._ttnn`. The mounted checkout lacked the package-dir
   native ext. Fixed once: `ln -sf ../../build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so` (absolute
   RUNPATH → symlink resolves deps; both hosts see it via NFS). Verified `import ttnn` on both hosts.
2. **`idle_erisc compile failure` → SIGSEGV on rank 1.** Root cause: C++ kernel-build root comes from
   `TT_METAL_RUNTIME_ROOT` (`rtoptions.cpp`), NOT `TT_METAL_HOME`. I'd set only HOME=mounted, so
   RUNTIME_ROOT stayed = old repo (from `~/.bashrc` line 133) → mounted lib (3632044) compiled old-repo
   kernels (f50eb740) → skew → segfault. Fix: also export `TT_METAL_RUNTIME_ROOT=mounted`. Confirmed
   tt-run forwards both to ranks via `-x KEY=value` (`ttrun.py`), overriding the remote `.bashrc`.
3. **Foreground 10-min wrapper timeout SIGTERM'd the job mid-fabric-init → wedged ETH** (heartbeat
   `Stuck`). A cold run (cache cleared) recompiles all kernels on 16 chips and exceeds 10 min. Fixes:
   (a) run launches in the **background**, poll the log; (b) recover the wedge by resetting **both hosts
   concurrently** (`ssh remote 'tt-smi -r' & tt-smi -r ; wait`) then wait ~90 s — sequential resets
   mistimed the QSFP retrain and stayed Stuck; concurrent + settle cleared it to PASS.

**Also set:** local kernel cache dirs created on both hosts; confirmed `/home/namvu/.cache` is on local
`sda2` (not the mount). Preflight `test_system_health` PASS from the mounted `build_Release`.

**STATUS: DONE. Mounted framework is now the documented/verified path (GUIDE + findings updated).**

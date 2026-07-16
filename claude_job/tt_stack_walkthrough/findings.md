# Findings — TT stack walkthrough (delta over `../dual_t3k_ops/findings.md`)

Verified 2026-07-15 on the mounted framework `/home/namvu/dual-t3k/tt-metal`, commit `3632044cfd6`,
launcher `t3k-node-b` + remote `t3k-node-a`, NIC `ens18`. This file records only what is **new** here;
the base env, the 13 hardware gotchas, and the two launch commands live in
[`../dual_t3k_ops/findings.md`](../dual_t3k_ops/findings.md).

## The proof workload (5 ops green across 16 chips)

`scripts/stack_workload.py` runs add, matmul, all_gather, all_reduce, reduce_scatter on the 1×16
Big-Mesh, verified per local shard vs torch golden. Launch = the Big-Mesh command from the base
findings, pointing at this script. Captured PASS: `scripts/PASS_output.txt`. Exit 0, clean teardown.

- CCL signatures used (stable ops): `all_gather(t, dim, cluster_axis=1, topology=Linear)`;
  `all_reduce(t, cluster_axis=1, topology=Linear)` (Sum, **no `dim`**);
  `reduce_scatter(t, dim, cluster_axis=1, topology=Linear)`. `Topology.Linear` for the 1×16 line.
- Accuracy: `all_gather` is a pure move → **exact** (pcc 1.0). add/matmul/all_reduce/reduce_scatter
  re-round in bf16 → ~0.9999. Threshold 0.99 for those, 0.999999 for all_gather.

## NEW gotcha #14 — deallocate intermediate tensors between ops on a many-chip mesh

A first run that allocated tensors for every op without freeing them crashed **rank 1 (the remote)**
partway through with:
```
Signal: Bus error (7), code: Non-existent physical address
  tt::umd::write16_to_device → SystemMemoryManager::fetch_queue_write → …write_shard_to_device
```
i.e. a host→device **command-queue ring** write hit an unbacked page — host-side CQ/pinned-buffer
resource accumulation. **Fix:** `ttnn.deallocate()` each op's tensors before the next op, and
`ttnn.distributed_context_barrier()` between ops to keep ranks in lockstep. After that, all 5 ops pass.
A SIGBUS rank death is a hard crash → **wedges the fabric** (MPI aborts the peer before teardown);
recover with the `tt-smi -r` both-hosts procedure (base findings #4/#13).

## NEW gotcha #15 — `run_fabric_manager` is a SINGLE-HOST tool; its split lifecycle doesn't fit a T3K

`tools/scaleout/fabric_manager/run_fabric_manager.cpp` proves the `FabricManagerMode` split lifecycle,
but on a dual-T3K Big-Mesh it has two hard limits:

1. **Hard-coded host rank.** `set_config_vars()` does `setenv("TT_MESH_HOST_RANK","0",1)` (overwrite)
   for **every** process (`run_fabric_manager.cpp:107`). The control plane reads exactly that env var to
   bind a host to its half of the mesh (`control_plane.cpp:275`). A Big-Mesh (`host_topology 1×2`) needs
   host_rank 0 **and** 1, so the tool cannot bring up the 1×16. It is single-host by construction
   (hence the galaxy 8×4 examples).
2. **ENABLED/TERMINATE break on T3K remote-over-ethernet discovery.** `--initialize-fabric` on one T3K
   (2×4) **works** — fabric up on 8 chips, `fabric_status.txt` written, fabric left up. But then:
   - an **ENABLED** attach (`set_fabric_config(FABRIC_2D, RELAXED_INIT, fabric_manager_mode=ENABLED)` +
     `open_mesh_device`) **fails**: `Timed out waiting for ETH heartbeat … Stuck at 0xabcd….` inside
     `TopologyDiscovery::discover_remote_devices`. T3K chips 4–7 are remote n300 halves reached over
     ethernet; the running EDM routers occupy those eth cores, so UMD can't get the heartbeat it needs
     to open the device. The `"Fabric initialized through Fabric Manager"` path
     (`fabric_firmware_initializer.cpp:320`) is never reached.
   - `--terminate-fabric` **fails the same way** (it also opens the device first → same heartbeat wall →
     unhandled `UmdException` → `std::terminate`). So on a T3K the FM fabric can be neither attached nor
     torn down once up; recovery = `tt-smi -r` both hosts.

   This is expected: `tests/scale_out/test_ccl_fabric_manager.py` is 8×4 **Galaxy-only** and still gated
   behind `# TODO: Enable … once Fabric Manager is ready`. On a Galaxy all chips are PCIe-direct, so
   there's no remote-over-eth discovery to conflict. Evidence: `scripts/fm_demo/01_init.log`,
   `02_enabled_workload.log`, `03_terminate.log`.

3. **The fix — drive the split lifecycle in ONE process** (`scripts/fm_lifecycle.py`,
   `scripts/fm_demo/04_lifecycle.log` + `04_TRANSCRIPT_inprocess.md`). The eth-heartbeat wall only bites a
   *fresh* process (new UMD `Cluster` ⇒ re-discovery). In one process the `Cluster` is built once, before
   any fabric is up, and reused across every mesh open/close — so re-discovery never happens. Sequence:
   `set_fabric_config(FABRIC_2D, fabric_manager_mode=INIT_FABRIC)` + open/close (fabric up, left up) →
   `set_fabric_config(FABRIC_2D, …=ENABLED)` (allowed: FABRIC_2D→FABRIC_2D keeps the config, only the
   manager mode changes; the guard at `metal_env.cpp:245` only needs the mesh closed) + open. **The ENABLED
   ATTACH now SUCCEEDS on the T3K** — captured `"Fabric initialized through Fabric Manager"`
   (`fabric_firmware_initializer.cpp:320`), the exact line the separate-process CLI could never reach.
   - **Remaining boundary:** a workload *dispatched* under the in-process ENABLED reattach **hangs on its
     first device write** on a T3K (diagnosed live: no compiler procs, 0 kernels built, main thread in
     `futex_wait`, one thread spinning 99% ⇒ hung device op, not a compile). Remote-chip (4-7) dispatch
     tunnels over the same ethernet the ENABLED-mode `configure_ethernet_cores_for_fabric_routers` (run
     while routers are live) disturbs. So a fully-green ENABLED workload is still not achievable on this
     T3K/commit. Killing the hung process leaves the fabric up → `tt-smi -r` both hosts to recover.
   - **Bottom line:** the FM mechanism + the ENABLED attach are proven on HW; the fully-green fabric proof
     stays the DEFAULT-mode 16-chip workload (`scripts/PASS_output.txt`). DEFAULT = INIT|TERMINATE = the
     same lifecycle with Metal owning both ends and no mid-life reattach.

## FabricManagerMode semantics (reference)

`fabric_types.hpp:44` bit-flags, gated in `fabric_firmware_initializer.cpp`:
`DEFAULT = INIT|TERMINATE` (Metal owns the lifecycle — what our proof workload uses),
`ENABLED = neither` (attach to someone else's fabric; teardown at `:347` is a no-op without the
TERMINATE flag → safe), `INIT_FABRIC` (up, leave up), `TERMINATE_FABRIC` (tear down).
Python: `ttnn.set_fabric_config(config, reliability_mode=…, fabric_manager_mode=…)`.

## Reusable file citations (spot-checked this session)

`fabric.cpp:501` SetFabricConfig · `metal_env.cpp:345` initialize_fabric_config ·
`distributed.py:644` open_mesh_device · `ttrun.py:1530` sets per-rank TT_MESH_HOST_RANK ·
`control_plane.cpp:275` reads it · `control_plane.cpp:2068` write_routing_tables_to_all_chips ·
`ccl_worker_builder.cpp:1139` append_fabric_connection_rt_args · single-T3K default MGD =
`tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto` (2×4, host 1×1, mesh 0).

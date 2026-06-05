# Two-rank send / compute / recv debug example

A minimal "orchestrator + worker" example to learn how cross-process
device communication works in tt-train.

## What it does

```
   Rank 0 (orchestrator)              Rank 1 (worker)
   submesh: 1x4 (4 chips)             submesh: 1x4 (4 chips)
   TT_VISIBLE_DEVICES=0,1             TT_VISIBLE_DEVICES=2,3
   (boards 0+1, 2 chips/board)        (boards 2+3, 2 chips/board)

   build a (1.5 fill)
   build b (2.25 fill)
   send a   ----------------------->  recv a
   send b   ----------------------->  recv b
                                       c = a + b   (on-device)
   recv c  <-----------------------   send c
   assert allclose(c, 3.75)
```

Both ranks run the **same** `example.py` and branch on `rank()`.

## Hardware assumption: full T3K

This example targets a T3K (a.k.a. "wh loudbox"): 4x N300 boards wired into
a 2x4 mesh of 8 wormhole chips. Each board is a single PCIe device with 2
chips on it (chip 0 reachable over PCIe, chip 1 reachable from chip 0 via
the on-board eth link), so `TT_VISIBLE_DEVICES` enumerates *boards* (valid
IDs `0..3`), not chips.

**All 4 boards must be described in the MGD even when a logical task only
uses a subset.** UMD's topology discovery walks the physical eth fabric
and learns the ASIC ID at the far end of every eth port. If the MGD doesn't
account for one of those ASICs, `ControlPlane::init_control_plane()`
aborts with:

```
TT_FATAL: Fabric node ID not yet assigned for ASIC id <id>
```

So the MGD splits the T3K into two `[1, 4]` line submeshes (4 chips each:
2 boards * 2 chips/board) and rank 0 / rank 1 each own one half. The 4
chips on a side snake as `b0-c0 - b1-c0 - b1-c1 - b0-c1`. This shape pairs
naturally with `enable_ddp=true` in `device.yaml`: ttml's
`ParallelismContext` takes the `is_line_topology` branch and assigns
the DDP axis to the non-trivial dim (length 4). See
`tech_reports/EthernetMultichip/BasicEthernetGuide.md` for the underlying
T3K topology and `tt_metal/fabric/mesh_graph_descriptors/` /
`tests/tt_metal/tt_fabric/custom_mesh_descriptors/` for the catalog of
shipped MGDs.

## Why MPI / `tt-run` and not `subprocess`

tt-train's `DistributedContext` is built on MPI: ranks, world size, and
the `SocketManager` send/recv addresses are all MPI rank ids. There is
no `multiprocessing.spawn`-style "orchestrator forks workers" path.
Both processes have to come up under the same MPI world.

`tt-run` (`ttnn/ttnn/distributed/ttrun.py`) is a thin wrapper around
`mpirun` that additionally:

1. Maps each MPI rank to a `(mesh_id, mesh_host_rank)` from
   `mgd.textproto`, so each process opens the right submesh.
2. Sets per-rank env vars (here, `TT_VISIBLE_DEVICES`) so each process
   only sees its half of the physical devices.

So the launch model is: **`tt-run` brings up N processes simultaneously
under MPI, and rank 0 plays the orchestrator role from inside the same
binary.**

## Files

| File | What it is |
|------|------------|
| `example.py` | The script. Both ranks run it and branch on rank id. |
| `runner.sh` | Convenience wrapper that invokes `tt-run`. |
| `configurations/local2/mgd.textproto` | Mesh Graph Descriptor: two `M0` (`[1, 4]`) line-mesh instances covering the whole T3K, wired with an inter-mesh `connections {}` block for fabric sockets. |
| `configurations/local2/rank_bindings.yaml` | Maps rank 0 -> mesh 0 with `TT_VISIBLE_DEVICES="0,1"` (boards 0+1 = 4 chips), rank 1 -> mesh 1 with `TT_VISIBLE_DEVICES="2,3"` (boards 2+3 = 4 chips). |
| `configurations/local2/hosts.txt` | `localhost slots=2`. |
| `configurations/local2/device.yaml` | Per-rank `mesh_shape: [1, 4]`, `enable_ddp: true`, `enable_tp: false`. Consumed by `ttml.common.utils.initialize_device`. |

## Run

```bash
export TT_METAL_HOME=/localdev/ichovpan/tt-metal   # adjust to your checkout
./tt-train/sources/examples/grpo_speedup/debug/runner.sh
```

Expected output (rank prefixes added by `mpirun --tag-output`):

```
[1,0]<stdout>:[rank 0/2] orchestrator ready
[1,1]<stdout>:[rank 1/2] worker ready
[1,0]<stdout>:[rank 0] sending a (filled with 1.5) to rank 1
[1,1]<stdout>:[rank 1] waiting for a from rank 0
[1,0]<stdout>:[rank 0] sending b (filled with 2.25) to rank 1
[1,1]<stdout>:[rank 1] waiting for b from rank 0
[1,1]<stdout>:[rank 1] sending a+b to rank 0
[1,0]<stdout>:[rank 0] received a+b: first 4 elements = [3.75, 3.75, 3.75, 3.75] (expected ~3.75)
[1,0]<stdout>:[rank 0] OK
[1,1]<stdout>:[rank 1] OK
```

The 32x32 input tile is **replicated** across each rank's 4-chip
`[1, 4]` submesh (default behavior of `ttml.autograd.Tensor.from_numpy`),
so `a + b` is an elementwise op evaluated on every device, and
`result.to_numpy().flatten()` reads back one device's worth of values
(same as the `mpi_minimal_example.py` pattern).

## Knobs

* **MPI vs FABRIC sockets.** Toggle `SOCKET_TYPE` at the top of
  `example.py`:
  * `SocketType.MPI` -- transit over the host network (TCP/IP). Simpler,
    works without fabric.
  * `SocketType.FABRIC` -- device-to-device over the Tenstorrent ethernet
    fabric. Faster. Requires the inter-mesh `connections {}` block in
    `mgd.textproto` (already declared here).
* **Different per-rank submesh shapes on T3K.** Other proven shapes from
  the in-tree catalog:
  * `[2, 2]` per rank with 2 ranks:
    `tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto`
    pattern, `TT_VISIBLE_DEVICES="0,1"` / `"3,2"`. Use this when both
    DDP and TP are enabled (2D mesh requires `num_enabled_parallelisms == 2`).
  * `[1, 2]` per rank with 4 ranks:
    `tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x2_mesh_graph_descriptor.textproto`.
  * `[1, 1]` per rank with 8 ranks:
    `tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x1_mesh_graph_descriptor.textproto`.
  In every case the union of the per-rank submeshes must equal the full
  8-chip T3K -- you cannot cover only a strict subset.
* **More ranks.** Scale to N ranks by adding more `mesh_descriptors`
  instances and `rank_bindings` entries. The `hierarchical_parallel`
  example under `tt-train/sources/examples/python/multihost/` shows the
  N-worker + aggregator + optimizer pattern.

## Where this lines up with existing examples

This is intentionally a slimmed-down version of
`tt-train/sources/examples/python/multihost/fabric_minimal_example/`:
same launcher, same MGD/rank-bindings/hostfile shape, but two ranks on a
single T3K instead of two loudboxes, and the all-reduce body replaced
with a simple `a + b` on the worker.

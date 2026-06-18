# moe_compute + cluster_axis=0 all_gather: tripped device ASSERT (GLM-4.7 decode, WH galaxy)

Status: **root cause partially isolated** (smoking-gun watcher assert captured; exact
triggering condition is full-model-only — not yet reduced to a standalone repro). Build:
tt-metal detached HEAD `c5ebc635109`, branch `mvasiljevic/moe-compute-axis0-deadlock-repro`.
Investigated by hand-integrating `ttnn.experimental.moe_compute` into a GLM-4.7 4-layer
decode model (3 dense + 1 MoE layer, mesh (4,8), FABRIC_1D_RING, DispatchCoreAxis.COL).

## Symptom

A GLM-4.7 decode forward using moe_compute (cluster_axis=0) hangs. The MoE layer itself
runs to completion (correct combine output `[8,16,5120]`, correct data flow); the forward
then appears to hang at the lm_head's `ttnn.all_gather(dim=0, cluster_axis=0)` (gather batch
across the 4 mesh rows). Wall-clock symptom = host blocks forever.

## Smoking gun (watcher)

Running the full model with `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DUMP_ALL=1`, the watcher
**stopped the device on a tripped assert** (not a passive fabric stall):

```
Watcher stopped the device due to tripped assert (watcher_device_reader.cpp:805)
Device 0 worker core(x=1,y=0) virtual(x=19,y=18): BRISC tripped an assert on line 260.
  Current kernel: ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp
Last waypoint: NSMD,CRBW,W,W,W
  BRISC:  .../all_gather_async/.../minimal_default_writer.cpp
  NCRISC: .../all_gather_async/.../minimal_default_reader.cpp
```

So the "hang" is really a **tripped ASSERT in the all_gather_async writer kernel** on the
core running a `cluster_axis=0` all_gather. An asserted RISC stalls → the host CCL waits on
its semaphores forever → looks like a deadlock.

## Where the assert is — pinpointed

`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp`,
**barrier-sync** block (gated on `use_barrier_sem`, lines ~220-272). Watcher reports BRISC
asserted at "line 260" (with "line number may be from a different file" → an inlined callee).
Line-number map around there:

```
238:  if constexpr (topology == Topology::Linear) { ...          // Linear branch
245:      safe_get_noc_addr(out_ready_sem_noc0_x, ...)            // (Linear)
253:      safe_get_noc_addr(opposite_core_sem_noc0_x, ...)        // (Linear, antipodal)
259:  } else if constexpr (topology == Topology::Ring) {         // Ring branch
260:      // multicast to entire ring of workers going in the same direction   <-- watcher "260"
262:      safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, barrier_sem, 0);
263:      fabric_multicast_noc_unicast_atomic_inc_with_state<...>(...)
267:  } else { ASSERT(false); }                                  // line 268
```

So the failing all_gather took the **Ring branch** (compile-time `topology == Topology::Ring`),
and the BRISC tripped on the **inlined `safe_get_noc_addr(out_ready_sem_noc0_x,
out_ready_sem_noc0_y, barrier_sem, 0)` at line 262** — i.e. a **bad NOC coordinate /
out-of-range core** for the barrier-multicast target. It is **NOT** the `ASSERT(false)` at 268
(that is the unknown-topology else; line 260 is the Ring branch).

`Topology` enum = `{NeighborExchange=0, Linear=1, Ring=2, Mesh=3, Torus=4}`
(tt_metal/api/tt-metalium/experimental/fabric/fabric_edm_types.hpp:12). `out_ready_sem_noc0_x/y`
(and the antipodal `opposite_core_sem_noc0_x/y`) are runtime args; the host computes the
opposite/target cores via `mesh_device->worker_core_from_logical_core(...)`
(all_gather_async_default_program_factory.cpp:663) and only enables this barrier path when a
`barrier_semaphore` is provided and not using persistent buffers (PF:734). The even-ring
special case `ring_size % 2 == 0 && ring_size > 2` (writer.cpp:278) is triggered by our
**4-device** axis-0 ring.

Best assessment: a **topology=Ring** `cluster_axis=0` all_gather over the **4-device** mesh-row
ring computes an **invalid `out_ready_sem` NOC core coordinate** for its barrier multicast and
trips `safe_get_noc_addr`'s validity assert. **DispatchCoreAxis.COL** (which moe_compute forces
— see below) reshapes the worker grid to 7x10 (column x=7 reserved), changing the
`worker_core_from_logical_core` mapping the barrier targets are derived from — the prime
suspect for the bad coordinate. (The all_gather is most likely an *internal* ring-fabric
all_gather, e.g. the index/score all-gather inside `all_to_all_dispatch_metadata`, since the
hand-written model all_gathers pass `topology=Linear`; the watcher abort lands in the MoE
layer before `dispatch_metadata` returns.)

## Why DispatchCoreAxis.COL is involved

moe_compute hardwires its tilize cores at the (5-6, 8-9) grid corner; under the default ROW
dispatch the DRAM matmul-core assignment spans the whole 8x9 grid and overlaps them
("tilize and matmul bounding boxes cannot overlap",
moe_compute/device/moe_compute_program_factory.cpp:98). COL dispatch (what the deepseek TG
reference uses) avoids the overlap but yields a 7x10 worker grid (x=7 reserved). So any model
using moe_compute on this build must run COL dispatch, which then perturbs the core/ring
geometry that the all_gather_async barrier-sync depends on.

## NOT reproducible in isolation (important — bisection)

`repro_moe_compute_axis0_deadlock.py` (this dir; random GLM-shaped weights, no HF load) was
used to bisect. ALL of these **pass cleanly** (no hang/assert):
- `--probe sync`  : dispatch_metadata + moe_compute(cluster_axis=0) + `synchronize_device`.
- `--probe axis0` : ... + `all_gather(dim=0, cluster_axis=0)` on the combine output.
- `--probe model_seq` : ... + axis-1 reduce_scatter/all_gather epilogue + a lm_head-shaped
  `all_gather(dim=0, cluster_axis=0)` on a `[16,1,18944]` tensor.
- `--probe model_seq --warmup 3` : prepend 3 rounds of cluster_axis=0 all_gather +
  cluster_axis=1 reduce_scatter/all_gather (mimicking the 3 dense layers + router axis-0
  traffic) before the moe_compute + axis-0 all_gather. **Still passes.**

So moe_compute + cluster_axis=0/1 CCLs (even with warmup traffic) is fine on its own. The
assert only fires in the **full GLM model**. Since the warmup CCL traffic did NOT reproduce
it, the trigger is **not** the moe_compute↔CCL ordering per se. The remaining differences the
harness does not model are the prime suspects:
- the **attention KV-cache point-to-point** (`PointToPointOp`, cluster_axis=0) run 4× before
  the MoE layer (a different op family than all_gather), and
- the **large persistent per-layer KV-cache tensors** (multi-GB) the model allocates, i.e.
  **DRAM/L1/core-allocation pressure** that can shift the internal all_gather's auto-assigned
  worker cores onto an invalid coordinate under the 7x10 COL grid.

i.e. the trigger is an emergent full-model interaction (most likely allocation/placement
pressure), not the standalone op sequence. Reducing further would essentially require
rebuilding the attention + KV-cache path, so **the full GLM model is the practical repro**
(below).

The full-model repro is the GLM-4.7 graph in the ttnn-models repo
(`mvasiljevic/glm-4.7-perf-tuning`, commit f220417): `GLM_CHECK_PCC=1 python main.py` hangs;
with `TT_METAL_WATCHER=10` it aborts on the tripped assert above. Per-op stderr markers show
dense layers 0-2 complete, then the abort lands in the MoE layer's / lm_head's cluster_axis=0
all_gather.

## REFUTED hypothesis (recorded so others don't chase it)

An earlier hypothesis blamed a template-arg bug in the moe combine writer
(`moe/selective_reduce_combine/device/kernels/dataflow/writer.cpp:334` passing `true` to
`fabric_multicast_bidirectional_atomic_inc_ring_1d<...>`). This is **NOT a bug**: the function
signature is `<LinearizedSrcMeshCoord, MeshRows, MeshCols, Axis, bool DoubleAntipodalAtomicInc=false, class SenderType>`
(ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp:788). The `true` correctly sets
`DoubleAntipodalAtomicInc`; `dispatch_devices` is computed *internally* (= MeshRows = 4 for
axis-0). And empirically moe_compute's combine terminates cleanly (`--probe sync` passes). The
fault is in **all_gather_async**, not the moe combine.

## Other ruled-out fixes (model-level)

- lm_head all_gather topology Ring vs Linear; num_links 4 vs auto.
- moe_compute topology=Ring + num_links=4.
- `ttnn.synchronize_device` between the MoE layer and lm_head (in the FULL model it also
  hangs — consistent with a RISC already stuck on the tripped assert; in isolation sync is
  fine).
- `USE_TORUS_MODE=1` (demo-only env; only selects FABRIC_1D_RING / Ring topology in
  models/demos/deepseek_v3 helpers; not read in tt_metal core).

## Related issues (none match exactly; do not open new ones)

- OPEN #41009 High Batch Deepseek - Optimized MoE Integration Followups
- OPEN #45794 P100 nightly test_moe_compute_single_card_deepseek FATALs: expected 12 matmul
  cores, got 11 (same tilize/matmul core-count area)
- CLOSED #42538, #41280, #43150, #43528 — DeepSeek MoE / all_to_all_dispatch / combine /
  reduce-scatter hangs on Galaxy (related symptom family, different ops).

## Suggested next steps for a tt-metal dev

1. Dump the failing all_gather_async instance's CT args (`topology`, `ring_size`,
   `num_targets_forward/backward`) and runtime args (`out_ready_sem_noc0_x/y`,
   `opposite_core_sem_noc0_x/y`, `use_barrier_sem`) on core (1,0) device 0 in the full GLM
   run — confirm whether the assert is `ASSERT(false)` (bad `topology` CT arg) or an inlined
   `safe_get_noc_addr` assert (bad opposite/out_ready NOC coord under COL dispatch).
2. Check `worker_core_from_logical_core(supplemental_core)` / the antipodal opposite-core
   computation for a 4-device cluster_axis=0 ring under DispatchCoreAxis.COL (7x10 grid).
3. Verify all_gather_async barrier-sync handles a cluster_axis=0 even ring (ring_size=4) when
   the dispatch axis is COL — the deepseek reference only exercises an 8-device dispatch axis.

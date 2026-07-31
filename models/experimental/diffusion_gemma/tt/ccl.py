# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-owned copy of the Gemma-4 CCL helpers.

WHY THIS COPY EXISTS. The shared `models/demos/gemma4/tt/ccl.py` routes `ccl_allreduce` through plain
`ttnn.all_reduce`, which internally reaches `ttnn::prim::reduce_scatter`. On a **program-cache miss**
that op's `create_mesh_workload` calls `create_global_semaphore`, and `GlobalSemaphore::setup_buffer`
does a *blocking* `enqueue_write_mesh_buffer` that ends in `FDMeshCommandQueue::finish_nolock` — a full
command-queue drain. Two failure modes, both measured on QB2:

1. **HANG (the serious one).** Inside `begin_trace_capture` commands are RECORDED, not executed, so a
   command-queue finish can never complete. Up-front denoise capture therefore deadlocks permanently
   and needs `tt-smi -r`. Confirmed by gdb on a hung `serving_smoke --upfront --num-layers 2`: the main
   thread parks in `pthread_cond_wait` under `finish_nolock`, below
   `create_mesh_workload -> create_global_semaphore -> setup_buffer -> reset_semaphore_value ->
   enqueue_write_mesh_buffer`. This is the same class as the documented
   `TT_FATAL: Writes are not supported during trace capture` for `ttnn.full` / `ttnn.zeros_like`, except
   it hangs instead of raising. The `AllBroadcast` second-request stall already recorded in this tree is
   the all-gather sibling of the same bug (`all_gather_via_broadcast_factory.cpp:297-298` creates two
   global semaphores the same way).
2. **LATENCY.** Outside capture each miss still forces a full CQ drain. A DG prefill issues ~90
   all-reduces (3 per layer x 30 layers), which is the measured prefill regression from ~0.2 s to ~12 s
   at constant prompt length — prompt-length independent, and invisible to traced denoise because trace
   replay never rebuilds a program.

THE FIX. `CCLManager` already pre-creates every semaphore these collectives need — 3 per
reduce-scatter, 2 per all-gather, plus a barrier — and the shared file never used them, because the live
path called the plain ops. The experimental decomposed ops accept caller-provided semaphores and never
call `create_global_semaphore`:
  * `ttnn.experimental.reduce_scatter_minimal_async` requires exactly 3
    (`reduce_scatter_minimal_async_op_device_operation.cpp:56-62`) and its directory contains no
    `create_global_semaphore` call at all.
  * `ttnn.experimental.all_gather_async` requires exactly 2 for its default factory
    (`all_gather_async_device_operation.cpp:44-58`: `MINIMAL_DEFAULT` `TT_FATAL`s on
    `semaphore.size() == 2`). The semaphore-creating `VIA_BROADCAST` factory is only selected by the
    explicit `use_all_gather_async_via_broadcast` opt-in, which we never pass.
So passing the pre-created semaphores removes the blocking write, which removes both failure modes.

The shared file left this path commented out pending a perf sweep; the point here is not speed, it is
that the plain path carries a reliability hazard nobody had attributed. Do NOT "simplify" this back to
`ttnn.all_reduce` / `ttnn.all_gather`.

`nonmoe_roofline`'s "the decomposed reduce_scatter+all_gather measures identically (0.67 ms)" result is
**not** an argument against this change: that was an eager per-op null, and the reason to switch is
semaphore avoidance.
"""

import ttnn
from models.common.utility_functions import is_blackhole


def default_num_links():
    """Default TP-collective link count for the current arch.

    Blackhole boards expose 2 ethernet links between adjacent mesh devices, so
    reduce-scatter / all-gather can run at ~2x bandwidth vs a single link — and
    on Gemma4 prefill the per-layer all-reduces are ~31% of device time, so this
    is the single highest-ROI CCL knob. Wormhole (T3K) defaults to 1 link here
    (its multi-link tuning needs a separate sweep).
    """
    return 2 if is_blackhole() else 1


SEMAPHORE_BUFFER_DEPTH = 2
"""How many semaphore sets to round-robin through.

The shared file chose 2 and the getters cycle on every call, so a 30-layer forward with 3 collectives
per layer cycles this 45 times. Reuse is safe here because DiffusionGemma runs a **single command
queue**: programs on one CQ execute in issue order, so collective N has completed before N+2 is
dispatched, and the double buffer is already one full collective of slack beyond what is needed.

Raise this if a second command queue is ever enabled (then two collectives really can be in flight and
2 is no longer obviously enough). Note that depth alone cannot rescue a semaphore that is never reset —
it only buys time for an eventual reset — so if a digest ever diverges, investigate the reset, do not
just raise the depth. The digest gate in `doc/optimize_perf/ccl_semaphore_deadlock.md` is what actually
proves the cycling is sound: it exercises ~13,000 collective invocations per run.
"""


class CCLManager:
    """CCL manager for DiffusionGemma tensor parallelism.

    Stores mesh_device reference and num_links for CCL operations, plus the pre-created global
    semaphores the decomposed async collectives require. Constructor and getter signatures are
    deliberately identical to the shared Gemma-4 version so call sites and tests are drop-in.
    """

    def __init__(self, mesh_device, num_links=None, topology=ttnn.Topology.Linear, buffer_depth=None):
        if num_links is None:
            num_links = default_num_links()
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.num_devices = mesh_device.get_num_devices()

        # These are the semaphores the decomposed collectives consume, which is what keeps
        # create_global_semaphore (and its blocking, capture-deadlocking device write) off the hot path.
        # Counts are fixed by the ops: reduce_scatter_minimal_async validates exactly 3, all_gather_async's
        # default factory validates exactly 2. Created ONCE per manager, before any trace capture.
        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        core_range_set = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

        depth = SEMAPHORE_BUFFER_DEPTH if buffer_depth is None else int(buffer_depth)
        if depth < 1:
            raise ValueError(f"buffer_depth must be >= 1, got {depth}")
        self._rs_semaphores = []
        self._ag_semaphores = []
        self._barrier_semaphores = []
        for _ in range(depth):
            self._rs_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)])
            self._ag_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)])
            self._barrier_semaphores.append(ttnn.create_global_semaphore(mesh_device, core_range_set, 0))
        ttnn.synchronize_device(mesh_device)

        self._depth = depth
        self._rs_idx = 0
        self._ag_idx = 0
        self._barrier_idx = 0

    def get_rs_semaphore(self):
        """Returns list of 3 semaphores for reduce_scatter (round-robins the buffer)."""
        sems = self._rs_semaphores[self._rs_idx]
        self._rs_idx = (self._rs_idx + 1) % self._depth
        return sems

    def get_ag_semaphore(self):
        """Returns list of 2 semaphores for all_gather (round-robins the buffer)."""
        sems = self._ag_semaphores[self._ag_idx]
        self._ag_idx = (self._ag_idx + 1) % self._depth
        return sems

    def get_barrier_semaphore(self):
        """Returns single barrier semaphore (round-robins the buffer)."""
        sem = self._barrier_semaphores[self._barrier_idx]
        self._barrier_idx = (self._barrier_idx + 1) % self._depth
        return sem


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce across TP devices, decomposed so no collective creates a global semaphore.

    `ttnn.all_reduce` is deliberately NOT used here — see this module's docstring. The reduce-scatter
    then all-gather decomposition is mathematically the same all-reduce and moves the same bytes
    (`all_reduce` is itself reduce_scatter + all_gather internally), but both halves take their
    semaphores from `ccl_manager`, so `create_global_semaphore` — and the blocking command-queue drain
    that deadlocks trace capture — is never reached.
    """
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis

    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        tensor,
        dim=3,
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        multi_device_global_semaphore=ccl_manager.get_rs_semaphore(),
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    gathered = ttnn.experimental.all_gather_async(
        scattered,
        dim=3,
        cluster_axis=tp_axis,
        mesh_device=ccl_manager.mesh_device,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        memory_config=memory_config,
    )
    scattered.deallocate(True)
    return gathered


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """DG-local replacement for `gemma4.tt.attention.operations.apply_allreduce`.

    Signature-compatible with the shared helper (which also ignores `hidden_size` and just forwards to
    `ccl_allreduce`). It exists so that DG call sites reaching the collective through `apply_allreduce`
    — attention out_proj, the batched and decode commits, chunked prefill — land on the DG
    semaphore-passing `ccl_allreduce` above rather than the shared one. Importing the shared
    `apply_allreduce` would silently route those four sites back to `ttnn.all_reduce` and reintroduce
    the capture deadlock on the attention collective, which is 30 of the ~90 per forward.
    """
    return ccl_allreduce(tensor, mesh_config, ccl_manager)


def ccl_allgather(tensor, mesh_config, ccl_manager, dim=3, memory_config=None):
    """All-gather across TP devices, with caller-provided semaphores.

    Same reasoning as `ccl_allreduce`: passing 2 semaphores selects `all_gather_async`'s
    `MINIMAL_DEFAULT` factory, which creates none of its own. The semaphore-creating `VIA_BROADCAST`
    factory is reached only via the explicit `use_all_gather_async_via_broadcast` opt-in, and it is the
    path behind the `AllBroadcast` second-request stall recorded in this tree.

    **NOT CURRENTLY ON ANY DiffusionGemma PATH — kept deliberately, and it is not yet validated.**
    Nothing in `models/experimental/diffusion_gemma/` calls this. The one all-gather in the denoise step
    is the LM-head gather at `models/demos/gemma4/tt/model.py:942`, which does a *local* import of the
    SHARED `ccl_allgather`, so it still runs the plain `ttnn.all_gather` and **retains the semaphore
    hazard**. Fixing that requires DiffusionGemma to own the LM-head path (a copy of `_apply_lm_head`),
    which is a separate, much larger change. This function exists so the copied module keeps the shared
    API surface and so that change is a one-line re-point when someone makes it. Do not cite it as
    evidence the all-gather hazard is fixed.
    """
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis

    gathered = ttnn.experimental.all_gather_async(
        tensor,
        dim=dim,
        cluster_axis=tp_axis,
        mesh_device=ccl_manager.mesh_device,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return gathered

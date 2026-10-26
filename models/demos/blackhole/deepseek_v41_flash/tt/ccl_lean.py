# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lean CCLs (lever L07): the gpt_oss MeshConfig.allreduce / allgather with ONE worker per link (``num_workers_per_link=1``).

Measured on the 4x8 galaxy (traced, [R,5120] bf16 per device, R=16/32): reduce_scatter_minimal_async 19.5 -> 17.2 us, all_gather_async of the [R,640] scatter result 22.6 -> 20.4 us,
RS+AG pair 41.5 -> 37.4 us (the stock ``ttnn.reduce_scatter`` + AG pair is 48.8 us). The result of every op is bit-identical (same ring order; the worker count only splits the transfer).
DSV41_CCL_LEAN=0 restores the stock MeshConfig methods. All-gathers are lean only when the caller passes ``lean_ag=True`` (the MoE combine gather of layer.py, [T, 640] per device): a one-worker gather of a big tensor (Engram kv) is slower (measured +137 us per layer in the 8-layer window)."""
import os
import types

import ttnn

WORKERS = int(os.environ.get("DSV41_CCL_WORKERS", "1"))


class LeanSems:
    """Own ping-pong semaphores for the one-worker ops: sharing the CCLManager's semaphores between ops of different worker counts gave wrong results at T < 16
    (measured: RS(w1)+AG(default) and RS+AG(w1) pairs after each other on the same semaphores: PCC 0.88 / 0.93 against 1.0).
    """

    def __init__(self, ccl):
        mk = lambda: ttnn.create_global_semaphore(ccl.mesh_device, ccl.ccl_cores, 0)
        self.rs = [[mk() for _ in range(3)] for _ in range(2)]
        self.ag = [[mk() for _ in range(2)] for _ in range(2)]
        self.bar = [mk() for _ in range(2)]
        self.i = [0, 0, 0]

    def _next(self, k, lst):
        v = lst[self.i[k]]
        self.i[k] ^= 1
        return v

    def rs_sem(self):
        return self._next(0, self.rs)

    def ag_sem(self):
        return self._next(1, self.ag)

    def barrier(self):
        return self._next(2, self.bar)


def sems(ccl):
    if not hasattr(ccl, "_lean_sems"):
        ccl._lean_sems = LeanSems(ccl)
    return ccl._lean_sems


def _allreduce(self, tensor, ccl_manager, memory_config=None, pad_size=None, axis=0):
    assert not pad_size
    mem = memory_config or ttnn.DRAM_MEMORY_CONFIG
    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        tensor,
        dim=3,
        multi_device_global_semaphore=sems(ccl_manager).rs_sem(),
        num_links=ccl_manager.num_links,
        memory_config=mem,
        topology=ccl_manager.topology,
        cluster_axis=axis,
        barrier_semaphore=sems(ccl_manager).barrier(),
        num_workers_per_link=WORKERS,
    )
    tensor.deallocate(True)
    gathered = ttnn.experimental.all_gather_async(
        scattered,
        dim=3,
        cluster_axis=axis,
        mesh_device=ccl_manager.mesh_device,
        topology=ccl_manager.topology,
        multi_device_global_semaphore=sems(ccl_manager).ag_sem(),
        num_links=ccl_manager.num_links,
        memory_config=mem,
        barrier_semaphore=sems(ccl_manager).barrier(),
        num_workers_per_link=WORKERS,
    )
    scattered.deallocate(True)
    return gathered


def _allgather(self, tensor, ccl_manager, memory_config=None, axis=0, dim=3, linear=False, lean_ag=False):
    lean = lean_ag and dim == 3 and not linear
    return ttnn.experimental.all_gather_async(
        tensor,
        dim=dim,
        cluster_axis=axis,
        mesh_device=ccl_manager.mesh_device,
        topology=ttnn.Topology.Linear if linear else ccl_manager.topology,
        multi_device_global_semaphore=sems(ccl_manager).ag_sem() if lean else ccl_manager.get_ag_ping_pong_semaphore(),
        num_links=ccl_manager.num_links,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        barrier_semaphore=sems(ccl_manager).barrier() if lean else ccl_manager.get_barrier_semaphore(),
        num_workers_per_link=WORKERS if lean else None,
    )


def make_lean(mesh_config):
    if os.environ.get("DSV41_CCL_LEAN", "1") != "0":
        mesh_config.allreduce = types.MethodType(_allreduce, mesh_config)
        mesh_config.allgather = types.MethodType(_allgather, mesh_config)
        mesh_config._dsv41_lean = True
    return mesh_config


def make_lean_sems(ccl):
    if os.environ.get("DSV41_CCL_LEAN", "1") != "0":
        sems(ccl)

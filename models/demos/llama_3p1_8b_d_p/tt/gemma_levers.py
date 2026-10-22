# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 prefill levers applied to Llama (local, env-gated).

LOCAL_LLAMA_GMM=1   Gemma4 matmul blocking and compute config (#57454, #58224), unmodified functions: the 1D
                    config where it applies, else the 2D one. LOCAL_LLAMA_GMM_2D=qkv,gate_up,down,o forces
                    the 2D function for those ops, grid width the widest divisor of N (gemma4_d_p/tt/mlp.py) or,
                    with LOCAL_LLAMA_GMM_FULLX=1, the full grid width (its attention projections).
LOCAL_LLAMA_SP=row  Gemma4 sequence-parallel residual (#58033): rows split across TP, plain norms.
LOCAL_LLAMA_GL1=1   Gemma4 short-lived MLP activations in L1.
LOCAL_LLAMA_GL1A=1  Same for attention-side activations: qkv output, concatenated heads, mid-layer residual (hidden SP).
LOCAL_LLAMA_GCCL=rs|all  Gemma4 async reduce-scatter (and all-gather with "all"), workers 4, chunks_per_sync 2,
                    buffers 2 (#59221, #59222); needs SP=row.
"""

import os

import ttnn
from models.demos.gemma4_d_p.tt.matmul_config import prefill_1d_matmul_program_config, prefill_matmul_program_config

# Rows per device from which the projections are FPU-bound (gemma4_d_p/tt/attention/operations.py).
_LOFI_PROJECTION_MIN_ROWS = 512

_cache = {}


def on(name):
    return os.getenv(f"LOCAL_LLAMA_{name}") == "1"


def sp_row():
    return os.getenv("LOCAL_LLAMA_SP") == "row"


def gmm():
    return on("GMM")


def projection_math_fidelity(rows):
    return ttnn.MathFidelity.LoFi if rows >= _LOFI_PROJECTION_MIN_ROWS else ttnn.MathFidelity.HiFi2


def projection_configs(x, weight, mesh_device, name):
    """(program_config, compute_kernel_config) as Gemma4's projection_matmul_configs, or (None, None)."""
    key = (name, tuple(x.padded_shape), tuple(weight.padded_shape))
    if key not in _cache:
        grid = mesh_device.compute_with_storage_grid_size()
        n_tiles = weight.padded_shape[-1] // ttnn.TILE_SIZE
        # Widest divisor of N as gemma4_d_p/tt/mlp.py, or the full grid width as its attention projections.
        grid_x = grid.x if os.getenv("LOCAL_LLAMA_GMM_FULLX") == "1" else max(
            d for d in range(1, grid.x + 1) if n_tiles % d == 0
        )
        force_2d = name in os.getenv("LOCAL_LLAMA_GMM_2D", "").split(",")
        program_config = None if force_2d else prefill_1d_matmul_program_config(x, weight, grid)
        program_config = program_config or prefill_matmul_program_config(
            x, weight, grid_x if force_2d else grid.x, grid.y, fp32_dest_acc=True
        )
        compute_kernel_config = None
        if program_config is not None:
            compute_kernel_config = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=projection_math_fidelity(x.shape[-2]),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        _cache[key] = (program_config, compute_kernel_config)
    return _cache[key]


class _Ccl:
    """Global semaphores for the async collectives, double-buffered as Gemma4's CCLManager."""

    def __init__(self, mesh_device):
        grid = mesh_device.compute_with_storage_grid_size()
        cores = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)

        def sems(n):
            return [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(n)]

        self.rs = [sems(3) for _ in range(2)]
        self.ag = [sems(2) for _ in range(2)]
        self.barrier = sems(2)
        self.count = {"rs": 0, "ag": 0, "barrier": 0}
        ttnn.synchronize_device(mesh_device)

    def next(self, kind):
        i = self.count[kind]
        self.count[kind] = (i + 1) % 2
        return getattr(self, kind)[i]


_ccl = {}


def _manager(mesh_device):
    if _ccl.get("device") is not mesh_device:
        _ccl.clear()
        _ccl["device"], _ccl["ccl"] = mesh_device, _Ccl(mesh_device)
    return _ccl["ccl"]


def _async_params():
    return dict(
        chunks_per_sync=int(os.getenv("LOCAL_LLAMA_CCL_CHUNKS", "2")),
        num_workers_per_link=int(os.getenv("LOCAL_LLAMA_CCL_WORKERS", "4")),
        num_buffers_per_channel=int(os.getenv("LOCAL_LLAMA_CCL_NBUF", "2")),
    )


def reduce_scatter_rows(tensor, tp_axis):
    if os.getenv("LOCAL_LLAMA_GCCL") in ("rs", "all"):
        ccl = _manager(tensor.device())
        return ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=None,
            dim=2,
            multi_device_global_semaphore=ccl.next("rs"),
            barrier_semaphore=ccl.next("barrier"),
            num_links=2,
            cluster_axis=tp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            **_async_params(),
        )
    return ttnn.reduce_scatter(
        tensor,
        dim=2,
        cluster_axis=tp_axis,
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
    )


def all_gather_rows(tensor, tp_axis):
    if os.getenv("LOCAL_LLAMA_GCCL") == "all":
        ccl = _manager(tensor.device())
        return ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=2,
            multi_device_global_semaphore=ccl.next("ag"),
            num_links=2,
            cluster_axis=tp_axis,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.next("barrier"),
            **_async_params(),
        )
    return ttnn.all_gather(
        tensor,
        dim=2,
        cluster_axis=tp_axis,
        num_links=2,
        topology=ttnn.Topology.Ring,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

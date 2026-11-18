# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.tt_ccl import TT_CCL

EPS = 1e-6
NUM_LINKS = 2
LOFI_FP32 = dict(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def multi_op_partial_mlp(h, w_norm, w_gate_up, w_down):
    """This chip's partial of the down projection, before the cross-chip sum and the residual add."""
    compute_config = ttnn.WormholeComputeKernelConfig(**LOFI_FP32)
    grid = h.device().compute_with_storage_grid_size()
    n = ttnn.rms_norm(h, weight=w_norm, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    product = ttnn.experimental.minimal_matmul(
        n,
        w_gate_up,
        config=ttnn.MinimalMatmulConfig(
            M_block_size=4,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=4,
            compute_with_storage_grid_size=(grid.x, grid.y),
        ),
        compute_kernel_config=compute_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        fuse_swiglu=True,
    )
    return ttnn.experimental.minimal_matmul(
        product,
        w_down,
        config=ttnn.MinimalMatmulConfig(
            M_block_size=4,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=4,
            compute_with_storage_grid_size=(grid.x, grid.y),
        ),
        compute_kernel_config=compute_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )


def multi_op_mlp(h, w_norm, w_gate_up, w_down, ccl: TT_CCL):
    """y = h + MLP(RMSNorm(h)) as a chain of existing ttnn ops."""
    mesh = h.device()
    partial_y = multi_op_partial_mlp(h, w_norm, w_gate_up, w_down)
    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        ttnn.reshape(partial_y, [1, 1, *partial_y.shape]),
        dim=3,
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
        num_links=NUM_LINKS,
        multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(1),
        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gathered = ttnn.experimental.all_gather_async(
        scattered,
        dim=3,
        cluster_axis=1,
        mesh_device=mesh,
        topology=ttnn.Topology.Ring,
        num_links=NUM_LINKS,
        multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.add(h, ttnn.reshape(gathered, list(h.shape)), memory_config=ttnn.DRAM_MEMORY_CONFIG)

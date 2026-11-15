# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""`swiglu_last_chunk` for all_gather_minimal_matmul_async on the Blackhole galaxy.

One op over [W_a (plain chunks) | W_b (tile-pair-interleaved gate/up)] must write the same bytes as the two ops it
replaces: the chunked matmul over W_a and the fused-SwiGLU matmul over W_b. The shapes give each separate op one N
block per core, so both sum K in the order the grouped split uses.
"""

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_wormhole_b0

from models.common.utility_functions import comp_pcc
from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
    create_fabric_router_config,
    create_global_semaphores,
    make_ag_buffers,
)
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

TILE = 32
M, K = 8192, 3072
PLAIN_CHUNK = 768  # per device; three plain chunks
SWIGLU_OUT = 1152  # per device; the SwiGLU chunk's weight is twice as wide
CORE_GRID = ttnn.CoreCoord(12, 9)  # transposed: 9 N cores, 8 tiles of each group per core


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": create_fabric_router_config(4096),
        }
    ],
    indirect=True,
    ids=["fabric_ring"],
)
def test_all_gather_minimal_matmul_swiglu_last_chunk(mesh_device, expect_error):
    if is_wormhole_b0():
        pytest.skip("core grid (12, 9) exceeds wormhole_b0 compute grid (8x8), blackhole-only config")

    cluster_axis, sp_axis = 0, 1
    tp, sp = mesh_device.shape[cluster_axis], mesh_device.shape[sp_axis]
    m_local = M // sp
    torch.manual_seed(0)
    x = torch.randn(M, K).bfloat16()
    w_plain = [(torch.randn(K, 3 * PLAIN_CHUNK) * K**-0.5).bfloat16() for _ in range(tp)]
    gate_up = [(torch.randn(K, 2 * SWIGLU_OUT) * K**-0.5).bfloat16() for _ in range(tp)]
    w_swiglu = [prepare_for_fused_swiglu(w, ndev=1, gate_is_first=True) for w in gate_up]

    def shard(t, dims):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )

    x_dims = [None, None]
    x_dims[cluster_axis], x_dims[sp_axis] = 1, 0
    w_dims = [None, None]
    w_dims[cluster_axis] = 1
    tt_x = shard(x, x_dims)
    tt_w_plain = shard(torch.cat(w_plain, dim=-1), w_dims)
    tt_w_swiglu = shard(torch.cat(w_swiglu, dim=-1), w_dims)
    tt_w_both = ttnn.concat([tt_w_plain, tt_w_swiglu], dim=-1)

    ccl_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 8))})
    ag_buffer = make_ag_buffers(mesh_device, m_local, K, ttnn.bfloat16, 1)[0]
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    config = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=2,
        subblock_w=1,
        compute_with_storage_grid_size=CORE_GRID,
    )

    def agmm(weight, **kwargs):
        return ttnn.experimental.all_gather_minimal_matmul_async(
            tt_x,
            weight,
            config=config,
            compute_kernel_config=compute_config,
            persistent_output_buffer=ag_buffer,
            multi_device_global_semaphore=create_global_semaphores(mesh_device, tp * sp, ccl_cores, 0),
            num_links=2,
            topology=ttnn.Topology.Ring,
            cluster_axis=cluster_axis,
            force_transpose=True,
            num_workers_per_link=6,
            **kwargs,
        )

    plain_sizes = [PLAIN_CHUNK] * 3
    separate = list(agmm(tt_w_plain, chunks=3, chunk_sizes=plain_sizes)) + list(agmm(tt_w_swiglu, fuse_swiglu=True))
    fused = agmm(tt_w_both, chunks=4, chunk_sizes=plain_sizes + [2 * SWIGLU_OUT], swiglu_last_chunk=True)
    assert len(fused) == 4

    for c, (a, b) in enumerate(zip(separate, fused, strict=True)):
        for i, (da, db) in enumerate(zip(ttnn.get_device_tensors(a), ttnn.get_device_tensors(b), strict=True)):
            out = ttnn.to_torch(db)
            assert torch.equal(ttnn.to_torch(da).view(torch.int16), out.view(torch.int16)), f"chunk {c} device {i}"
            t, s = divmod(i, sp)
            x_s = x[s * m_local : (s + 1) * m_local].float()
            if c < 3:
                ref = x_s @ w_plain[t][:, c * PLAIN_CHUNK : (c + 1) * PLAIN_CHUNK].float()
            else:
                gate, up = (x_s @ gate_up[t].float()).chunk(2, dim=-1)
                ref = torch.nn.functional.silu(gate) * up
            passing, pcc = comp_pcc(ref, out.reshape(ref.shape).float(), 0.999)
            assert passing, f"chunk {c} device {i}: pcc {pcc}"

    with expect_error(RuntimeError, "swiglu_last_chunk"):
        agmm(tt_w_both, swiglu_last_chunk=True)

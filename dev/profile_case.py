# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One untraced sparse_ring_sdpa call at a production shape, for the device profiler. Dev branch only.

Setup is seconds: Q / K / V and the pooled K / V are generated on device (ttnn.rand, values in [-1, 1)), the
persistent all-gather buffers are allocated uninitialized, and only the block mask is built on host, at the
per-device size (one TP group's heads, replicated over TP). Nothing is read back or checked.

    TT_METAL_DEVICE_PROFILER=1 pytest dev/profile_case.py -s -k "s10 and k256 and d020 and sparse_only"

ids: shape s05 / s10 / s15 (4736 / 9184 / 13664 rows per device), piece size k128 / k256 / k384 / k512, mask
density d010 / d020 / d050 / d100 (random fraction of full blocks kept, plus block 0 and the +-1 band), and
sparse_only / plus_dense (plus_dense also runs one dense ring-joint call; dev/profile_summary.py takes each
device's longest run as the attention op, which is then the dense one).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    call_sdpa,
    close_ring_joint_sdpa_runtime,
    open_ring_joint_sdpa_runtime,
)
from tests.nightly.blackhole.sdpa.test_sparse_ring_sdpa import (
    BLOCK,
    D,
    HEADS_PER_DEVICE,
    Geometry,
    make_structured_mask,
)

SHAPES = {"s05": (4736, (320, 384)), "s10": (9184, (256, 512)), "s15": (13664, (256, 512))}  # rows, dense chunks
DENSITIES = {"d010": 0.1, "d020": 0.2, "d050": 0.5, "d100": 1.0}


@pytest.fixture(scope="module")
def runtime():
    rt = open_ring_joint_sdpa_runtime(MESH_CONFIG)
    yield rt
    close_ring_joint_sdpa_runtime(rt)


def block_mask(geo, S, H, density, band=1, seed=0):
    """[S, H, nb_pad, S * nb_pad] int32 0/1, the structure of test_sparse_ring_sdpa.make_mask (block 0 of every
    shard, the |i - j| <= band diagonal in global block order, random full blocks), vectorized. The block lengths
    follow from rows and logical_n = S * rows: only padding blocks past nb, and a partial last block, are not full."""
    g = torch.Generator().manual_seed(seed)
    nb, nb_pad = geo.nb(), geo.nb_pad()
    b = torch.arange(nb_pad)
    lens = (geo.rows - b * BLOCK).clamp(0, BLOCK)  # the same for every shard
    full = (lens == BLOCK).repeat(S)  # [S * nb_pad]
    mask = torch.rand(S, H, nb_pad, S * nb_pad, generator=g) < density
    k_global = torch.arange(S * nb_pad)
    k_index = (k_global // nb_pad) * nb + (k_global % nb_pad)
    for s in range(S):
        q_index = s * nb + b
        mask[s, :, (q_index[:, None] - k_index[None, :]).abs() <= band] = True
    mask[..., torch.arange(S) * nb_pad] = True
    mask &= full
    mask[:, :, nb:] = False
    return mask.to(torch.int32)


@pytest.mark.parametrize("q_blocks_per_pass", [None, 10], ids=["qauto", "q10"])
@pytest.mark.parametrize("with_dense", [False, True], ids=["sparse_only", "plus_dense"])
@pytest.mark.parametrize("density_id", list(DENSITIES))
@pytest.mark.parametrize("k_chunk", [256, 128, 384, 512], ids=lambda k: f"k{k}")
@pytest.mark.parametrize("shape_id", list(SHAPES))
def test_profile(runtime, shape_id, k_chunk, density_id, with_dense, q_blocks_per_pass):
    rt = runtime
    dev = rt.mesh_device
    S = MESH_CONFIG.sp_size
    rows, dense_chunks = SHAPES[shape_id]
    geo = Geometry(rows, rows * S, k_chunk, dense_chunks=dense_chunks)
    Hd = HEADS_PER_DEVICE

    def rand(shape, seed):
        return ttnn.rand(shape, dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, low=-1.0, high=1.0, seed=seed)

    # Per-device shapes; ttnn.rand replicates them, which is all the op looks at.
    tq, tk, tv = (rand([1, Hd, rows, D], 11 + i) for i in range(3))
    t_pk, t_pv = (rand([1, Hd, S * geo.nb_pad(), D], 21 + i) for i in range(2))
    buf_k, buf_v = (
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, Hd, S * rows, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(2)
    )
    # DEV_PROFILE_STRUCTURE=band|clustered|uneven|random: make_structured_mask's layouts at the same density (default:
    # this file's random mask with the +-1 band).
    structure = os.environ.get("DEV_PROFILE_STRUCTURE")
    if structure:
        mask = make_structured_mask(geo, S, Hd, structure=structure, density=DENSITIES[density_id]).to(torch.int32)
    else:
        mask = block_mask(geo, S, Hd, DENSITIES[density_id])
    d = [None, None]
    d[rt.sp_axis] = 0  # shard s's rows on SP device s; the same heads' mask on every TP row
    t_mask = ttnn.from_torch(
        mask,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        mesh_mapper=ttnn.ShardTensor2dMesh(dev, mesh_shape=tuple(dev.shape), dims=d),
    )
    kept = mask[:, :, : geo.nb()].float().mean().item()
    logger.info(
        f"[profile {shape_id} k{k_chunk} {density_id} {structure or 'random+band'}] rows {rows}, nb {geo.nb()}, mask mean {kept:.3f}"
    )
    ttnn.synchronize_device(dev)  # inputs in place on every device before the op starts

    out = ttnn.transformer.sparse_ring_sdpa(
        tq,
        tk,
        tv,
        t_mask,
        t_pk,
        t_pv,
        persistent_output_buffer_k=buf_k,
        persistent_output_buffer_v=buf_v,
        q_blocks_per_pass=q_blocks_per_pass,
        logical_n=geo.logical_n,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=rt.sdpa_compute_grid,
            q_chunk_size=BLOCK,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        ),
        compute_kernel_config=rt.compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=rt.ccl_semaphore_handles,
        num_links=rt.num_links,
        cluster_axis=rt.sp_axis,
        mesh_device=dev,
        topology=rt.topology,
        subdevice_id=rt.worker_sub_device_id,
        ccl_core_grid_offset=(rt.ccl_column, 0),
        use_column_major_ccl=True,
        block_size=BLOCK,
    )
    ttnn.synchronize_device(dev)
    if with_dense:
        call_sdpa(
            tq,
            tk,
            tv,
            geo.logical_n,
            False,
            False,
            buf_k,
            buf_v,
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=rt.sdpa_compute_grid,
                q_chunk_size=dense_chunks[0],
                k_chunk_size=dense_chunks[1],
                exp_approx_mode=False,
            ),
            rt.compute_kernel_config,
            rt.ccl_semaphore_handles,
            rt.num_links,
            rt.sp_axis,
            dev,
            rt.topology,
            rt.worker_sub_device_id,
            rt.ccl_column,
        )
        ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    logger.info(f"[profile {shape_id}] done, output {list(out.shape)}")

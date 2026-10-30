# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Can the block-cyclic reorder be done in a fixed number of ops instead of one slice per block?

``_gather_and_reorder`` used to restore natural token order by slicing every 256-token block out of
the gathered buffer and concatenating them in permuted order. That is ``4 * slabs`` slice ops plus a
concat of the same arity, per K and V, per layer -- at a 32K prefix, 128 slices x 2 x 32 layers =
8192 slice dispatches for one chunk, each moving only 32 KB. Measured per-chunk cost grew ~24 ms
per extra 1024 gathered tokens, which is far more than those bytes justify, so the reorder was
dispatch-bound rather than bandwidth-bound.

``_reorder_natural`` replaced it with the shape-only route measured here, whose op count does NOT
depend on the prefix:

    reshape (1,1,capacity,128) -> (4,stride,128)   expose the SP rank as its own dim
    slice   each rank's active rows                ONE slice, not one per block
    reshape -> (4,slabs,256,128)                   expose the slab
    permute (1,0,2,3) -> (slabs,4,256,128)         slab-major IS natural order
    reshape -> (1,1,extent,128)                    then copy out, since all of the above are views

Both routes are graded against ground truth, not just against each other: chunk i is written as the
constant i+1, so natural order must read 1,1,...,2,2,...  A wrong permutation shows up as blocks in
the wrong order, and a broken tile view shows up as garbage. bfloat16 keeps small integers exact, so
the comparison is bit-exact and any difference is real.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tt.attention import FullCausalAttention
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache, write_kv_chunk

MESH_SHAPE = (4, 8)
SP, TP = MESH_SHAPE
SP_AXIS = 0
CHUNK = 1024
BLOCK = CHUNK // SP
HEAD_DIM = Llama31_8BConfig.HEAD_DIM
NUM_KV_HEADS = Llama31_8BConfig.NUM_KEY_VALUE_HEADS
CAPACITY = 8192
LAYER = 3


def _to_chunk(mesh_device, values):
    return ttnn.from_torch(
        values.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
    )


def _gather(cache_tensor, output_tensor, *, batch_index, extent):
    return ttnn.experimental.high_bw_all_gather(
        cache_tensor,
        dim=2,
        output_tensor=output_tensor,
        cluster_axis=SP_AXIS,
        num_links=1,
        input_batch_index=batch_index,
        gathered_dim_size=extent,
    )


def _reorder_by_blocks(gathered, geometry, extent):
    """Today's route: one slice per 256-token block, concatenated in permuted order."""
    blocks = [
        ttnn.slice(gathered, [0, 0, block * BLOCK, 0], [1, 1, (block + 1) * BLOCK, HEAD_DIM])
        for block in geometry.prefix_gather_block_order(extent)
    ]
    natural = ttnn.concat(blocks, dim=2)
    for block in blocks:
        block.deallocate(True)
    return natural


def _reorder_by_permute(gathered, capacity, extent):
    """Candidate route: a fixed number of shape ops, independent of the prefix length."""
    stride = capacity // SP
    slabs = extent // CHUNK
    full = slabs * BLOCK == stride
    ranked = ttnn.reshape(gathered, (SP, stride, HEAD_DIM))
    # A slice that spans every row is a no-op that hands back a view of the input, so at a full
    # prefix `active` IS the caller's persistent gather buffer and must be left alone.
    active = ranked if full else ttnn.slice(ranked, [0, 0, 0], [SP, slabs * BLOCK, HEAD_DIM])
    split = ttnn.reshape(active, (SP, slabs, BLOCK, HEAD_DIM))
    slab_major = ttnn.permute(split, (1, 0, 2, 3))
    # The reshapes are views and a permute of a unit-length dimension is a metadata swap, so the
    # result has to be copied out before `active` is freed; the concat the block route ends on
    # materializes the same bytes, which keeps the timings below comparable.
    natural = ttnn.clone(ttnn.reshape(slab_major, (1, 1, extent, HEAD_DIM)))
    if not full:
        active.deallocate(True)
    return natural


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_reorder_routes_agree_and_time(mesh_device):
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    # Borrow the module's own geometry and gather buffer rather than rebuilding them here, so the
    # probe exercises exactly the tensors the model gathers into.
    attention = FullCausalAttention(mesh_device, mesh_config, cache_dtype=ttnn.bfloat16, max_seq_len=CAPACITY)
    geometry = attention.geometry
    cache = allocate_kv_cache(mesh_device, mesh_config, max_seq_len=CAPACITY, cache_dtype=ttnn.bfloat16)
    # The attention owns persistent gather buffers and, after the control call below, a cached mask;
    # release them even if an assertion fails, since the mesh is shared with the rest of the suite.
    try:
        for index in range(CAPACITY // CHUNK):
            payload = torch.full((NUM_KV_HEADS, CHUNK, HEAD_DIM), float(index + 1))
            tt_k, tt_v = _to_chunk(mesh_device, payload), _to_chunk(mesh_device, payload)
            write_kv_chunk(
                cache,
                tt_k,
                tt_v,
                slot_idx=0,
                layer_idx=LAYER,
                actual_start=index * CHUNK,
                actual_end=(index + 1) * CHUNK,
            )
            tt_k.deallocate(True)
            tt_v.deallocate(True)
        ttnn.synchronize_device(mesh_device)

        buffer = attention.gathered_k
        batch_index = 0 * cache.num_layers + LAYER

        # The gather refused this pair once with "input and output tensors must be on the same mesh
        # device" even though both came from the same fixture, so say what the two tensors actually are.
        for name, tensor in (("cache.k", cache.k), ("buffer", buffer)):
            logger.info(
                f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} layout={tensor.layout} "
                f"storage={tensor.storage_type()} device={tensor.device()} mesh={mesh_device}"
            )
        # A call through the module itself is the control: if this works and the bare op does not, the
        # difference is in the probe, not in the op.
        probe_q = ttnn.from_torch(
            torch.zeros(1, Llama31_8BConfig.NUM_ATTENTION_HEADS, CHUNK, HEAD_DIM, dtype=torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
        )
        control = attention(probe_q, cache, slot_idx=0, layer_idx=LAYER, actual_start=0, actual_end=CHUNK)
        ttnn.synchronize_device(mesh_device)
        control.deallocate(True)
        probe_q.deallocate(True)
        logger.info("control: a full attention call through the module gathered fine")

        # Both routes must succeed. `_reorder_natural` takes the permute route unconditionally in
        # production, so an op that refuses it is a broken model, not an unsupported experiment --
        # this used to demote such a failure to a skip, which would have hidden exactly that.
        for extent in (1024, 2048, 4096, 8192):
            want = torch.cat([torch.full((CHUNK,), float(i + 1)) for i in range(extent // CHUNK)])

            gathered = _gather(cache.k, buffer, batch_index=batch_index, extent=extent)
            blocks_out = _reorder_by_blocks(gathered, geometry, extent)
            ttnn.synchronize_device(mesh_device)
            blocks_host = ttnn.to_torch(ttnn.get_device_tensors(blocks_out)[0]).float()[0, 0, :, 0]
            assert torch.equal(blocks_host, want), f"block route wrong at extent {extent}"
            blocks_out.deallocate(True)

            gathered = _gather(cache.k, buffer, batch_index=batch_index, extent=extent)
            permute_out = _reorder_by_permute(gathered, CAPACITY, extent)
            ttnn.synchronize_device(mesh_device)
            permute_host = ttnn.to_torch(ttnn.get_device_tensors(permute_out)[0]).float()[0, 0, :, 0]
            assert torch.equal(permute_host, want), (
                f"permute route wrong at extent {extent}: "
                f"first mismatch at {int((permute_host != want).nonzero()[0])}"
            )
            permute_out.deallocate(True)
            logger.info(f"extent {extent}: both routes reproduce natural order exactly")

        # Time each route in isolation. The gather is common to both and is timed separately so the
        # comparison is reorder-vs-reorder rather than gather-plus-reorder.
        reps = 20
        logger.info(f"{'extent':>8} {'gather ms':>10} {'blocks ms':>10} {'permute ms':>11} {'speedup':>8}")
        for extent in (1024, 2048, 4096, 8192):
            timings = {}
            for name, route in (("gather", None), ("blocks", _reorder_by_blocks), ("permute", _reorder_by_permute)):
                for _ in range(3):  # warm the program cache for this shape
                    gathered = _gather(cache.k, buffer, batch_index=batch_index, extent=extent)
                    if route is _reorder_by_blocks:
                        _reorder_by_blocks(gathered, geometry, extent).deallocate(True)
                    elif route is _reorder_by_permute:
                        _reorder_by_permute(gathered, CAPACITY, extent).deallocate(True)
                ttnn.synchronize_device(mesh_device)
                start = time.perf_counter()
                for _ in range(reps):
                    gathered = _gather(cache.k, buffer, batch_index=batch_index, extent=extent)
                    if route is _reorder_by_blocks:
                        _reorder_by_blocks(gathered, geometry, extent).deallocate(True)
                    elif route is _reorder_by_permute:
                        _reorder_by_permute(gathered, CAPACITY, extent).deallocate(True)
                ttnn.synchronize_device(mesh_device)
                timings[name] = (time.perf_counter() - start) / reps * 1e3
            blocks_only = timings["blocks"] - timings["gather"]
            permute_only = timings["permute"] - timings["gather"]
            ratio = blocks_only / permute_only if permute_only > 0 else float("inf")
            logger.info(
                f"{extent:>8} {timings['gather']:>10.3f} {blocks_only:>10.3f} " f"{permute_only:>11.3f} {ratio:>7.2f}x"
            )
    finally:
        cache.k.deallocate(True)
        cache.v.deallocate(True)
        attention.close()

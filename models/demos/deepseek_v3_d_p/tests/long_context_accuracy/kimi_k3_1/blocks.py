# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 device Blocks of the long-context cases (``common/prefill/long_context_accuracy/block.DeviceBlock``),
each fed host rows from the FP64 chain golden ``k31-l0-3-fp64-1m`` and returning host ``BlockOutputs`` in global
row order:

* ``KimiK31MoeBlock`` (m06n.11.4): ``TtMoEBlockFfn`` of layer 1 or 2 on one 4x4 stage; ``extras`` hold the routing
  (expert ids and weights); no carried state.
* ``KimiK31DsaBlock`` (m06n.11.5): ``ttMLA`` of layer 3 with the adapter's caches, sized from the context length;
  ``extras`` hold the indexer top-k; the carried state is the KV and index caches.

Each Block's code, including its imports, stays inside its own section below.
"""

from __future__ import annotations

# ==== MoE: KimiK31MoeBlock (m06n.11.4) ==============================================================================
import torch

from models.demos.common.prefill.long_context_accuracy.block import BlockOutputs
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage

MOE_TOP_K = KimiK31Config.NUM_EXPERTS_PER_TOKEN  # 16
MOE_NUM_EXPERTS = KimiK31Config.NUM_ROUTED_EXPERTS  # 896


def owner_positions(
    sp_row: int, tp_col: int, *, rows_per_sp_row: int, pad_rows: int, row_blocks: int, tp: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the MoE block chip (``sp_row``, ``tp_col``)'s owner rows (its routing rows, in order) sit in the chunk:
    (index into the chip's owner rows, chunk row) for every real row it owns. SP row r holds chunk rows
    ``[r * rows_per_sp_row, (r + 1) * rows_per_sp_row)`` plus ``pad_rows`` block padding rows at its end; in each of
    the ``row_blocks`` blocks of that SP row, TP column p owns the p-th of ``tp`` equal pieces."""
    sub = (rows_per_sp_row + pad_rows) // row_blocks
    piece = sub // tp
    pos = torch.cat([j * sub + tp_col * piece + torch.arange(piece) for j in range(row_blocks)])
    owned = torch.arange(pos.numel())
    keep = pos < rows_per_sp_row
    return owned[keep], sp_row * rows_per_sp_row + pos[keep]


def moe_routing(ffn, block, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The routing of the MoE block's last call as (checkpoint expert ids ``[rows, 16]`` int64, weights ``[rows, 16]``
    bf16) in the chunk's token order.

    ``block.last_head.routing_rows`` holds, on each chip, its owner rows: per token 16 expert ids (block order) then
    16 bf16 weights as uint16 words (``router_common.hpp routing_rows``). Mesh chip i = (mr, mc) is the block's chip
    (SP row, TP column) = (mr, mc) with SP on axis 0, (mc, mr) with SP on axis 1; ``owner_positions`` places its owner
    rows in the chunk. Block expert b is checkpoint expert ``block_order_perm[b]``."""
    import ttnn
    from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block_ffn import block_order_perm

    mesh = ffn.mesh
    R, C = tuple(mesh.shape)
    sp_axis = ffn.options.sp_axis
    sp, tp = (R, C) if sp_axis == 0 else (C, R)
    chunk = block.resources.config.chunk_tokens
    assert block.splits == 1, f"the routing rows hold the last of {block.splits} splits only"
    words = torch.full((chunk, 2 * MOE_TOP_K), -1, dtype=torch.int64)
    for i, d in enumerate(ttnn.get_device_tensors(block.last_head.routing_rows)):
        mr, mc = divmod(i, C)
        r, p = (mr, mc) if sp_axis == 0 else (mc, mr)
        owned, pos = owner_positions(
            r, p, rows_per_sp_row=chunk // sp, pad_rows=block.pad_rows, row_blocks=block.row_blocks, tp=tp
        )
        words[pos] = (ttnn.to_torch(d).to(torch.int64).reshape(-1, 2 * MOE_TOP_K) & 0xFFFF)[owned]
    assert (words[:rows] >= 0).all(), "a chunk row no chip's routing rows hold"
    perm = block_order_perm(mesh, MOE_NUM_EXPERTS // (R * C), sp_axis)
    return perm[words[:rows, :MOE_TOP_K]], words[:rows, MOE_TOP_K:].to(torch.int16).view(torch.bfloat16)


class KimiK31MoeBlock:
    """The production K3.1 MoE (``TtMoEBlockFfn``, the MoE block) of ``layer`` as one K3.1 stage of the mesh builds it
    (``k31_stage.STAGE_LAYOUTS``): its routed experts from the stage's TTNN weight cache (skips without one), its
    router, shared expert and latent pair from the adapter's checkpoint (``moe_block_checkpoint``).

    ``forward`` feeds bf16 rows ``[hi - lo, hidden]``, zero-padded to the chunk with the real span passed as the runtime
    passes it, and returns the MoE output (routed + shared, ``[hi - lo, hidden]``) and the routing
    (``extras["expert_ids"]`` in checkpoint expert numbering, ``extras["expert_weights"]``). No carried state."""

    def __init__(self, layer: int) -> None:
        self.layer = layer
        self.mesh = self.ffn = self.mapper = None
        self.chunk = 0

    def build(self, mesh, context_len: int, chunk: int) -> None:
        import ttnn
        from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block_ffn import MoEBlockFfnOptions, TtMoEBlockFfn
        from models.demos.deepseek_v3_d_p.tt.moe_impl import moe_impl_of

        assert moe_impl_of(k31_stage.ADAPTER) == "block", "the production K3.1 MoE is the MoE block"
        mesh_shape = tuple(mesh.shape)
        layout = k31_stage.STAGE_LAYOUTS[mesh_shape]
        assert chunk == layout.chunk, f"a {mesh_shape} K3.1 stage prefills {layout.chunk}-row chunks, not {chunk}"
        self.ffn = TtMoEBlockFfn(
            mesh,
            KimiK31Config,
            self.layer,
            cache_path=k31_stage.weight_cache_path(mesh_shape),
            options=MoEBlockFfnOptions(sp_axis=layout.sp_axis),
        )
        dims = [None, None]
        dims[layout.sp_axis], dims[layout.tp_axis] = -2, -1
        self.mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh_shape, dims=tuple(dims))
        self.mesh, self.chunk = mesh, chunk

    def forward(self, x: torch.Tensor, lo: int, hi: int) -> BlockOutputs:
        import ttnn
        from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import compose

        rows = hi - lo
        x_rows = torch.zeros(self.chunk, KimiK31Config.EMB_SIZE, dtype=torch.bfloat16)
        x_rows[:rows] = x
        tt_x = ttnn.from_torch(
            x_rows.reshape(1, 1, self.chunk, -1),
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mapper,
        )
        y = self.ffn(tt_x, actual_isl=rows, actual_start=lo)  # (the runtime passes every chunk's real span)
        ttnn.deallocate(tt_x)
        ids, weights = moe_routing(self.ffn, self.ffn._block(self.chunk), rows)  # (the call's cached block)
        output = compose(self.mesh, y, rows)
        ttnn.deallocate(y)
        return BlockOutputs(output, {"expert_ids": ids, "expert_weights": weights})

    def carried_state(self) -> dict[str, torch.Tensor]:
        return {}

    def release(self) -> None:
        from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block_ffn import release_blocks

        if self.mesh is not None:
            release_blocks(self.mesh)
        self.ffn = self.mapper = None


# ==== end of MoE ====================================================================================================

# ==== DSA: KimiK31DsaBlock (m06n.11.5) ==============================================================================
# isort: off
import torch

import ttnn
from models.demos.common.prefill.long_context_accuracy.block import BlockOutputs
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import CACHE_LEN, cache_rows, compose
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import k31_topk_rows

# isort: on


def device_selection(topk: torch.Tensor) -> torch.Tensor:
    """The device's top-k ``[rows, k]`` as int64 with its uint32 no-key sentinel (0xFFFFFFFF) as -1."""
    topk = topk.long()
    return torch.where((topk < 0) | (topk >= 2**32 - 1), -1, topk)


class KimiK31DsaBlock:
    """``ttMLA`` of DSA layer ``layer`` from the stage's TTNN weight cache, with the adapter's scaled-FP8 kvpe and bf8
    index-key caches (one user, eager, chunked prefill), sized for the run from the context length. ``forward`` feeds
    the host input zero-padded to the chunk and returns the attention output and ``extras["topk"]``, the indexer's
    selection (``device_selection``); ``carried_state`` reads the caches back in token order as ``kv_cache``
    ``[rows, lora + rope]`` and ``index_cache`` ``[rows, index_dim]``."""

    def __init__(self, layer: int):
        self.layer = layer
        self._mesh = self._layout = self._caches = self._mla = self._mapper = None
        self._rows = self._chunk = 0

    def build(self, mesh, context_len: int, chunk: int) -> None:
        from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

        mesh_shape = tuple(mesh.shape)
        layout = k31_stage.STAGE_LAYOUTS[mesh_shape]
        if chunk != layout.chunk:
            raise ValueError(f"the {mesh_shape} stage prefills {layout.chunk}-row chunks, not {chunk}")
        self._mesh, self._layout, self._rows, self._chunk = mesh, layout, context_len, chunk
        max_seq_len = -(-context_len // CACHE_LEN) * CACHE_LEN
        hf_config, params = k31_stage.window_params(
            mesh_shape, self.layer, 1, max_seq_len, is_first_rank=False, is_last_rank=True
        )
        adapter = k31_stage.ADAPTER
        self._caches = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf_config, params=params)
        self._mla = ttMLA(
            hf_config,
            {},
            mesh,
            layer_idx=self.layer,
            seq_len=max_seq_len,
            sp_axis=layout.sp_axis,
            tp_axis=layout.tp_axis,
            topology=per_axis_topology(mesh_device=mesh),
            is_balanced=False,  # chunked prefill is block-cyclic, as the runtime builds it
            weight_cache_path=params.weight_cache_path,
            is_chunked=True,
            active_seq_len=chunk,
            slot_num=adapter.kv_cache_slots(params),
            layer_num=1,
            first_layer_idx=self.layer,
            tp_shard_kv=True,
            sparse_kv_cache_format=adapter.resolve_sparse_kv_cache_format(params.sparse_kv_cache_format),
        )
        shard_dims = [None, None]
        shard_dims[layout.sp_axis], shard_dims[layout.tp_axis] = -2, -1
        self._mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh_shape, dims=tuple(shard_dims))

    def forward(self, x: torch.Tensor, lo: int, hi: int) -> BlockOutputs:
        mesh, layout, chunk = self._mesh, self._layout, self._chunk
        padded = torch.zeros(chunk, x.shape[-1], dtype=torch.bfloat16)
        padded[: hi - lo] = x
        x_tt = ttnn.from_torch(
            padded.reshape(1, 1, chunk, -1),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._mapper,
        )
        attn, indices = self._mla.forward(
            x_tt,
            {},  # K3.1's MLA is NoPE: no rotary tables
            self._caches.kvpe,
            index_kv_cache=self._caches.index,
            cache_layer_idx=0,
            actual_start=lo,
            actual_end=hi,
            cache_user_id=0,
            return_indexer_indices=True,
        )
        ttnn.deallocate(x_tt)
        topk = k31_topk_rows(mesh, indices, chunk, layout.sp_axis, layout.tp_axis)[: hi - lo]
        out = compose(mesh, attn, hi - lo)
        ttnn.deallocate(attn)
        ttnn.deallocate(indices)
        return BlockOutputs(output=out, extras={"topk": device_selection(topk)})

    def carried_state(self) -> dict[str, torch.Tensor]:
        return {
            "kv_cache": cache_rows(self._mesh, self._caches.kvpe, self._chunk, self._rows)[0],
            "index_cache": cache_rows(self._mesh, self._caches.index, self._chunk, self._rows)[0],
        }

    def release(self) -> None:
        if self._mla is not None:
            self._mla.release_sparse_mla_overlap_manager()
            self._mla = None
        if self._caches is not None:
            ttnn.deallocate(self._caches.kvpe.storage)
            if self._caches.index is not None:
                ttnn.deallocate(self._caches.index)
            self._caches = None


# ==== end of DSA ====================================================================================================

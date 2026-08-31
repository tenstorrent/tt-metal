# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-side VSA coarse stage for MiniMax-H3 (R3 of VSA_SCOPE.md), unfused ttnn ops.

Per attention call, given per-device Q/K/V heads (post QK-norm + RoPE, tiled order):

(a) pool local Q/K/V per 64-token tile via a matmul with the host-built block-diagonal
    averaging matrix (entries 1/valid_count, zero rows for pad tiles);
(b) all-gather pooled K_c/V_c on the SP axis (persistent-buffer all-gather);
(c) scores = Q_c @ K_c^T / sqrt(head_dim); O_c = softmax(scores) @ V_c broadcast tile -> 64 tokens;
(d) selection: top-k over candidate columns only, every row's index list =
    [all exempt tile ids] + [its top-k candidate ids] (exempt-query rows list all real tile ids),
    emitted as the uint32 sentinel-tailed index tensor vsa_sdpa consumes.

The pooling matmuls run transposed (X^T @ A^T) so the averaging matrix broadcasts across heads as
a batch-1 weight; the pooled tensors are tiny ([heads, head_dim, n_tiles]), so the extra
transposes are cheap. Everything here is static per (geometry, sparsity): no host readback, all
shapes fixed -- trace-compatible.

Exempt-query rows are blended in via a host-built row mask (int32 select on device), because the
per-shard count of exempt rows differs between shards and SPMD mesh ops need uniform shapes.
"""

from __future__ import annotations

import math

import torch

import ttnn

from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, MiniMaxH3VSAGeometry
from ....utils.tensor import from_torch

VSA_INDEX_SENTINEL = 0xFFFFFFFF
_TOPK_K_MULTIPLE = 16  # ttnn.experimental.topk_large_indices wants k % 16 == 0, k in [16, 2048]


def compute_topk(sparsity: float, num_candidates: int) -> int:
    """Candidate tiles to keep; FastVideo's compute_topk, clamped to [1, n]."""
    return max(1, min(math.ceil((1 - sparsity) * num_candidates), num_candidates))


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


class MiniMaxH3VSACoarseStage:
    """Uploads the static geometry tensors once; `__call__` runs the coarse stage on device."""

    def __init__(
        self,
        geometry: MiniMaxH3VSAGeometry,
        *,
        sparsity: float,
        head_dim: int,
        mesh_device: ttnn.MeshDevice,
        sp_axis: int,
        ccl_manager=None,
    ) -> None:
        self.geometry = geometry
        self.sparsity = sparsity
        self.head_dim = head_dim
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.ccl_manager = ccl_manager

        n_tiles = geometry.n_tiles
        tiles_per_shard = geometry.tiles_per_shard
        rows_per_shard = tiles_per_shard  # one selection row per 64-token q tile
        exempt_ids = torch.nonzero(geometry.is_exempt, as_tuple=False).reshape(-1)
        real_ids = torch.nonzero(geometry.valid_counts > 0, as_tuple=False).reshape(-1)
        self.n_exempt = int(exempt_ids.numel())
        self.n_candidates = int(geometry.is_candidate.sum())
        self.k = compute_topk(sparsity, self.n_candidates)
        self.k_pad = max(_TOPK_K_MULTIPLE, _round_up(self.k, _TOPK_K_MULTIPLE))
        # Index width: global tile count, padded so W*4 bytes meets DRAM row alignment and the
        # static row layout [exempt | top-k | sentinel tail] fits.
        self.index_width = _round_up(max(n_tiles, self.n_exempt + self.k), _TOPK_K_MULTIPLE)

        # --- averaging matrices, sharded: shard s owns the diagonal block of A^T ---
        matrix = geometry.averaging_matrix()  # [n_tiles, padded_len] fp64->fp32
        rows_local = geometry.padded_len // geometry.sp_factor
        blocks = [
            matrix[s * tiles_per_shard : (s + 1) * tiles_per_shard, s * rows_local : (s + 1) * rows_local].T
            for s in range(geometry.sp_factor)
        ]
        a_t = torch.cat(blocks, dim=0)  # [padded_len, tiles_per_shard]
        shard2 = [None] * len(list(mesh_device.shape)) + [None, None]
        mesh_axes = [..., sp_axis, None]
        del shard2
        self.a_t_kv = from_torch(
            a_t.reshape(1, 1, geometry.padded_len, tiles_per_shard),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            mesh_axes=mesh_axes,
        )
        self.a_t_q = from_torch(
            (a_t / math.sqrt(head_dim)).reshape(1, 1, geometry.padded_len, tiles_per_shard),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            mesh_axes=mesh_axes,
        )

        # --- selection constants (replicated; row space is shard-local but content is global) ---
        # additive candidate mask over score columns: 0 for candidates, -inf otherwise
        cand_mask = torch.where(geometry.is_candidate, 0.0, -float("inf")).to(torch.float32)
        self.cand_mask = from_torch(
            cand_mask.reshape(1, 1, 1, n_tiles), device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=None
        )

        # per-shard row mask: rows whose q tile is exempt take the dense (all real tiles) list
        row_exempt = geometry.is_exempt.reshape(geometry.sp_factor, rows_per_shard)
        dense_row = torch.full((self.index_width,), VSA_INDEX_SENTINEL, dtype=torch.int64)
        dense_row[: real_ids.numel()] = real_ids
        self._host_exempt_ids = exempt_ids
        self._host_dense_row = dense_row
        self._host_row_exempt = row_exempt

    def _upload_row_constants(self, num_heads: int) -> None:
        """Head-expanded selection constants (uploaded lazily once num_heads is known)."""
        if hasattr(self, "_rows_ready"):
            return
        geometry = self.geometry
        rows = geometry.tiles_per_shard
        w = self.index_width
        # tensors are [sp, H, rows, *], sharded on dim 0 across the SP axis -> per-device [1, H, rows, *]
        mesh_axes = [self.sp_axis, None, None, None]

        def upload(x: torch.Tensor, dtype, layout=ttnn.Layout.TILE) -> ttnn.Tensor:
            return from_torch(
                x.contiguous(), device=self.mesh_device, dtype=dtype, layout=layout, mesh_axes=mesh_axes
            )

        # [sp, H, rows, n_exempt] exempt prefix (identical content everywhere)
        prefix = (
            self._host_exempt_ids.to(torch.int32).reshape(1, 1, 1, -1).expand(geometry.sp_factor, num_heads, rows, -1)
        )
        self.exempt_prefix = upload(prefix, ttnn.uint32, ttnn.Layout.ROW_MAJOR)

        # [sp, H, rows, tail] sentinel tail
        tail = w - self.n_exempt - self.k
        sentinel = torch.full((geometry.sp_factor, num_heads, rows, tail), -1, dtype=torch.int32)
        self.sentinel_tail = upload(sentinel, ttnn.uint32, ttnn.Layout.ROW_MAJOR)

        # dense-list blend, int32 TILE domain: final = sparse * keep + dense_masked
        row_exempt = self._host_row_exempt.to(torch.int32)  # [sp, rows]
        keep = (1 - row_exempt).reshape(geometry.sp_factor, 1, rows, 1).expand(-1, num_heads, -1, w)
        dense = self._host_dense_row.to(torch.int32).reshape(1, 1, 1, w) * row_exempt.reshape(
            geometry.sp_factor, 1, rows, 1
        )
        dense = dense.expand(-1, num_heads, -1, -1)
        self.blend_keep = upload(keep, ttnn.int32)
        self.blend_dense = upload(dense, ttnn.int32)
        self._rows_ready = True

    def _all_gather(self, x: ttnn.Tensor, dim: int) -> ttnn.Tensor:
        if self.geometry.sp_factor == 1:
            return x
        return self.ccl_manager.all_gather_persistent_buffer(x, dim=dim, mesh_axis=self.sp_axis)

    def pool(self, x_bhnd: ttnn.Tensor, *, scaled: bool) -> ttnn.Tensor:
        """[1, H, S_local, d] -> pooled [1, H, tiles_local, d] (fp math in bf16, matches oracle to bf16)."""
        x_t = ttnn.transpose(x_bhnd, 2, 3)  # [1, H, d, S_local]
        pooled_t = ttnn.matmul(x_t, self.a_t_q if scaled else self.a_t_kv)  # [1, H, d, tiles_local]
        ttnn.deallocate(x_t)
        pooled = ttnn.transpose(pooled_t, 2, 3)  # [1, H, tiles_local, d]
        ttnn.deallocate(pooled_t)
        return pooled

    def __call__(
        self, q_bhnd: ttnn.Tensor, k_bhnd: ttnn.Tensor, v_bhnd: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the coarse stage. Returns (o_c [1,H,S_local,d] bf16, indices [1,H,rows,W] uint32 ROW_MAJOR)."""
        num_heads = q_bhnd.shape[1]
        self._upload_row_constants(num_heads)

        q_c = self.pool(q_bhnd, scaled=True)  # scores scale baked into the Q averaging matrix
        k_c_t = ttnn.transpose(self.pool(k_bhnd, scaled=False), 2, 3)  # [1, H, d, tiles_local]
        v_c = self.pool(v_bhnd, scaled=False)

        k_c_t_g = self._all_gather(k_c_t, dim=3)  # [1, H, d, n_tiles]
        v_c_g = self._all_gather(v_c, dim=2)  # [1, H, n_tiles, d]

        scores = ttnn.matmul(q_c, k_c_t_g)  # [1, H, tiles_local, n_tiles]
        ttnn.deallocate(q_c)

        # (c) coarse output, broadcast tile -> 64 tokens
        probs = ttnn.softmax(scores, dim=-1)
        o_c_tiles = ttnn.matmul(probs, v_c_g)  # [1, H, tiles_local, d]
        ttnn.deallocate(probs)
        o_c = ttnn.repeat_interleave(o_c_tiles, VSA_TILE_TOKENS, dim=2)  # [1, H, S_local, d]
        ttnn.deallocate(o_c_tiles)

        # (e) selection: top-k over candidate columns only
        masked = ttnn.add(scores, self.cand_mask)  # -inf on non-candidate columns
        ttnn.deallocate(scores)
        masked_rm = ttnn.to_layout(masked, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(masked)
        topk_ids = ttnn.experimental.topk_large_indices(masked_rm, k=self.k_pad)  # [1,H,rows,k_pad] uint32
        ttnn.deallocate(masked_rm)
        if self.k != self.k_pad:
            topk_ids = ttnn.slice(
                topk_ids, [0, 0, 0, 0], [1, num_heads, self.geometry.tiles_per_shard, self.k]
            )

        sparse_rows = ttnn.concat([self.exempt_prefix, topk_ids, self.sentinel_tail], dim=-1)  # [1,H,rows,W]
        ttnn.deallocate(topk_ids)

        # exempt-query rows take the dense list: final = sparse * keep + dense_masked (int32 select;
        # the sentinel 0xFFFFFFFF is -1 in int32 and survives the arithmetic)
        sparse_i32 = ttnn.typecast(ttnn.to_layout(sparse_rows, ttnn.TILE_LAYOUT), ttnn.int32)
        ttnn.deallocate(sparse_rows)
        blended = ttnn.add(ttnn.multiply(sparse_i32, self.blend_keep), self.blend_dense)
        ttnn.deallocate(sparse_i32)
        indices = ttnn.to_layout(ttnn.typecast(blended, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(blended)

        return o_c, indices

    def block_counts_tensor(self) -> ttnn.Tensor:
        """[1,1,1,W] uint32 valid tokens per block, replicated (vsa_sdpa's block_counts input)."""
        counts = torch.zeros(self.index_width, dtype=torch.int32)
        counts[: self.geometry.n_tiles] = self.geometry.valid_counts.to(torch.int32)
        return from_torch(
            counts.reshape(1, 1, 1, -1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=None,
        )

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""VSA routing pre-pass for ``ttnn.transformer.sparse_ring_sdpa``.

One instance per ladder rung. Every buffer has a shape fixed by the rung, the SP factor and the local heads, and
``update`` writes a request's geometry into them in place, so the rung's trace replays for every request it serves.
The request's top-k ``k`` is data too: top-k always runs at the rung's widest ``k`` and the k-th value is gathered
at the request's ``k``.

Per device, from the local Q/K/V shards [1, H, rows, D] laid out by ``vsa_geometry`` (one 64-row block per tile,
``rows = rung / S``), produces the sparse op's mask and counts and the coarse branch output:

* ``block_mask`` uint32 row-major [1, H, nb_pad, n_blocks_pad]: 1 = local Q tile attends global key tile
  ``s * nb_pad + b`` exactly. A row holds every exempt tile plus the row's top-k candidate tiles (every real tile
  for an exempt query tile); a pad block's row holds nothing, and the op outputs zeros for a row that keeps nothing.
* ``block_counts`` uint32 row-major [1, 1, 1, n_blocks_pad]: real tokens per global block, so the op drops the pad
  rows of a ragged tile.
* ``coarse_out`` [1, 1, rows, H * D]: softmax(Q̄ K̄ᵀ / sqrt(D)) V̄ per tile, copied to the tile's 64 rows, in the
  head-concatenated layout the gate is applied in. None when called with ``coarse=False``.

Plan: vsa_attn_pln.md next to this file.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch

import ttnn

from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, MiniMaxH3VSAGeometry
from ....utils.tensor import from_torch
from ....utils.torch_threads import cap_torch_threads_to_half_the_cpus
from ....utils.tracing import StateTensor

TILE = 32
PREFIX_EXTRA = 32


@dataclass(frozen=True)
class MiniMaxH3VSAConfig:
    sparsity: float


def compute_topk(sparsity: float, num_candidates: int) -> int:
    """Candidate tiles to keep; FastVideo's compute_topk, clamped to [1, n]."""
    return max(1, min(math.ceil((1 - sparsity) * num_candidates), num_candidates))


_TIME_VSA = os.environ.get("TIME_VSA", "").lower() in ("1", "true")


def _timed(timings: dict[str, float] | None, name: str, sync: ttnn.MeshDevice | None = None):
    """Record the block's host wall time in ms when `TIME_VSA` is "1" or "true" (read at import); otherwise a no-op
    that neither reads the clock nor drains the device. With `sync`, the device is drained before the clock stops so
    the time includes the block's device work."""
    if not _TIME_VSA:
        return nullcontext()
    return _timed_section(timings, name, sync)


@contextmanager
def _timed_section(timings: dict[str, float] | None, name: str, sync: ttnn.MeshDevice | None):
    if timings is None:
        yield
        return
    t0 = time.perf_counter()
    yield
    if sync is not None:
        ttnn.synchronize_device(sync)
    timings[name] = (time.perf_counter() - t0) * 1e3


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


class VSAPrepass:
    """Device-side VSA selection and coarse branch for one rung; shared by every attention layer."""

    def __init__(
        self,
        *,
        rung: int,
        sp_factor: int,
        sparsity: float,
        num_heads: int,
        head_dim: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
        sp_axis: int,
    ) -> None:
        assert tuple(mesh_device.shape)[sp_axis] == sp_factor, "the SP factor must match the mesh"
        self.rung = rung
        self.sp_factor = sp_factor
        self.sparsity = sparsity
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.rows = rung // sp_factor
        self.blocks_per_shard = -(-self.rows // VSA_TILE_TOKENS)
        self.nb_pad = _round_up(self.blocks_per_shard, TILE)
        self.n_blocks_pad = sp_factor * self.nb_pad
        # The widest top-k any geometry in the rung can need: every block a candidate.
        self.k_max = compute_topk(sparsity, sp_factor * self.blocks_per_shard)

        tiles = self.blocks_per_shard
        bcast_tiles_by_rows = torch.kron(torch.eye(tiles), torch.ones(1, VSA_TILE_TOKENS))[:, : self.rows]
        bcast_flat = torch.zeros(1, 1, self.rows, self.nb_pad)
        bcast_flat[0, 0, :, :tiles] = bcast_tiles_by_rows.T
        self.bcast_flat = from_torch(bcast_flat, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=None)
        self.hifi4 = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )

        self.pool_weights = StateTensor()
        self.always = StateTensor()
        self.pad_q_threshold = StateTensor()
        self.cand_mask = StateTensor()
        self.pad_mask = StateTensor()
        self.block_counts = StateTensor()
        self.kth_index = StateTensor()
        # The prefix query rows get their exact attention from the attention's prefix-row recompute, not from the
        # sparse op: these are its row count and its pad-row marker for this geometry.
        self.prefix_rows = 0
        self.prefix_marker = StateTensor()

    @cap_torch_threads_to_half_the_cpus()
    def update(self, geometry: MiniMaxH3VSAGeometry, *, traced: bool, timings: dict[str, float] | None = None) -> None:
        """Write one request's geometry into the rung's buffers; in place when `traced`, so a captured trace replays.

        `timings`, when given, receives each step's host wall time in ms.
        """
        sp, tiles = self.sp_factor, self.blocks_per_shard
        assert geometry.padded_len == self.rung and geometry.sp_factor == sp, "the geometry must fill this rung"

        with _timed(timings, "host masks"):
            k = compute_topk(self.sparsity, int(geometry.is_candidate.sum()))

            def global_cols(per_tile: torch.Tensor) -> torch.Tensor:
                """[n_blocks] -> [n_blocks_pad] in the op's padded global numbering (zero on padding slots)."""
                out = torch.zeros(sp, self.nb_pad, dtype=per_tile.dtype)
                out[:, :tiles] = per_tile.reshape(sp, tiles)
                return out.reshape(-1)

            real_cols = global_cols(geometry.valid_counts > 0)
            cand_cols = global_cols(geometry.is_candidate)
            exempt_cols = global_cols(geometry.is_exempt).float()

            # Columns a scored query tile keeps regardless of its scores: the exempt tiles. Per shard, since the
            # scored query tiles differ between shards.
            always = exempt_cols.reshape(1, 1, 1, -1).repeat(sp, 1, self.nb_pad, 1)
            # Only the video query tiles are scored. A pad block's query rows are dropped at unpacking, and the
            # prefix query rows are recomputed densely after the sparse op, so both keep nothing: the +inf threshold
            # empties the top-k part (a pad block's zero pooled query would otherwise tie every candidate score and
            # keep them all).
            real_q = (geometry.valid_counts > 0).reshape(sp, tiles)
            scored_q = real_q & ~geometry.is_exempt.reshape(sp, tiles)
            always[:, 0, :tiles] *= scored_q[:, :, None].float()
            pad_q_threshold = torch.zeros(sp, 1, self.nb_pad, 1)
            pad_q_threshold[:, 0, :tiles, 0] = torch.where(scored_q, 0.0, float("inf"))
            self.prefix_rows = geometry.n_prefix_tiles * VSA_TILE_TOKENS

        with _timed(timings, "pool weights"):
            # The weights of geometry.averaging_matrix(), with the same float64 reciprocal; pad rows weigh zero.
            block = geometry.block_of_row
            counts = geometry.valid_counts[block]
            real_row = torch.arange(self.rung) - geometry.block_starts[block] < counts
            weights = torch.where(real_row, 1.0 / counts.to(torch.float64), 0.0).to(torch.float32)
            # -1 in the marker channel of every key row the prefix recompute must not attend: the pad rows of each
            # shard (inside partial blocks and whole pad blocks) and the recompute's extra rows, except its probe key.
            extra_rows = self.rows + PREFIX_EXTRA
            prefix_marker = torch.zeros(sp, 1, extra_rows, self.head_dim + PREFIX_EXTRA)
            prefix_marker[:, 0, : self.rows, self.head_dim] = -(~real_row).float().reshape(sp, self.rows)
            prefix_marker[:, 0, self.rows : extra_rows - 1, self.head_dim] = -1.0

        device = self.mesh_device
        sharded = [self.sp_axis, None, None, None]
        with _timed(timings, "upload pool weights", sync=device):
            self.pool_weights.update(
                from_torch(weights.reshape(sp, 1, self.rows, 1), device=device, dtype=ttnn.bfloat16, mesh_axes=sharded),
                traced=traced,
            )
        with _timed(timings, "upload always", sync=device):
            self.always.update(from_torch(always, device=device, dtype=ttnn.float32, mesh_axes=sharded), traced=traced)
        with _timed(timings, "upload pad_q_threshold", sync=device):
            self.pad_q_threshold.update(
                from_torch(pad_q_threshold, device=device, dtype=ttnn.float32, mesh_axes=sharded), traced=traced
            )
        with _timed(timings, "upload cand_mask", sync=device):
            self.cand_mask.update(
                from_torch(
                    torch.where(cand_cols, 0.0, -float("inf")).reshape(1, 1, 1, -1),
                    device=device,
                    dtype=ttnn.float32,
                    mesh_axes=None,
                ),
                traced=traced,
            )
        with _timed(timings, "upload pad_mask", sync=device):
            # Head-uniform softmax mask with one identical row per tile row (attention_softmax takes height 1 or 32).
            self.pad_mask.update(
                from_torch(
                    torch.where(real_cols, 0.0, -float("inf")).reshape(1, 1, 1, -1).repeat(1, 1, TILE, 1),
                    device=device,
                    dtype=ttnn.bfloat16,
                    mesh_axes=None,
                ),
                traced=traced,
            )
        with _timed(timings, "upload block_counts", sync=device):
            self.block_counts.update(
                from_torch(
                    global_cols(geometry.valid_counts).to(torch.int32).reshape(1, 1, 1, -1),
                    device=device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_axes=None,
                ),
                traced=traced,
            )
        with _timed(timings, "upload kth_index", sync=device):
            self.kth_index.update(
                from_torch(
                    torch.full((1, self.num_heads, self.nb_pad, 1), k - 1, dtype=torch.int32),
                    device=device,
                    dtype=ttnn.uint32,
                    mesh_axes=None,
                ),
                traced=traced,
            )
        with _timed(timings, "upload prefix_marker", sync=device):
            self.prefix_marker.update(
                from_torch(prefix_marker, device=device, dtype=ttnn.bfloat16, mesh_axes=sharded), traced=traced
            )

    def _pool(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[1, H, rows, D] -> per-tile means: [1, H, nb_pad, D]."""
        weighted = ttnn.multiply(x, self.pool_weights.value)
        padded = ttnn.pad(weighted, [(0, 0), (0, 0), (0, self.nb_pad * VSA_TILE_TOKENS - self.rows), (0, 0)], 0.0)
        ttnn.deallocate(weighted)
        blocks = ttnn.experimental.view(padded, [self.num_heads, self.nb_pad, VSA_TILE_TOKENS, x.shape[-1]])
        pooled = ttnn.sum(blocks, dim=2, compute_kernel_config=self.hifi4)
        ttnn.deallocate(padded)
        return ttnn.reshape(pooled, [1, self.num_heads, self.nb_pad, x.shape[-1]])

    def _all_gather(self, x: ttnn.Tensor, dim: int) -> ttnn.Tensor:
        return self.ccl_manager.all_gather(x, dim=dim, mesh_axis=self.sp_axis, use_hyperparams=False)

    def __call__(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        *,
        coarse: bool = True,
        timings: dict[str, float] | None = None,
    ) -> dict:
        """`timings`, when given, receives each routing and coarse sub-section's host wall time in ms, device work
        included."""
        assert q.shape[2] == self.rows and q.shape[1] == self.num_heads, (q.shape, self.rows, self.num_heads)
        sync = self.mesh_device
        # Pooled per block: q_bar, k_bar and v_bar [1, H, nb_pad, D]; k_bar and v_bar gathered over SP to
        # [1, H, n_blocks_pad, D]. Only the coarse branch needs V.
        with _timed(timings, "vsa routing: pool q", sync=sync):
            q_bar = self._pool(q)
        with _timed(timings, "vsa routing: pool + gather k", sync=sync):
            k_bar = self._pool(k)
            k_bar_g = self._all_gather(k_bar, dim=2)
            ttnn.deallocate(k_bar)

        # Unscaled logits in fp32: top-k is invariant to the positive 1/sqrt(D) scale, and fp32 keeps near-ties apart.
        with _timed(timings, "vsa routing: scores", sync=sync):
            scores = ttnn.matmul(q_bar, k_bar_g, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
            ttnn.deallocate(q_bar)
            ttnn.deallocate(k_bar_g)

        with _timed(timings, "vsa routing: topk", sync=sync):
            masked = ttnn.add(scores, self.cand_mask.value)
            top_values, top_indices = ttnn.topk(masked, k=self.k_max, dim=-1)
            ttnn.deallocate(top_indices)
            kth_value = ttnn.gather(top_values, -1, index=self.kth_index.value)
            ttnn.deallocate(top_values)
        with _timed(timings, "vsa routing: threshold", sync=sync):
            threshold = ttnn.add(kth_value, self.pad_q_threshold.value)
            ttnn.deallocate(kth_value)
            kept = ttnn.ge(masked, threshold)
            ttnn.deallocate(masked)
            ttnn.deallocate(threshold)
            mask01 = ttnn.gtz(ttnn.add(kept, self.always.value))
            ttnn.deallocate(kept)
        with _timed(timings, "vsa routing: mask to uint32", sync=sync):
            block_mask = ttnn.to_layout(ttnn.typecast(mask01, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(mask01)

        coarse_out = None
        if coarse:
            with _timed(timings, "vsa coarse: pool + gather v", sync=sync):
                v_bar = self._pool(v)
                v_bar_g = self._all_gather(v_bar, dim=2)
                ttnn.deallocate(v_bar)
            with _timed(timings, "vsa coarse: softmax", sync=sync):
                probs = ttnn.transformer.attention_softmax_(
                    scores, head_size=self.head_dim, attention_mask=self.pad_mask.value
                )
                probs_bf16 = ttnn.typecast(probs, ttnn.bfloat16)
                ttnn.deallocate(probs)
            with _timed(timings, "vsa coarse: probs @ v", sync=sync):
                out_tiles = ttnn.matmul(probs_bf16, v_bar_g, compute_kernel_config=self.hifi4)
                ttnn.deallocate(probs_bf16)
                ttnn.deallocate(v_bar_g)
            with _timed(timings, "vsa coarse: broadcast", sync=sync):
                out_flat = ttnn.transformer.concatenate_heads(out_tiles)
                ttnn.deallocate(out_tiles)
                out_flat = ttnn.reshape(out_flat, [1, 1, out_flat.shape[-2], out_flat.shape[-1]])
                coarse_out = ttnn.matmul(self.bcast_flat, out_flat)
                ttnn.deallocate(out_flat)
        else:
            ttnn.deallocate(scores)

        return {
            "block_mask": block_mask,
            "block_counts": self.block_counts.value,
            "coarse_out": coarse_out,
        }

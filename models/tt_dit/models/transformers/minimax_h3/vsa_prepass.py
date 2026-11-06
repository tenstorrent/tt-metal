# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""VSA routing pre-pass for ``ttnn.transformer.sparse_ring_sdpa``.

One instance per ladder rung. Every buffer has a shape fixed by the rung, the SP factor and the local heads, and
``update`` writes a request's geometry into them in place, so the rung's trace replays for every request it serves.
The request's top-k ``k`` is data too: the selection is sized for the rung's widest ``k`` and reads the request's
``k`` from a device buffer.

Per device, from the local Q/K/V shards [1, H, rows, D] laid out by ``vsa_geometry`` (one 64-row block per tile,
``rows = rung / S``), the fused ``BlockTopKRouting`` op (``models/tt_dit/layers/block_topk_routing.py``) produces the
sparse op's mask and counts and the coarse branch output; this class derives its constants from the geometry:

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

from ....layers.block_topk_routing import BlockTopKRouting, RoutingRequest
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


def vsa_routing_request(geometry: MiniMaxH3VSAGeometry, nb_pad: int, k: int) -> RoutingRequest:
    """The geometry's routing rules in the padded per-shard numbering: prefix tiles are exempt, video tiles are
    candidates, and only the video query tiles are scored (prefix query rows are recomputed densely after the sparse
    op and pad rows are dropped at unpacking)."""
    sp, tiles = geometry.sp_factor, geometry.blocks_per_shard

    def per_shard(per_block: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(sp, nb_pad, dtype=per_block.dtype)
        out[:, :tiles] = per_block.reshape(sp, tiles)
        return out

    counts = per_shard(geometry.valid_counts)
    exempt = per_shard(geometry.is_exempt)
    return RoutingRequest(
        counts=counts,
        candidate=per_shard(geometry.is_candidate),
        exempt=exempt,
        scored=(counts > 0) & ~exempt,
        k=k,
    )


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
        self.sp_axis = sp_axis
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.rows = rung // sp_factor
        self.blocks_per_shard = -(-self.rows // VSA_TILE_TOKENS)
        self.nb_pad = _round_up(self.blocks_per_shard, TILE)
        self.n_blocks_pad = sp_factor * self.nb_pad
        # The widest top-k any geometry in the rung can need: every block a candidate.
        self.k_max = compute_topk(sparsity, sp_factor * self.blocks_per_shard)

        # The prefix query rows get their exact attention from the attention's prefix-row recompute, not from the
        # sparse op: these are its row count and its pad-row marker for this geometry.
        self.prefix_rows = 0
        self.prefix_marker = StateTensor()
        self.routing = BlockTopKRouting(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            sp_axis=sp_axis,
            num_heads=num_heads,
            head_dim=head_dim,
            rows=self.rows,
            k_max=self.k_max,
        )

    @cap_torch_threads_to_half_the_cpus()
    def update(self, geometry: MiniMaxH3VSAGeometry, *, traced: bool, timings: dict[str, float] | None = None) -> None:
        """Write one request's geometry into the rung's buffers; in place when `traced`, so a captured trace replays.

        `timings`, when given, receives each step's host wall time in ms.
        """
        sp = self.sp_factor
        assert geometry.padded_len == self.rung and geometry.sp_factor == sp, "the geometry must fill this rung"

        with _timed(timings, "host masks"):
            k = compute_topk(self.sparsity, int(geometry.is_candidate.sum()))
            self.prefix_rows = geometry.n_prefix_tiles * VSA_TILE_TOKENS
            block = geometry.block_of_row
            real_row = torch.arange(self.rung) - geometry.block_starts[block] < geometry.valid_counts[block]
            # -1 in the marker channel of every key row the prefix recompute must not attend: the pad rows of each
            # shard (inside partial blocks and whole pad blocks) and the recompute's extra rows, except its probe key.
            extra_rows = self.rows + PREFIX_EXTRA
            prefix_marker = torch.zeros(sp, 1, extra_rows, self.head_dim + PREFIX_EXTRA)
            prefix_marker[:, 0, : self.rows, self.head_dim] = -(~real_row).float().reshape(sp, self.rows)
            prefix_marker[:, 0, self.rows : extra_rows - 1, self.head_dim] = -1.0

        device = self.mesh_device
        sharded = [self.sp_axis, None, None, None]
        with _timed(timings, "upload prefix_marker", sync=device):
            self.prefix_marker.update(
                from_torch(prefix_marker, device=device, dtype=ttnn.bfloat16, mesh_axes=sharded), traced=traced
            )
        with _timed(timings, "upload routing constants", sync=device):
            self.routing.update(vsa_routing_request(geometry, self.nb_pad, k), traced=traced)

    def __call__(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        *,
        coarse: bool = True,
        timings: dict[str, float] | None = None,
    ) -> dict:
        """`timings`, when given, receives each routing sub-section's host wall time in ms, device work included."""
        assert q.shape[2] == self.rows and q.shape[1] == self.num_heads, (q.shape, self.rows, self.num_heads)
        sync = self.mesh_device
        return self.routing(
            q, k, v, coarse=coarse, section=lambda name: _timed(timings, f"vsa fused: {name}", sync=sync)
        )

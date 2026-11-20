# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview token embedding on the SC1 layout (tenstorrent/tt-metal#65892): the 120832 x 6144
table sharded by vocab over all 32 chips, 3776 token ids to a chip, read out in the hidden-split
layout the residual streams use.

Mirrors ../reference/lm_head.py Hy4Embedding; ``VocabShards`` there is the contract of which chip
owns a token id (chips counted row-major, the order ttnn.get_device_tensors returns them in).

How a chip that does not own a token contributes nothing. A chip's table is its 3776 rows with a
zero row before and a zero row after them. For a token id t, chip c looks up row

    clamp(t - c * 3776 + 1, 0, 3777)

which is t's own row on the chip that owns it, the leading zero row on every chip after it and the
trailing zero row on every chip before it. There is no mask and no data-dependent control flow:
all 32 chips run the same lookup on the same ids, and exactly one of them returns a non-zero row.
``table_rows`` does that arithmetic on device from token ids that are the same on every chip (the
form an on-device argmax or sampler hands back); ``host_table_rows`` does it on the host.

How the rows reach the hidden-split layout. After the lookup a token's row is whole, [T, 6144], on
one chip and zero on the other 31. Two all-reduces, over the head axis (mesh axis 0, 8 chips) and
then over the hidden axis (mesh axis 1, 4 chips), leave it whole on every chip; each chip then
keeps its own hidden slice (a chip-local partition):

    lookup              [1, 1, T, 6144] per chip, non-zero on the owner only
    all_reduce(head)    2 collectives of T x 6144 BF16 in total
    all_reduce(hidden)
    partition(hidden)   [1, 1, T, 1536] per chip, the same on the 8 chips that share a hidden slice

The sums add one real row and 31 zero rows, so they are exact in BF16 and the output is the
checkpoint's BF16 row bit for bit. For decode T is the number of users, so the two collectives
move 32 x 6144 values per step; against the several collectives of each of the 78 layers that is
noise. A table sharded 8 ways by vocab (over the head axis) and 4 ways by hidden (over the hidden
axis) would be the same 1/32 per chip and need one all-reduce of T x 1536; it is not what SC1
specifies for the embedding and is left as a possible optimisation.

The output is cast to ``out_dtype`` (fp32, what the iHC streams are held in) and is ready for
tt/ihc.py ``ihc_expand``.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hy4_preview.reference.lm_head import LMHeadConfig, VocabShards
from models.experimental.hy4_preview.tt.common import MeshPlan


def per_chip_mapper(plan: MeshPlan):
    """Mesh mapper for a host tensor [mesh axis 0, mesh axis 1, ...]: chip (r, c) gets entry [r, c],
    so a host tensor [chips, ...] reshaped to the mesh shape lands on the chips in row-major order."""
    if plan.hidden_axis is None:
        return None
    return plan.shard(hidden_dim=plan.hidden_axis, head_dim=plan.head_axis)


def per_chip_shape(plan: MeshPlan):
    """The two leading dims of a host tensor laid out for ``per_chip_mapper``."""
    return (1, 1) if plan.hidden_axis is None else plan.mesh_shape


class TtHy4Embedding(LightweightModule):
    """Token ids -> hidden states [1, 1, T, hidden/tp], split over the hidden axis.

    ``weight`` is the table [vocab, hidden]. ``dtype`` is the table's dtype on device (the
    checkpoint is BF16, so BF16 holds it exactly); ``out_dtype`` is the dtype of the result.
    """

    def __init__(
        self, plan: MeshPlan, cfg: LMHeadConfig, weight: torch.Tensor, dtype=ttnn.bfloat16, out_dtype=ttnn.float32
    ):
        super().__init__()
        self.plan, self.cfg = plan, cfg
        self.out_dtype = out_dtype
        self.chip_shape = per_chip_shape(plan)
        self.shards = shards = VocabShards(cfg.vocab_size, self.chip_shape[0] * self.chip_shape[1])
        assert tuple(weight.shape) == (cfg.vocab_size, cfg.hidden_size), tuple(weight.shape)
        assert cfg.hidden_size % plan.hidden_tp == 0, (cfg.hidden_size, plan.hidden_tp)

        mesh_device, mapper = plan.mesh_device, per_chip_mapper(plan)
        table = shards.split_table(weight.float())  # [chips, rows + 2, hidden]
        self.table = ttnn.from_torch(
            table.reshape(*self.chip_shape, shards.table_rows, cfg.hidden_size),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=dtype,
            mesh_mapper=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Per chip, the token id of its leading zero row: its first token id - 1.
        self.row_origin = ttnn.from_torch(
            (shards.offsets() - 1).to(torch.int32).reshape(*self.chip_shape, 1, 1),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=mapper,
        )

    # ------------------------------------------------------------------------------------------
    def host_table_rows(self, token_ids: torch.Tensor):
        """Host token ids [T] -> device uint32 [1, T] per chip: the row each chip looks up."""
        T = token_ids.numel()
        rows = self.shards.table_rows_for(token_ids.flatten().long())  # [chips, T]
        rows = ttnn.from_torch(
            rows.to(torch.int32).reshape(*self.chip_shape, 1, T),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.plan.mesh_device,
            dtype=ttnn.uint32,
            mesh_mapper=per_chip_mapper(self.plan),
        )
        return ttnn.reshape(rows, [1, T])

    def host_token_ids(self, token_ids: torch.Tensor):
        """Host token ids [T] -> device uint32 [1, 1, 1, T], the same on every chip: the input of ``table_rows``."""
        in_vocab = (token_ids >= 0) & (token_ids < self.cfg.vocab_size)
        assert (
            in_vocab.all()
        ), f"token ids outside [0, {self.cfg.vocab_size}): {token_ids[~in_vocab].flatten()[:8].tolist()}"
        return ttnn.from_torch(
            token_ids.flatten().to(torch.int32).reshape(1, 1, 1, -1),
            layout=ttnn.TILE_LAYOUT,
            device=self.plan.mesh_device,
            dtype=ttnn.uint32,
        )

    def table_rows(self, token_ids):
        """Device token ids, uint32 [1, 1, 1, T] and the same on every chip -> uint32 [1, T] per chip.

        Signed arithmetic: the difference is negative on every chip after the owner. Nothing on
        device can refuse an id outside the vocab: it clamps to a zero row on every chip and the
        embedding comes back as zeros (the host paths, ``host_table_rows`` and ``host_token_ids``,
        refuse it).
        """
        T = token_ids.shape[-1]
        if token_ids.layout != ttnn.TILE_LAYOUT:  # ttnn.sampling returns ROW_MAJOR
            token_ids = ttnn.to_layout(token_ids, ttnn.TILE_LAYOUT)
        rows = ttnn.subtract(ttnn.typecast(token_ids, ttnn.int32), self.row_origin)
        rows = ttnn.clamp(rows, min=0, max=self.shards.rows_per_chip + 1)
        rows = ttnn.typecast(rows, ttnn.uint32)
        return ttnn.reshape(ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT), [1, T])

    def lookup(self, rows):
        """uint32 [1, T] per chip -> [1, 1, T, hidden] per chip: the token's row on its owner, zeros elsewhere."""
        T = rows.shape[-1]
        out = ttnn.embedding(rows, self.table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return ttnn.reshape(out, [1, 1, T, self.cfg.hidden_size])

    def spread(self, x):
        """[1, 1, T, hidden], non-zero on one chip -> [1, 1, T, hidden/tp] split over the hidden axis."""
        plan = self.plan
        if plan.hidden_axis is None:
            return x
        whole = plan.all_reduce(plan.all_reduce(x, plan.head_axis), plan.hidden_axis)
        return plan.partition(whole, dim=3, axis=plan.hidden_axis)

    def forward(self, rows):
        """``rows`` as ``table_rows`` or ``host_table_rows`` returns them -> [1, 1, T, hidden/tp] in ``out_dtype``."""
        x = self.spread(self.lookup(rows))
        return x if x.dtype == self.out_dtype else ttnn.typecast(x, self.out_dtype)

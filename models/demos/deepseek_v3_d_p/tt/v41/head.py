# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 token embedding and output head (bead F1).

Reference (``inference/model.py``): ``ParallelEmbedding`` (plain row lookup; the reference shards the
vocabulary over ranks, which does not change the values), and after the last block
``head(norm(hc_pre(h, pre_mix)))``: ``RMSNorm(dim, norm_eps)`` then ``ParallelHead``, whose weight is the
bf16 checkpoint tensor held in fp32 and which returns fp32 logits of the **last position only**.

Device mapping:

* :class:`TtV41Embedding` -- the in-tree ``TtParallelEmbedding`` (bf16 table, hidden dim sharded over TP,
  replicated over SP): its output is already the block layout ``[1, 1, S/sp, hidden/tp]``, so no
  collective is needed; the lookup is exact.
* :class:`TtV41Head` -- takes the collapsed final hidden ``[1, 1, S/sp, hidden/tp]`` (F8 owns the collapse),
  narrows it to the tile holding the requested row, applies the distributed RMSNorm (eps from the config)
  and the in-tree ``TtLMHead`` column-parallel over the vocabulary (the reference's split): bf16 weights
  (exact), fp32 activations, HiFi4 with fp32 accumulation, fp32 logits.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.utils import global_to_local_token_id
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding

TP_AXIS, SP_AXIS = 1, 0

HEAD_COMPUTE_CONFIG = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


class TtV41Embedding(LightweightModule):
    def __init__(self, mesh_device, config, weight: torch.Tensor):
        """``weight``: the bf16 ``embed.weight`` ``[vocab, hidden]``."""
        self.embedding = TtParallelEmbedding(
            mesh_device,
            vocab_size=config.VOCAB_SIZE,
            emb_dim=config.EMB_SIZE,
            torch_weight=weight,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
            dtype=ttnn.bfloat16,
        )

    def forward(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """token_ids ``[1, 1, S/sp]`` uint32 (SP-sharded, replicated over TP) -> ``[1, 1, S/sp, hidden/tp]`` bf16."""
        out = self.embedding(token_ids)  # [1, S/sp, hidden/tp]
        return ttnn.reshape(out, (1, 1, out.shape[-2], out.shape[-1]))


class TtV41Head(LightweightModule):
    def __init__(
        self,
        mesh_device,
        config,
        norm_weight: torch.Tensor,
        head_weight: torch.Tensor,
        num_links: int = 1,
    ):
        """``norm_weight``: ``norm.weight`` ``[hidden]``; ``head_weight``: bf16 ``head.weight`` ``[vocab, hidden]``."""
        self.mesh_device = mesh_device
        self.sp = mesh_device.shape[SP_AXIS]
        self.tp = mesh_device.shape[TP_AXIS]
        topology = per_axis_topology()[TP_AXIS]  # both collectives run on the TP axis
        self.norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=config.EMB_SIZE,
            torch_weight=norm_weight,
            epsilon=config.RMS_NORM_EPS,
            cluster_axis=TP_AXIS,
            num_links=num_links,
            topology=topology,
        )
        self.head = TtLMHead(
            mesh_device,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
            emb_dim=config.EMB_SIZE,
            vocab_size=config.VOCAB_SIZE,
            torch_weight=head_weight,
            num_links=num_links,
            topology=topology,
            weights_dtype=ttnn.bfloat16,
            compute_kernel_config=HEAD_COMPUTE_CONFIG,
            is_balanced=False,
            is_column_parallel=True,
        )

    def forward(self, h: ttnn.Tensor, row: int) -> tuple[ttnn.Tensor, tuple[int, int]]:
        """h ``[1, 1, S/sp, hidden/tp]`` (sequence contiguous over SP), ``row``: the row to project (the last
        real token) -> (logits ``[1, 1, 32, vocab/tp]`` fp32 per chip, (sp_row, offset)): the requested
        row's logits sit at ``offset`` of the tile on mesh row ``sp_row``; see :meth:`logits_to_host`."""
        rows_per_chip = h.shape[-2]
        sp_row, local = global_to_local_token_id(row, self.sp, rows_per_chip * self.sp, is_balanced=False)
        tile_start = local // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        x = ttnn.narrow(h, dim=-2, start=tile_start, length=ttnn.TILE_SIZE)
        x = ttnn.typecast(self.norm(x), ttnn.float32)
        # x now holds one tile per chip; address the row inside it (sequential SP layout of 32-row shards)
        logits, _ = self.head(x, sp_row * ttnn.TILE_SIZE + local - tile_start)
        return logits, (sp_row, local - tile_start)

    def logits_to_host(self, logits: ttnn.Tensor, position: tuple[int, int]) -> torch.Tensor:
        """The requested row's fp32 logits ``[vocab]``: vocab shards of mesh row ``sp_row`` concatenated."""
        sp_row, offset = position
        return self._tile_to_host(logits, sp_row)[offset]

    def rows_to_host(self, h: ttnn.Tensor, rows: list[int]) -> torch.Tensor:
        """fp32 logits ``[len(rows), vocab]`` of rows ``rows`` of ``h`` (as :meth:`forward`), one projection per
        distinct 32-row tile."""
        found = {}
        for row in rows:
            if row in found:
                continue
            logits, (sp_row, offset) = self(h, row)
            tile, first = self._tile_to_host(logits, sp_row), row - offset  # a tile never spans chips
            found |= {r: tile[r - first] for r in rows if first <= r < first + ttnn.TILE_SIZE}
        return torch.stack([found[r] for r in rows])

    def _tile_to_host(self, logits: ttnn.Tensor, sp_row: int) -> torch.Tensor:
        """The fp32 logits ``[32, vocab]`` of mesh row ``sp_row``'s tile: its vocab shards concatenated."""
        shards = ttnn.get_device_tensors(logits)
        pieces = [ttnn.to_torch(shards[sp_row * self.tp + j]) for j in range(self.tp)]
        return torch.cat(pieces, dim=-1).reshape(-1, pieces[0].shape[-1] * self.tp).float()

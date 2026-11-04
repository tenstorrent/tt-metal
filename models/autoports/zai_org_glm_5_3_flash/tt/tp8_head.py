# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14 M5: the TP8 model head — lm_head as a vocab-column shard per chip, streamed by the DS readers as a
megakernel reader phase (7a's ReaderPhases / MMPhase, the same weight-streaming kernels as the layer's matmuls).

Shard contract (shard_contract.md "lm_head"): the 00c cache's lm_head.weight [4096, 154,880] bfp8 TILE; chip r = the
tile-aligned vocab columns [W r, W (r + 1)), W = 154,880 / tp (TP8: 19,360 = 605 tiles; tp_padded_vocab adds nothing:
4,840 tiles divide by 8) -> lossless slices of the bfp8 tiles. A chip's shard is `splits` DS matmul phases of W / splits
columns (one reader program; TP8: 1 phase, the DS layout pads it to 24 readers x 26 tiles = 19,968 columns, zero).
Logit column c of chip r, chunk j is vocab id W r + (W / splits) j + c.
"""

from __future__ import annotations

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import VOCAB_SIZE, tp_padded_vocab

DRAM = ttnn.DRAM_MEMORY_CONFIG
HIDDEN = 4096


class TPHead:
    """lm_head on every chip of `mesh` (TP over the vocab). __call__(h [1,1,B,4096] bf16, replicated) -> the chip's
    logits chunks [[1,1,B,>= W / splits] bf16] (columns past W / splits are DS padding)."""

    def __init__(self, mesh, source, ocfg, *, kb: int = 8, splits: int | None = None):
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import MMPhase, ReaderPhases
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import DSLinear
        from models.autoports.zai_org_glm_5_3_flash.tt.precision import hifi2_fp32

        self.mesh = mesh
        self.tp = int(mesh.get_num_devices())
        self.vpad = tp_padded_vocab(VOCAB_SIZE, self.tp)
        self.w = self.vpad // self.tp
        e = source.tensors["lm_head.weight"]
        full = ttnn.to_torch(source._load(e)).reshape(e["shape"])  # [4096, V] (the bfp8 values, exactly)
        self.host_w = full
        full = torch.nn.functional.pad(full, (0, self.vpad - VOCAB_SIZE))
        self.splits = splits or max(1, 8 // self.tp)  # <= 605 tiles per DS phase (L1 of the readers' output CB)
        assert self.w % (32 * self.splits) == 0, (self.w, self.splits)
        self.cw = self.w // self.splits
        ocfg.ds_readers_per_role = ";".join(filter(None, [ocfg.ds_readers_per_role, "lm_head:3"]))
        self.ds = []
        for j in range(self.splits):
            cols = [full[:, self.w * r + self.cw * j : self.w * r + self.cw * (j + 1)] for r in range(self.tp)]
            shard = ttnn.from_torch(torch.cat(cols, dim=-1), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                                    memory_config=DRAM, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))  # fmt: skip
            self.ds.append(DSLinear(mesh, shard, role="lm_head", cfg=ocfg, ckc=hifi2_fp32(mesh), dtype=ttnn.bfloat16))
        del full, cols
        self.readers = ReaderPhases(mesh, [MMPhase(ds=d, kb=kb) for d in self.ds])

    def __call__(self, h):
        return self.readers([h] * self.splits)

    def gather(self, chunks) -> torch.Tensor:
        """Host: every chip's chunks -> [B, 154,880] logits in vocab order (DS padding dropped)."""
        per = [[ttnn.to_torch(t).float().reshape(-1, t.shape[-1])[:, : self.cw] for t in ttnn.get_device_tensors(c)]
               for c in chunks]  # fmt: skip
        return torch.cat([per[j][r] for r in range(self.tp) for j in range(self.splits)], dim=-1)[:, :VOCAB_SIZE]

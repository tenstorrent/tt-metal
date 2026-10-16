# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode-frame routed experts on the decode MoE kernel (all_to_all_dispatch_metadata -> moe_compute -> fused
score-weighted reduce -> reduce-scatter), models/common/modules/moe/TTMoEDecode with the gemma_4_26b config.

Layout: the frame's rows (all lanes, gathered on every chip) are split over the mesh rows (the dispatch axis,
cluster_axis 0); every column holds 32 experts (4 per chip on the Galaxy, 16 on a (2,4) block), the same placement
the sparse path uses. Each column computes its experts' contributions for every token in its row-chip's slice, the
reduce-scatter over the columns sums them, and two all-gathers restore the replicated [frame, hidden] tensor.
Expert weights are the kernel's bfloat4_b DRAM-sharded layout (a second copy next to the sparse path's).

Enabled with GEMMA4_MOE_DECODE_KERNEL=1 (see experts/__init__.py); GEMMA4_MOE_DECODE_FRAME sets the frame rows.
"""
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.modules.moe.tt_moe_decode import TTMoEDecode
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig

CONFIG_YAML = Path(__file__).resolve().parents[4] / "common" / "modules" / "moe" / "configs" / "gemma_4_26b.yaml"


def moe_decode_kernel_enabled():
    return os.environ.get("GEMMA4_MOE_DECODE_KERNEL", "0") == "1"


class Gemma4MoEDecode:
    def __init__(self, mesh_device, config, gate_up_proj, down_proj, frame_rows=None):
        """gate_up_proj [E, 2I, H] (gate rows first), down_proj [E, H, I]: HF tensors of this layer."""
        import yaml

        self.mesh_device = mesh_device
        rows, cols = mesh_device.shape
        self.rows, self.cols = rows, cols
        self.hidden = config.hidden_size
        self.top_k = config.top_k
        self.num_experts = config.num_experts
        self.frame_rows = int(frame_rows or os.environ.get("GEMMA4_MOE_DECODE_FRAME", "128"))
        assert self.frame_rows % rows == 0, f"frame {self.frame_rows} must split over {rows} mesh rows"
        self.batch_per_device = self.frame_rows // rows
        data = yaml.safe_load(CONFIG_YAML.read_text())
        data["mesh_shape"] = [rows, cols]
        data["batch_per_device"] = self.batch_per_device
        data["hidden_size"] = self.hidden
        data["select_experts_k"] = self.top_k
        data["num_routed_experts"] = self.num_experts
        data["compute"]["intermediate_size"] = config.moe_intermediate_size
        self.tt_config = TTMoEDecodeConfig.from_yaml(
            yaml.safe_dump(data, sort_keys=False), topology=ttnn.Topology.Linear
        )
        I = config.moe_intermediate_size
        # kernel layout: w0 = gate [L, E, H, N], w1 = up [L, E, H, N], w2 = down [L, E, N, H]
        gate = gate_up_proj[:, :I, :].to(torch.bfloat16).transpose(1, 2).contiguous().unsqueeze(0)
        up = gate_up_proj[:, I:, :].to(torch.bfloat16).transpose(1, 2).contiguous().unsqueeze(0)
        down = down_proj.to(torch.bfloat16).transpose(1, 2).contiguous().unsqueeze(0)
        self.module = TTMoEDecode(mesh_device, self.tt_config, gate, up, down)
        logger.info(
            f"Gemma4MoEDecode: mesh {rows}x{cols}, frame {self.frame_rows} rows ({self.batch_per_device}/chip), "
            f"post_combine_tilize={getattr(self.tt_config, 'use_post_combine_tilize', None)}"
        )

    def forward(self, hidden, routing):
        """hidden [1,1,R,H] and routing [1,1,R,E] replicated on every chip, R <= frame_rows (padded here).
        Returns [1,1,R,H] replicated: the top-k weighted expert sum."""
        R = int(hidden.shape[2])
        assert R <= self.frame_rows, f"{R} rows > frame {self.frame_rows}"
        if R < self.frame_rows:
            pad = self.frame_rows - R
            hidden = ttnn.pad(hidden, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
            routing = ttnn.pad(routing, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
        x = ttnn.mesh_partition(hidden, dim=2, cluster_axis=0)  # [1,1,B,H] per row-chip
        r = ttnn.mesh_partition(routing, dim=2, cluster_axis=0)  # [1,1,B,E]
        scores, indices = ttnn.topk(r, k=self.top_k, dim=-1)  # [1,1,B,K] bf16 / uint16 (TILE)
        ttnn.deallocate(r)
        B = self.batch_per_device
        x = ttnn.to_layout(ttnn.reshape(x, (B, 1, 1, self.hidden)), ttnn.ROW_MAJOR_LAYOUT)
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        scores = ttnn.to_layout(ttnn.reshape(scores, (B, 1, 1, self.top_k)), ttnn.ROW_MAJOR_LAYOUT)
        indices = ttnn.to_layout(ttnn.reshape(indices, (B, 1, 1, self.top_k)), ttnn.ROW_MAJOR_LAYOUT)
        out = self.module.forward(x, scores, indices)  # [1,1,B,H/cols] per chip
        ttnn.deallocate(x)
        ttnn.deallocate(scores)
        ttnn.deallocate(indices)
        out = ttnn.reshape(out, (1, 1, B, self.hidden // self.cols))
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        if self.cols > 1:
            full = ttnn.all_gather(out, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)  # [1,1,B,H]
            ttnn.deallocate(out)
        else:
            full = out
        if self.rows > 1:
            frame = ttnn.all_gather(full, dim=2, cluster_axis=0, topology=ttnn.Topology.Linear)  # [1,1,frame,H]
            ttnn.deallocate(full)
        else:
            frame = full
        if R < self.frame_rows:
            frame = ttnn.slice(frame, [0, 0, 0, 0], [1, 1, R, self.hidden])
        return frame

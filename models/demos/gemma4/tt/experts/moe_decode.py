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


# One TTMoEDecode per mesh, shared by every layer: the module keeps per-instance L1 scratch and semaphores (a
# per-layer instance x 30 left ~200 KB of resident L1 and its static circular buffers then clashed), and its weights
# are layered ([L, E, ...], forward(layer_id)). Layers register their HF tensors as they are built; the module is
# built when the last one arrives.
_SHARED = {}


class Gemma4MoEDecode:
    def __init__(self, mesh_device, config, gate_up_proj, down_proj, layer_idx=0, num_layers=1, frame_rows=None):
        """gate_up_proj [E, 2I, H] (gate rows first), down_proj [E, H, I]: HF tensors of this layer."""
        self.mesh_device = mesh_device
        rows, cols = mesh_device.shape
        self.rows, self.cols = rows, cols
        self.hidden = config.hidden_size
        self.top_k = config.top_k
        self.num_experts = config.num_experts
        self.inter = config.moe_intermediate_size
        self.layer_idx = int(layer_idx)
        self.num_layers = int(num_layers)
        self.frame_rows = int(frame_rows or os.environ.get("GEMMA4_MOE_DECODE_FRAME", str(rows * ttnn.TILE_SIZE)))
        assert self.frame_rows % rows == 0, f"frame {self.frame_rows} must split over {rows} mesh rows"
        self.batch_per_device = self.frame_rows // rows
        key = id(mesh_device)
        reg = _SHARED.setdefault(key, {"layers": {}, "module": None, "config": None})
        reg["layers"][self.layer_idx] = (gate_up_proj, down_proj)
        self._reg = reg
        if reg["module"] is None and len(reg["layers"]) >= self.num_layers:
            self._build(config)

    def _build(self, config):
        import yaml

        reg = self._reg
        data = yaml.safe_load(CONFIG_YAML.read_text())
        data["mesh_shape"] = [self.rows, self.cols]
        data["batch_per_device"] = self.batch_per_device
        data["hidden_size"] = self.hidden
        data["select_experts_k"] = self.top_k
        data["num_routed_experts"] = self.num_experts
        data["compute"]["intermediate_size"] = self.inter
        # the module's dispatch/compute ops require the ring fabric and Ring topology
        tt_config = TTMoEDecodeConfig.from_yaml(yaml.safe_dump(data, sort_keys=False), topology=ttnn.Topology.Ring)
        I = self.inter
        layers = [reg["layers"][l] for l in range(self.num_layers)]
        # kernel layout: w0 = gate [L, E, H, N], w1 = up [L, E, H, N], w2 = down [L, E, N, H]
        gate = torch.stack([gu[:, :I, :].to(torch.bfloat16).transpose(1, 2).contiguous() for gu, _ in layers], dim=0)
        up = torch.stack([gu[:, I:, :].to(torch.bfloat16).transpose(1, 2).contiguous() for gu, _ in layers], dim=0)
        down = torch.stack([d.to(torch.bfloat16).transpose(1, 2).contiguous() for _, d in layers], dim=0)
        reg["module"] = TTMoEDecode(self.mesh_device, tt_config, gate, up, down)
        reg["config"] = tt_config
        reg["layers"].clear()
        logger.info(
            f"Gemma4MoEDecode: shared module for {self.num_layers} layers on mesh {self.rows}x{self.cols}, frame "
            f"{self.frame_rows} rows ({self.batch_per_device}/chip), post_combine_tilize="
            f"{getattr(tt_config, 'use_post_combine_tilize', None)}"
        )

    @property
    def module(self):
        assert self._reg["module"] is not None, "decode MoE module not built: not every layer registered"
        return self._reg["module"]

    def forward(self, hidden, routing):
        """hidden [1,1,R,H] and routing [1,1,R,E] replicated on every chip, R <= frame_rows (padded here).
        Returns [1,1,R,H] replicated: the top-k weighted expert sum."""
        R = int(hidden.shape[2])
        assert R <= self.frame_rows, f"{R} rows > frame {self.frame_rows}"
        if R < self.frame_rows:
            pad = self.frame_rows - R
            hidden = ttnn.pad(hidden, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
            routing = ttnn.pad(routing, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
        B = self.batch_per_device
        # Row-major before the per-row split: a slice of B < 32 rows is not tile-aligned, reshapes of uint16 tensors
        # are only views in row-major, and the gathers below must not carry tile padding between chips.
        hidden_rm = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
        routing_rm = ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.mesh_partition(hidden_rm, dim=2, cluster_axis=0)  # [1,1,B,H] per row-chip
        r = ttnn.mesh_partition(routing_rm, dim=2, cluster_axis=0)  # [1,1,B,E]
        ttnn.deallocate(hidden_rm)
        ttnn.deallocate(routing_rm)
        r = ttnn.to_layout(r, ttnn.TILE_LAYOUT)
        scores, indices = ttnn.topk(r, k=self.top_k, dim=-1)  # [1,1,B,K] bf16 / uint16 (TILE)
        ttnn.deallocate(r)
        scores = ttnn.reshape(ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT), (B, 1, 1, self.top_k))
        indices = ttnn.reshape(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), (B, 1, 1, self.top_k))
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        x = ttnn.reshape(x, (B, 1, 1, self.hidden))
        if os.environ.get("GEMMA4_MOE_DECODE_L1_DUMP") and not getattr(self, "_dumped", False):
            # which L1 buffers are resident when the module's static circular buffers are laid out
            self._dumped = True
            prefix = os.environ["GEMMA4_MOE_DECODE_L1_DUMP"].rstrip("/") + "/moe_decode_pre_"
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
            ttnn.dump_device_memory_state(self.mesh_device, prefix)
            logger.info(f"Gemma4MoEDecode: device memory state dumped to {prefix}*")
        out = self.module.forward(x, scores, indices, layer_id=self.layer_idx)  # [1,1,B,H/cols] per chip
        ttnn.deallocate(x)
        ttnn.deallocate(scores)
        ttnn.deallocate(indices)
        if out.layout != ttnn.ROW_MAJOR_LAYOUT:
            out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)  # strips any tile padding of a B < 32 slice
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (1, 1, B, self.hidden // self.cols))
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
        frame = ttnn.to_layout(frame, ttnn.TILE_LAYOUT)
        if R < self.frame_rows:
            frame = ttnn.slice(frame, [0, 0, 0, 0], [1, 1, R, self.hidden])
        return frame

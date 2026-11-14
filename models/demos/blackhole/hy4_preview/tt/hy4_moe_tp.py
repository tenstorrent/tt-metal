# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe-perf experiment: routed experts as PSEUDO-EXPERTS, expert-parallel over the mesh COLUMNS x tensor-parallel over mesh ROWS (default OFF, HY4_MOE_TP=2|4).

Why: moe_compute follows the BUSIEST chip. With 32 activations (B=4 x top-8) on 32 chips the busiest chip holds ~3.2 distinct experts (mean 0.92); every expert is a 21 MB bfp4 read.
Cut every expert along its intermediate dim into TP slices of 2048/TP columns (gate/up columns, down rows) and put slice s of expert e on mesh row s (TP=4; TP=2: row 2g+s of the row pair g = (e % 32) // 16)
of the expert's own mesh column (e // 32): per-chip weight bytes are unchanged (32 experts x 1/4 slice), but an active expert now costs 1/TP of its time on each of TP chips, so the busiest chip
holds ~1.5 expert-equivalents (TP=4) instead of 3.2. The block is the stock ``Hy4MoEBlock`` with select_experts_k = 8 x TP "pseudo selections" (pseudo id p = e * TP + s, same score for every slice),
num_routed_experts = 256 x TP, intermediate 2048 / TP, an explicit expert -> chip mapping; the tail sums the slice partials like any other slot.
"""
import os

import ttnn
from models.common.moe.tt_moe_decode_config import TTMoEDecodeConfig
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import clamp_text
from models.demos.blackhole.hy4_preview.tt.hy4_moe import CONFIG_PATH, Hy4MoEBlock

NE, ROWS, COLS = 256, 4, 8


def tp() -> int:
    return int(os.environ.get("HY4_MOE_TP", "0") or 0)


def pseudo_chip(e: int, s: int, TP: int) -> int:
    """linear chip id (row * 8 + col) of slice s of expert e."""
    c = e // 32
    r = s if TP == ROWS else 2 * ((e % 32) // 16) + s
    return r * COLS + c


def mapping(TP: int):
    return [pseudo_chip(e, s, TP) for e in range(NE) for s in range(TP)]


def slice_weights(W: dict, TP: int) -> dict:
    """host dict of Hy4 expert weights ([1,E,6144,2048] x2, [1,E,2048,6144]) -> pseudo experts [1,E*TP,6144,2048/TP] / [1,E*TP,2048/TP,6144], p = e*TP + s."""
    w0, w1, w2 = W["w0"], W["w1"], W["w2"]
    H, N = w0.shape[2], w0.shape[3]
    n = N // TP
    f = lambda w: w.reshape(1, NE, H, TP, n).permute(0, 1, 3, 2, 4).reshape(1, NE * TP, H, n).contiguous()
    g = lambda w: w.reshape(1, NE, TP, n, H).reshape(1, NE * TP, n, H).contiguous()
    out = dict(W)
    out["w0"], out["w1"], out["w2"] = f(w0), f(w1), g(w2)
    return out


def pseudo_routing(w, idx, TP: int):
    """torch (w [T,8], idx [T,8] int) -> pseudo (w [T,8*TP], idx [T,8*TP]): slot j*TP + s = (expert idx[j], slice s)."""
    import torch

    pid = (idx.long().unsqueeze(-1) * TP + torch.arange(TP)).reshape(idx.shape[0], -1)
    pw = w.unsqueeze(-1).expand(-1, -1, TP).reshape(w.shape[0], -1)
    return pw, pid


class Hy4MoETPBlock(Hy4MoEBlock):
    @staticmethod
    def _configs(mesh_device, batch_per_device, topology, shared_in_moe):
        TP = tp()
        text = CONFIG_PATH.read_text()
        text = text.replace("batch_per_device: 1 ", f"batch_per_device: {batch_per_device} ", 1)
        text = text.replace("select_experts_k: 8", f"select_experts_k: {8 * TP}", 1)
        text = text.replace("num_routed_experts: 256", f"num_routed_experts: {NE * TP}", 1)
        text = text.replace("intermediate_size: 2048", f"intermediate_size: {2048 // TP}", 1)
        text = text.replace(
            "  expert_mapping: sequential", "  expert_mapping: [" + ", ".join(str(c) for c in mapping(TP)) + "]", 1
        )
        text = clamp_text(text, "decode")
        text = text.replace("num_shared_experts: 1", "num_shared_experts: 0").replace(
            "  shared_expert_ids_to_devices: fully_replicated\n", ""
        )
        cfg = TTMoEDecodeConfig.from_yaml(text, topology=topology)
        if cfg.mesh_shape != tuple(mesh_device.shape):
            cfg = cfg.with_mesh_shape(tuple(mesh_device.shape))
        if cfg.num_fast_reduce_outputs == 1:
            cfg = cfg.model_copy(
                update={"reduce": cfg.reduce.model_copy(update={"output_memory_config": ttnn.DRAM_MEMORY_CONFIG})}
            )
        return cfg, None

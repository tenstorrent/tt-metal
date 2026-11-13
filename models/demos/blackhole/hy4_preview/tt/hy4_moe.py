# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview MoE block (decode): Hy4Gate router + 256 routed experts (8 per chip, bfp4, moe_compute) [+ shared expert partial folded in].

A subclass of the DSV4 ``DSV41MoEBlock`` (reused unchanged: _TailDecode, moe_tail, warmup, for_batch, forward): only the config (hy4_preview.yaml:
hidden 6144, 256 experts, k 8, inter 2048, limit 10) and the gate (sigmoid, 256, k 8) differ.
Placement: ``tt/hy4_moe_place.py`` (sequential: expert e on mesh column e // 32, row (e % 32) // 8).
"""
from pathlib import Path

import ttnn
from models.common.moe.tt_moe_decode_config import TTMoEDecodeConfig
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock, clamp_text
from models.demos.blackhole.hy4_preview.tt.hy4_router import Hy4Gate

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "hy4_preview.yaml"


class Hy4MoEBlock(DSV41MoEBlock):
    def __init__(
        self,
        mesh_device,
        weights: dict,
        topology=ttnn.Topology.Linear,
        batch_per_device: int = 1,
        shared_in_moe: bool = False,
        buffers=None,
        expert_state=None,
        router_mode=None,
    ):
        self._topology, self._shared_in_moe = topology, shared_in_moe
        decode_cfg = self._configs(mesh_device, batch_per_device, topology, shared_in_moe)[0]
        self.mesh_device = mesh_device
        self.decode_config = decode_cfg
        self.gate = Hy4Gate(mesh_device, weights["gate_weight"], weights["gate_bias"], mode=router_mode)
        if expert_state is not None:
            self.decode = self._weightless_decode(mesh_device, decode_cfg, buffers, expert_state=expert_state)
        else:
            self._build_decode(mesh_device, decode_cfg, weights, shared_in_moe, buffers)
        self.decode._mesh = mesh_device

    @staticmethod
    def _configs(mesh_device, batch_per_device, topology, shared_in_moe):
        text = CONFIG_PATH.read_text()
        text = text.replace("batch_per_device: 1 ", f"batch_per_device: {batch_per_device} ", 1)
        text = clamp_text(text, "decode")
        if not shared_in_moe:
            text = text.replace("num_shared_experts: 1", "num_shared_experts: 0").replace(
                "  shared_expert_ids_to_devices: fully_replicated\n", ""
            )
        mesh_shape = tuple(mesh_device.shape)
        decode_cfg = TTMoEDecodeConfig.from_yaml(text, topology=topology)
        if decode_cfg.mesh_shape != mesh_shape:
            decode_cfg = decode_cfg.with_mesh_shape(mesh_shape)
        if decode_cfg.batch_per_device != batch_per_device:
            decode_cfg = decode_cfg.model_copy(update={"batch_per_device": batch_per_device})
        if decode_cfg.num_fast_reduce_outputs == 1:
            decode_cfg = decode_cfg.model_copy(
                update={
                    "reduce": decode_cfg.reduce.model_copy(update={"output_memory_config": ttnn.DRAM_MEMORY_CONFIG})
                }
            )
        return decode_cfg, None

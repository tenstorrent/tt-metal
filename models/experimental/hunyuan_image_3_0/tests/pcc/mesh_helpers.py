# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for 2×2 mesh SP/TP/EP integration tests.

from __future__ import annotations

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.ref.weights import load_prefixed_state_dict, resolve_base_model_dir
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.tt_dit.parallel.manager import CCLManager
from pcc_common import transformer_cfg

MESH_NL = int(__import__("os").environ.get("HY_NUM_LAYERS", "2"))


def mesh_layer_loader(i: int) -> dict[str, torch.Tensor]:
    sd = load_prefixed_state_dict(resolve_base_model_dir(), f"model.layers.{i}.")
    prefix = f"model.layers.{i}."
    return {f"{prefix}{k}": v for k, v in sd.items()}


def mesh_ccl(mesh_device) -> CCLManager:
    return CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)


def replicate_to_mesh(mesh_device, t: torch.Tensor, *, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def build_mesh_model(
    mesh_device, ccl, *, sp_factor: int = 1, tp_factor: int = 1, num_layers: int = MESH_NL
) -> HunyuanTtModel:
    c = transformer_cfg()
    kwargs = dict(
        num_layers=num_layers,
        hidden_size=c["H"],
        num_heads=c["HEADS"],
        num_kv_heads=c["KV"],
        head_dim=c["HD"],
        num_experts=c["E"],
        moe_topk=c["K"],
        use_qk_norm=c["QKN"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        rms_norm_eps=c["EPS"],
        weight_dtype=ttnn.bfloat16,
        stream_experts=False,
        layer_loader=mesh_layer_loader,
        apply_final_norm=False,
        ccl_manager=ccl,
        sp_axis=0,
        sp_factor=sp_factor,
    )
    if tp_factor > 1:
        kwargs.update(tp_axis=1, tp_factor=tp_factor)
    else:
        kwargs["tp_factor"] = 1
    return HunyuanTtModel(mesh_device, **kwargs)


def causal_mask(seq_len: int) -> torch.Tensor:
    return torch.triu(torch.full((seq_len, seq_len), -1.0e30), diagonal=1).reshape(1, 1, seq_len, seq_len)


def run_embeds_forward(mesh_device, model, x: torch.Tensor, seq_len: int, mask: torch.Tensor) -> torch.Tensor:
    out = model.forward(
        inputs_embeds=replicate_to_mesh(mesh_device, x),
        seq_len=seq_len,
        attention_mask=replicate_to_mesh(mesh_device, mask),
    )
    return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[: x.shape[0]].float()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TtV4PrefillBlock (mHC + {SWA|CSA|HCA} + V4 MoE) vs the reference DeepseekV4DecoderLayer, one layer per kind,
random weights, reduced expert count (64 -> 8 per chip on 2x4, the galaxy's per-chip load). Output streams PCC >=
0.99. The reference applies swiglu_limit; the device MoE runs Silu without the clamp (plan M8)."""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4RotaryEmbedding,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v4.block import TtV4PrefillBlock
from models.demos.deepseek_v3_d_p.tt.v4.kv_cache import allocate_v4_flash_kv_caches
from models.demos.deepseek_v3_d_p.tt.v4.moe import reference_moe_weights

_SEED = 42
_PCC = 0.99
_EXPERTS = 64
_SEQ = 1024

_MESH_CONFIGS = [
    pytest.param(
        (2, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        id="fabric2d-mesh-2x4",
    ),
]


def _cfg():
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)  # SWA SWA CSA HCA
    cfg.n_routed_experts = _EXPERTS
    return cfg


def init_reference_layer(cfg, layer_idx, seed=_SEED):
    torch.manual_seed(seed)
    layer = DeepseekV4DecoderLayer(cfg, layer_idx).eval()
    with torch.no_grad():
        a = layer.self_attn
        a.q_a_norm.weight.uniform_(0.5, 1.5)
        a.kv_norm.weight.uniform_(0.5, 1.5)
        a.sinks.normal_(0.0, 1.0)
        if a.compressor is not None:
            a.compressor.position_bias.normal_(0.0, 0.5 if hasattr(a.compressor, "indexer") else 0.02)
            a.compressor.kv_norm.weight.uniform_(0.5, 1.5)
            if hasattr(a.compressor, "indexer"):
                a.compressor.indexer.position_bias.normal_(0.0, 0.5)
                a.compressor.indexer.kv_norm.weight.uniform_(0.5, 1.5)
        for hcm in (layer.attn_hc, layer.ffn_hc):
            hcm.fn.normal_(0.0, 0.02)
            hcm.base.normal_(0.0, 0.3)
            hcm.scale.copy_(torch.tensor([1.1, 0.9, 1.3]))
        layer.input_layernorm.weight.uniform_(0.5, 1.5)
        layer.post_attention_layernorm.weight.uniform_(0.5, 1.5)
        m = layer.mlp
        m.gate.weight.normal_(0.0, 0.02)
        if hasattr(m.gate, "e_score_correction_bias"):
            m.gate.e_score_correction_bias.normal_(0.0, 0.1)
        if hasattr(m.gate, "tid2eid"):
            m.gate.tid2eid.copy_(torch.randint(0, cfg.n_routed_experts, tuple(m.gate.tid2eid.shape)))
        m.experts.gate_up_proj.normal_(0.0, 0.02)
        m.experts.down_proj.normal_(0.0, 0.02)
        for lin in (m.shared_experts.gate_proj, m.shared_experts.up_proj, m.shared_experts.down_proj):
            lin.weight.normal_(0.0, 0.02)
    return layer


def reference_layer_weights(layer) -> dict:
    w = {k: v.detach().clone() for k, v in layer.state_dict().items() if not k.startswith("mlp.experts.")}
    mw = reference_moe_weights(layer.mlp)
    w["__experts__"] = [{k: v.float() for k, v in e.items()} for e in mw["routed_expert_weights"]]
    return w


def reference_layer_forward(layer, cfg, rot, streams, input_ids):
    """streams [1, S, 4, D] -> [1, S, 4, D]; sliding causal mask; both rope tables."""
    S = streams.shape[1]
    pos = torch.arange(S).unsqueeze(0)
    i, j = torch.arange(S).view(S, 1), torch.arange(S).view(1, S)
    mask = torch.zeros(S, S).masked_fill(~((j <= i) & (i - j < cfg.sliding_window)), float("-inf")).view(1, 1, S, S)
    with torch.no_grad():
        pe = {
            "main": rot(streams, position_ids=pos, layer_type="main"),
            "compress": rot(streams, position_ids=pos, layer_type="compress"),
        }
        return layer(
            streams,
            position_embeddings=pe,
            position_ids=pos,
            attention_mask=mask,
            input_ids=input_ids,
            past_key_values=None,
        )


def to_device_streams(mesh_device, streams, sp_axis=0, tp_axis=1):
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
    return [
        ttnn.from_torch(
            streams[:, :, h, :].unsqueeze(1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
        )
        for h in range(4)
    ]


def to_host_streams(mesh_device, tt_streams):
    comp = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
    return torch.stack([ttnn.to_torch(t, mesh_composer=comp)[0, 0] for t in tt_streams], dim=1).unsqueeze(
        0
    )  # [1, S, 4, D]


@pytest.mark.parametrize("layer_idx", [0, 2, 3], ids=["swa-layer0-hash", "csa-layer2-hash", "hca-layer3-topk"])
@pytest.mark.parametrize("mesh_device, device_params", _MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_v4_block_vs_reference_layer(mesh_device, device_params, layer_idx):
    cfg = _cfg()
    layer = init_reference_layer(cfg, layer_idx)
    rot = DeepseekV4RotaryEmbedding(cfg)
    torch.manual_seed(_SEED + 1)
    streams = (torch.randn(1, _SEQ, 4, cfg.hidden_size) * 1.5).to(torch.bfloat16).float()
    input_ids = torch.randint(0, cfg.vocab_size, (1, _SEQ))
    out_ref = reference_layer_forward(layer, cfg, rot, streams, input_ids)

    sp = mesh_device.shape[0]
    params = SimpleNamespace(
        max_seq_len=_SEQ,
        sp_factor=sp,
        first_layer_idx=0,
        num_layers=4,
        mesh_shape=tuple(mesh_device.shape),
        sp_axis=0,
        num_users=1,
    )
    caches = allocate_v4_flash_kv_caches(mesh_device=mesh_device, hf_config=cfg, params=params)
    block = TtV4PrefillBlock(
        mesh_device,
        cfg,
        layer_idx,
        reference_layer_weights(layer),
        rotary_emb=rot,
        seq_len_per_chip=_SEQ // sp,
        num_routed_experts=_EXPERTS,
    )
    block.alloc_states(1, _SEQ, _SEQ)
    acks = []
    out = block(
        to_device_streams(mesh_device, streams),
        slot=0,
        caches=caches,
        actual_start=0,
        actual_end=_SEQ,
        input_ids=input_ids.view(-1),
        on_layer_complete=acks.append,
    )
    assert acks == [layer_idx]
    got = to_host_streams(mesh_device, out)
    worst = 1.0
    for h in range(4):
        _, pcc = comp_pcc(out_ref[0, :, h, :].float(), got[0, :, h, :].float())
        logger.info(f"layer {layer_idx} ({block.kind}) stream {h}: PCC {pcc:.6f}")
        worst = min(worst, pcc)
    assert worst >= _PCC, f"worst stream PCC {worst:.6f} < {_PCC}"

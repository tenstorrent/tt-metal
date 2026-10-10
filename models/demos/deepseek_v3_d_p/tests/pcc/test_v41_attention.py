# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill attention vs the checkpoint's own ``inference/model.py``, REAL weights, REAL inputs: the
needle-1200 prompt's layer inputs from tt-blaze's per-layer prefill golden (``prefill_trace.py``,
``V41_PREFILL_TRACE``). The reference attention runs on the host in this process (tt-blaze ``golden/reference.py``,
``model.py`` + its CPU kernel shim; ``PYTHONPATH`` must carry tt-blaze's ttnn-free ``cpu_stub`` blaze package).

    PYTHONPATH=<this worktree>:<cpu_stub> pytest models/demos/deepseek_v3_d_p/tests/pcc/test_v41_attention.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v41.attention import V41SWA, v41_hf_config
from models.demos.deepseek_v3_d_p.tt.v41.config import V41Config
from models.demos.deepseek_v3_d_p.tt.v41.weights import checkpoint

TRACE = os.environ.get("V41_PREFILL_TRACE", "/mnt/tt-data/sdawle/dsv41_golden/prefill_trace_n1200")
SEQ = int(os.environ.get("V41_TEST_SEQ", "1024"))  # a multiple of 32 * sp
PCC = float(os.environ.get("V41_TEST_PCC", "0.99"))

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


def _reference_attention(layer_id: int, x_in: torch.Tensor, pre_in: torch.Tensor):
    """model.py on the host: (attention input [S, d], attention output [S, d]) for the block's golden input."""
    from blaze.models.deepseek_v4_1_flash.golden.reference import build_model

    model, *_ = build_model(n_layers=layer_id + 1, max_seq_len=SEQ + 64, threads=16, expert_cache_gb=4.0)
    layer = model.layers[layer_id]
    with torch.inference_mode():
        h = layer.attn_norm(layer.hc_pre(x_in.unsqueeze(0), pre_in.unsqueeze(0)))
        out = layer.attn(h, 0)
    return h[0].float(), out[0].float()


def _to_device_hidden(mesh_device, h: torch.Tensor):
    """[S, d] -> [1, 1, S/sp, d/tp] (SP on mesh dim 0, TP on dim 1)."""
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
    return ttnn.from_torch(
        h.to(torch.bfloat16).reshape(1, 1, *h.shape),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
    )


def _to_host_hidden(mesh_device, t) -> torch.Tensor:
    comp = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
    return ttnn.to_torch(t, mesh_composer=comp)[0, 0].float()


@pytest.mark.parametrize("layer_id", [0], ids=["swa-layer0"])
@pytest.mark.parametrize("mesh_device, device_params", _MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_v41_attention_vs_model_py(mesh_device, device_params, layer_id):
    cfg = V41Config.load()
    rot = DeepseekV4RotaryEmbedding(v41_hf_config(cfg, max_seq=SEQ))
    t = torch.load(os.path.join(TRACE, f"layer_{layer_id}.pt"))
    x_in, pre_in = t["x_in"][:SEQ].float(), t["pre_in"][:SEQ].float()
    h_ref, out_ref = _reference_attention(layer_id, x_in, pre_in)
    logger.info(
        f"[v41 attn L{layer_id}] reference: input rms {h_ref.square().mean().sqrt():.4f}, "
        f"output rms {out_ref.square().mean().sqrt():.4f}"
    )

    w = checkpoint().layer(layer_id)
    attn = V41SWA.from_weights(mesh_device, cfg, layer_id, w, rot)
    state = attn.alloc_state(SEQ, chunk_tokens=SEQ)
    y = attn(_to_device_hidden(mesh_device, h_ref), seq_len_actual=SEQ, state=state)
    got = _to_host_hidden(mesh_device, y)[:SEQ]
    _, pcc = comp_pcc(out_ref, got)
    rel = float((got - out_ref).norm() / out_ref.norm())
    logger.info(f"[v41 attn L{layer_id}] PCC {pcc:.6f}, |dev - ref| / |ref| {rel:.4f}")
    assert pcc >= PCC, f"layer {layer_id} attention PCC {pcc:.6f} < {PCC}"

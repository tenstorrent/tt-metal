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
from functools import lru_cache

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v41.attention import V41CSA, V41SWA, V41CSAConsumer, v41_hf_config
from models.demos.deepseek_v3_d_p.tt.v41.config import V41Config
from models.demos.deepseek_v3_d_p.tt.v41.weights import checkpoint

TRACE = os.environ.get("V41_PREFILL_TRACE", "/mnt/tt-data/sdawle/dsv41_golden/prefill_trace_n1200")
SEQ = int(os.environ.get("V41_TEST_SEQ", "1024"))  # a multiple of 32 * sp (256 on 8 x 4)
PCC = float(os.environ.get("V41_TEST_PCC", "0.99"))
# the full galaxy by default: a 2 x 4 FABRIC_2D sub-mesh of a 32-chip galaxy fails the router handshake toward the chips
# outside it (host 30, 2026-10-10: "Fabric Router Sync: Timeout ... on Device 1", also right after a glx reset)
MESH = tuple(int(v) for v in os.environ.get("V41_TEST_MESH", "8,4").split(","))

_MESH_CONFIGS = [
    pytest.param(
        MESH,
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        id=f"fabric2d-mesh-{MESH[0]}x{MESH[1]}",
    ),
]


@lru_cache(maxsize=1)
def _reference_model(n_layers: int):
    from blaze.models.deepseek_v4_1_flash.golden.reference import build_model

    model, *_ = build_model(n_layers=n_layers, max_seq_len=SEQ + 64, threads=16, expert_cache_gb=4.0)
    return model


def _reference_attention(layers: list[int]):
    """model.py on the host, ``layers`` IN ORDER (a consumer reads the compressed KV / top-k its source just published
    through ``shared_attn``): {layer: (attention input [S, d], attention output [S, d])} for each block's golden input.
    """
    model = _reference_model(max(layers) + 1)
    out = {}
    for lid in layers:
        t = torch.load(os.path.join(TRACE, f"layer_{lid}.pt"))
        # the streams stay bf16 as in model.py (its bf16 Linears -- e.g. the indexer's wk -- reject an fp32 input)
        x_in, pre_in = t["x_in"][:SEQ].to(torch.bfloat16), t["pre_in"][:SEQ].float()
        layer = model.layers[lid]
        with torch.inference_mode():
            h = layer.attn_norm(layer.hc_pre(x_in.unsqueeze(0), pre_in.unsqueeze(0)))
            y = layer.attn(h, 0)
        out[lid] = (h[0].float(), y[0].float())
        logger.info(
            f"[v41 attn L{lid}] reference: input rms {h.float().square().mean().sqrt():.4f}, "
            f"output rms {y.float().square().mean().sqrt():.4f}"
        )
    return out


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


def _check(lid, got, ref):
    _, pcc = comp_pcc(ref, got)
    rel = float((got - ref).norm() / ref.norm())
    logger.info(f"[v41 attn L{lid}] PCC {pcc:.6f}, |dev - ref| / |ref| {rel:.4f}")
    return pcc


# (layers run in order on one chunk; the last is checked, every one is logged)
#   swa-layer0: sliding window, "main" RoPE
#   csa-layer2: KV + index source at ratio 2 (V4.1 compressor, latent-derived index keys, 32-head indexer, dense path B)
#   reuse-layer3: layer 3 over layer 2's entries and top-k (no compressor / indexer of its own)
@pytest.mark.parametrize("layers", [[0], [2], [2, 3]], ids=["swa-layer0", "csa-layer2", "reuse-layer3"])
@pytest.mark.parametrize("mesh_device, device_params", _MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_v41_attention_vs_model_py(mesh_device, device_params, layers):
    cfg = V41Config.load()
    ref = _reference_attention(layers)
    ck = checkpoint()
    mods, states = {}, {}
    for lid in layers:
        r = cfg.role(lid)
        rot = DeepseekV4RotaryEmbedding(v41_hf_config(cfg, max_seq=SEQ))
        w = ck.layer(lid)
        if r.mode == "swa":
            m = V41SWA.from_weights(mesh_device, cfg, lid, w, rot)
        elif r.mode == "full":
            m = V41CSA.from_weights(mesh_device, cfg, lid, w, rot)
        else:
            m = V41CSAConsumer.from_weights(mesh_device, cfg, lid, w, rot, source=mods[r.kv_source])
        mods[lid], states[lid] = m, m.alloc_state(SEQ, chunk_tokens=SEQ)
    pccs = {}
    for lid in layers:
        y = mods[lid](_to_device_hidden(mesh_device, ref[lid][0]), seq_len_actual=SEQ, state=states[lid])
        pccs[lid] = _check(lid, _to_host_hidden(mesh_device, y)[:SEQ], ref[lid][1])
    lid = layers[-1]
    assert pccs[lid] >= PCC, f"layer {lid} attention PCC {pccs[lid]:.6f} < {PCC}"

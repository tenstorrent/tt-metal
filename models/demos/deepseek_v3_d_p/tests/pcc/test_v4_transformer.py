# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TtV4PrefillTransformer (embedding -> 4 streams -> SWA,SWA,CSA,HCA blocks -> HyperHead -> norm), driven in two
chunks through the engine-shaped API, vs the reference DeepseekV4Model in one unchunked pass. Random weights, 64
experts (8 per chip on 2x4). Final hidden PCC >= 0.99 per chunk."""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tests.pcc.test_v4_block import init_reference_layer, reference_layer_weights
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v4.kv_cache import allocate_v4_flash_kv_caches
from models.demos.deepseek_v3_d_p.tt.v4.runtime import TtV4PrefillRuntime, TtV4PrefillRuntimeConfig

_SEED = 42
_PCC = 0.99
_EXPERTS = 64
_CHUNK = 1024
_CHUNKS = [1024, 1024]

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


def _init_model(cfg):
    torch.manual_seed(_SEED)
    model = DeepseekV4Model(cfg).eval()
    with torch.no_grad():
        model.embed_tokens.weight.normal_(0.0, 0.5)
        model.hc_head.hc_fn.normal_(0.0, 0.02)
        model.hc_head.hc_base.normal_(0.0, 0.3)
        model.hc_head.hc_scale.fill_(0.8)
        model.norm.weight.uniform_(0.5, 1.5)
    for i, layer in enumerate(model.layers):
        model.layers[i] = init_reference_layer(cfg, i, seed=_SEED + 10 + i)
    return model


@pytest.mark.parametrize("mesh_device, device_params", _MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_v4_transformer_two_chunks(mesh_device, device_params):
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)  # SWA SWA CSA HCA
    cfg.n_routed_experts = _EXPERTS
    model = _init_model(cfg)
    total = sum(_CHUNKS)
    torch.manual_seed(_SEED + 1)
    ids = torch.randint(0, cfg.vocab_size, (1, total))
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).last_hidden_state  # [1, total, D]

    sp, tp = mesh_device.shape
    top = {
        "model.embed_tokens.weight": model.embed_tokens.weight.detach().clone(),
        "model.hc_head.hc_fn": model.hc_head.hc_fn.detach().clone(),
        "model.hc_head.hc_base": model.hc_head.hc_base.detach().clone(),
        "model.hc_head.hc_scale": model.hc_head.hc_scale.detach().clone(),
        "model.norm.weight": model.norm.weight.detach().clone(),
    }
    rc = TtV4PrefillRuntimeConfig(
        chunk_size=_CHUNK,
        max_seq_len=total,
        first_layer_idx=0,
        num_layers=4,
        is_first_rank=True,
        is_last_rank=True,
        num_users=1,
        mesh_shape=(sp, tp),
        kv_only_last_layer=False,
    )
    params = SimpleNamespace(
        max_seq_len=total, sp_factor=sp, first_layer_idx=0, num_layers=4, mesh_shape=(sp, tp), sp_axis=0, num_users=1
    )
    caches = allocate_v4_flash_kv_caches(mesh_device=mesh_device, hf_config=cfg, params=params)
    rt = TtV4PrefillRuntime(
        mesh_device,
        cfg,
        rc,
        layer_weights=lambda i: reference_layer_weights(model.layers[i]),
        top_level_weights=top,
        num_routed_experts=_EXPERTS,
    )
    rt.compile(caches)
    acks = []
    rt.set_layer_completion_sink(lambda layer_idx, request_id: acks.append((request_id, layer_idx)))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=(2, 3))
    start = 0
    worst = 1.0
    for r, n in enumerate(_CHUNKS):
        x = rt.make_chunk_input(ids[0, start : start + n].tolist())
        out = rt.model(
            x,
            slot=0,
            caches=caches,
            actual_start=start,
            actual_end=start + n,
            input_ids=ids[0, start : start + n],
            on_layer_complete=lambda i: acks.append((r, i)),
        )
        got = ttnn.to_torch(out, mesh_composer=composer)[0, 0]  # [n, D]
        _, pcc = comp_pcc(ref[0, start : start + n].float(), got.float())
        logger.info(f"chunk {r} [{start}, {start + n}): final hidden PCC {pcc:.6f}")
        worst = min(worst, pcc)
        start += n
    assert acks == [(r, i) for r in range(len(_CHUNKS)) for i in range(4)]
    assert worst >= _PCC, f"worst chunk PCC {worst:.6f} < {_PCC}"

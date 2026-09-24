# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TtSWA (V4-Flash sliding-window attention, layers 0-1) vs the reference DeepseekV4Attention with random weights.

Mirrors tests/pcc/test_ttnn_hca.py: single-shot for arbitrary prompt lengths (padding + trim + mask), and chunked
prefill with the window carry across chunks against an UNCHUNKED reference. Floors: 0.998 single-shot, 0.997 per
chunk (the HCA floors; SWA has no compressor, so it should meet them). Sub-galaxy configs run via
TT_VISIBLE_DEVICES (+ TT_MESH_GRAPH_DESC_PATH for [2,4]) and carry no requires_mesh_topology mark."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4RotaryEmbedding,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v4.attention.swa import TtSWA
from tests.ttnn.utils_for_testing import assert_with_pcc

_SEED = 42
_SINGLE_PCC = 0.998
_CHUNK_PCC = 0.997
_SHAPES = [128, 130, 1024, 4095]
# (name, chunk width, real lengths): non-final chunks >= 128 and tile-aligned; the last may be ragged.
_CHUNKED = [
    ("chunk1024-ragged", 1024, [1024, 1024, 300]),
    ("chunk5120-varying", 5120, [1024, 256, 5120]),
    ("chunk5120-full", 5120, [5120, 5120]),
]

_MESH_CONFIGS = [
    pytest.param((1, 1), {}, 0, 1, id="single-1x1"),
    pytest.param((2, 1), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, 0, 1, id="sp2-2x1"),  # SP 2, TP 1
    pytest.param(
        (2, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        0,
        1,
        id="fabric2d-mesh-2x4",
    ),
    pytest.param(
        (8, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        0,
        1,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-mesh-8x4",
    ),
]


def _ref_layer(cfg):
    ref = DeepseekV4Attention(cfg, layer_idx=0).eval()
    assert ref.compressor is None, "layer 0 of V4-Flash is a sliding_attention layer"
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.sinks.normal_(0.0, 1.0)
    return ref


def _ref_forward(ref, rot, hidden, sw):
    total = hidden.shape[1]
    position_ids = torch.arange(total).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rot(hidden, position_ids=position_ids, layer_type="main")
        i = torch.arange(total).view(total, 1)
        j = torch.arange(total).view(1, total)
        mask = torch.zeros(total, total).masked_fill(~((j <= i) & (i - j < sw)), float("-inf")).view(1, 1, total, total)
        out, _ = ref(hidden, {"main": (cos, sin)}, position_ids, mask, past_key_values=None)
    return out


def _dims(mesh_device, sp_axis, tp_axis):
    dims = [None, None]
    if mesh_device.shape[sp_axis] > 1:
        dims[sp_axis] = 2
    if mesh_device.shape[tp_axis] > 1:
        dims[tp_axis] = 3
    return dims


def _to_device(mesh_device, x, sp_axis, tp_axis):
    dims = _dims(mesh_device, sp_axis, tp_axis)
    mapper = (
        ttnn.ReplicateTensorToMesh(mesh_device)
        if dims == [None, None]
        else ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    )
    return ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)


def _to_host(mesh_device, t, sp_axis, tp_axis):
    dims = [0, 0]
    dims[sp_axis], dims[tp_axis] = 2, 3
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))
    )


@pytest.mark.parametrize("seq_len", _SHAPES, ids=[f"seq{s}" for s in _SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_swa_forward(mesh_device, device_params, sp_axis, tp_axis, seq_len):
    torch.manual_seed(_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref, rot = _ref_layer(cfg), DeepseekV4RotaryEmbedding(cfg)
    sp = mesh_device.shape[sp_axis]
    hidden = torch.randn(1, seq_len, cfg.hidden_size)
    out_ref = _ref_forward(ref, rot, hidden, cfg.sliding_window)

    tt = TtSWA.from_reference(mesh_device, ref, cfg, rotary_emb=rot, sp_axis=sp_axis, tp_axis=tp_axis)
    padded, real = TtSWA.prepare_input(hidden, sp)
    state = tt.alloc_state(padded.shape[1])
    out = tt(_to_device(mesh_device, padded.unsqueeze(1), sp_axis, tp_axis), seq_len_actual=real, state=state)
    out = _to_host(mesh_device, out, sp_axis, tp_axis).squeeze(1)[:, :real]
    assert out.shape == out_ref.shape
    ok, msg = assert_with_pcc(out_ref.float(), out.float(), pcc=_SINGLE_PCC)
    logger.info(f"SWA single-shot seq {seq_len} mesh {tuple(mesh_device.shape)}: {msg}")
    assert ok, msg


@pytest.mark.parametrize("name, chunk_size, iters_valid", _CHUNKED, ids=[n for n, _, _ in _CHUNKED])
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_swa_chunked_prefill(mesh_device, device_params, sp_axis, tp_axis, name, chunk_size, iters_valid):
    """The reference is NOT chunked: one pass over the whole prompt, each chunk compared to its slice, so the
    carry has to reproduce plain sliding-window attention."""
    torch.manual_seed(_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref, rot = _ref_layer(cfg), DeepseekV4RotaryEmbedding(cfg)
    total = sum(iters_valid)
    hidden = torch.randn(1, total, cfg.hidden_size)
    out_ref = _ref_forward(ref, rot, hidden, cfg.sliding_window)

    tt = TtSWA.from_reference(mesh_device, ref, cfg, rotary_emb=rot, sp_axis=sp_axis, tp_axis=tp_axis)
    state = tt.alloc_state(total, chunk_tokens=chunk_size)
    kv_actual, pccs = 0, []
    programs = mesh_device.num_program_cache_entries()
    for it, valid in enumerate(iters_valid):
        chunk = torch.zeros(1, chunk_size, cfg.hidden_size)
        chunk[:, :valid] = hidden[:, kv_actual : kv_actual + valid]
        out = tt(_to_device(mesh_device, chunk.unsqueeze(1), sp_axis, tp_axis), seq_len_actual=valid, state=state)
        out = _to_host(mesh_device, out, sp_axis, tp_axis).squeeze(1)[:, :valid]
        _, pcc = comp_pcc(out_ref[:, kv_actual : kv_actual + valid].float(), out.float())
        pccs.append((it, kv_actual, valid, pcc))
        now = mesh_device.num_program_cache_entries()
        logger.info(
            f"  {name} iter {it} (kv_actual={kv_actual} valid={valid}): PCC {pcc:.6f}, programs +{now - programs}"
        )
        if it > 0:
            assert now == programs, f"chunk {it} compiled {now - programs} new program(s); a shape or attribute moved"
        programs = now
        kv_actual += valid
    assert state.kv_actual == total and state.compressed_kv is None
    worst = min(pccs, key=lambda r: r[3])
    assert worst[3] >= _CHUNK_PCC, f"worst chunk PCC {worst[3]:.6f} (iter {worst[0]}) < {_CHUNK_PCC}"

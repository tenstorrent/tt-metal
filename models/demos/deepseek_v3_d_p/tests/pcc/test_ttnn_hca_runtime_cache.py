# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""M4: TtHCA / TtSWA mirror their compressed entries and the 128-row window ring into the engine-owned UNIFIED
caches of the prefill <-> decode contract (tt/v4/kv_contract.py), 2 users x 2 layers per kind, chunked.

Golden for the cache CONTENT (not the attention output): the reference compressor's entries for rows
``128 + w`` and the reference's roped K rows of the last 128 real tokens for rows ``[0, 128)`` in ring order
(row = token % 128). The caches are allocated the way the runtime allocates them (allocate_v4_flash_kv_caches:
bfp8 tiles, ND-sharded, SP-replicated), so this also proves fill_cache_for_user_ on that layout."""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCA
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v4.attention.swa import TtSWA
from models.demos.deepseek_v3_d_p.tt.v4.kv_cache import allocate_v4_flash_kv_caches

_SEED = 42
_ENTRY_PCC = 0.998  # the HCA test's cache floor; the unified copy is bfp8
_RING_PCC = 0.99
_USERS = 2
# (name, chunk width, real lengths). 3000 = final chunk >= one window but not a multiple of the rate (exact carry);
# 100 = final chunk shorter than a window (prev-carry merge); 256 after 1024 = E_prev % 128 == 0 but a short slab.
_SCENARIOS = [
    ("5120+3000", 5120, [5120, 3000]),
    ("1024+256+100", 1024, [1024, 256, 100]),
]
_MESH_CONFIGS = [
    pytest.param((1, 1), {}, 0, 1, id="single-1x1"),
    pytest.param((2, 1), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, 0, 1, id="sp2-2x1"),
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
]


def _init_ref(ref):
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.sinks.normal_(0.0, 1.0)
        if ref.compressor is not None:
            ref.compressor.position_bias.normal_(0.0, 0.02)
            ref.compressor.kv_norm.weight.uniform_(0.5, 1.5)
    return ref


def _ref_k_rows(ref, rot, hidden, layer_type):
    """The reference's cached K (== V) rows: kv_norm(kv_proj(h)) roped at each token's position."""
    total = hidden.shape[1]
    pos = torch.arange(total).unsqueeze(0)
    with torch.no_grad():
        kv = ref.kv_norm(ref.kv_proj(hidden)).view(1, total, 1, -1).transpose(1, 2)  # [1, 1, S, 512]
        cos, sin = rot(hidden, position_ids=pos, layer_type=layer_type)
        return apply_rotary_pos_emb(kv, cos, sin)[0, 0]  # [S, 512]


def _mapper(mesh_device, sp_axis, tp_axis):
    dims = [None, None]
    if mesh_device.shape[sp_axis] > 1:
        dims[sp_axis] = 2
    if mesh_device.shape[tp_axis] > 1:
        dims[tp_axis] = 3
    if dims == [None, None]:
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)


def _read_replicated(mesh_device, t):
    return ttnn.to_torch(
        t, mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1)))
    )


def _ring_golden(k_rows, total, sw=128):
    ring = torch.zeros(sw, k_rows.shape[-1])
    for t in range(max(0, total - sw), total):
        ring[t % sw] = k_rows[t]
    return ring


def _drive(mesh_device, sp_axis, tp_axis, module, cache, batch_idx, hidden, chunk_size, iters_valid):
    state = module.alloc_state(sum(iters_valid), chunk_tokens=chunk_size)
    kv_actual = 0
    for valid in iters_valid:
        chunk = torch.zeros(1, chunk_size, hidden.shape[-1])
        chunk[:, :valid] = hidden[:, kv_actual : kv_actual + valid]
        x = ttnn.from_torch(
            chunk.unsqueeze(1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=_mapper(mesh_device, sp_axis, tp_axis),
        )
        module(x, seq_len_actual=valid, state=state, export=(cache, batch_idx))
        kv_actual += valid
    return state


@pytest.mark.parametrize("name, chunk_size, iters_valid", _SCENARIOS, ids=[n for n, _, _ in _SCENARIOS])
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_unified_cache_export(mesh_device, device_params, sp_axis, tp_axis, name, chunk_size, iters_valid):
    torch.manual_seed(_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=6)  # layers: SWA SWA CSA HCA CSA HCA
    total = sum(iters_valid)
    sp = mesh_device.shape[sp_axis]
    params = SimpleNamespace(
        max_seq_len=-(-total // 5120) * 5120,
        sp_factor=sp,
        first_layer_idx=0,
        num_layers=6,
        mesh_shape=tuple(mesh_device.shape),
        sp_axis=sp_axis,
        num_users=_USERS,
    )
    caches = allocate_v4_flash_kv_caches(mesh_device=mesh_device, hf_config=cfg, params=params)
    geom = caches.geometry
    assert geom.hca_layers == (3, 5) and geom.swa_layers == (0, 1)
    assert tuple(caches.hca_unified.shape)[0] == _USERS * 2 and tuple(caches.swa_window.shape) == (
        _USERS * 2,
        1,
        128,
        512,
    )
    rot = DeepseekV4RotaryEmbedding(cfg)

    ref_hca = _init_ref(DeepseekV4Attention(cfg, layer_idx=3).eval())
    ref_swa = _init_ref(DeepseekV4Attention(cfg, layer_idx=0).eval())
    tt_hca = TtHCA.from_reference(mesh_device, ref_hca, cfg, sp_axis=sp_axis, tp_axis=tp_axis)
    tt_swa = TtSWA.from_reference(mesh_device, ref_swa, cfg, rotary_emb=rot, sp_axis=sp_axis, tp_axis=tp_axis)

    hiddens = {}
    for slot in range(_USERS):
        for kind_rank in range(2):  # layers 3 and 5 share weights here; different inputs per (slot, layer)
            hiddens[(slot, kind_rank)] = torch.randn(1, total, cfg.hidden_size)

    # ---- HCA layers 3, 5: entries + ring
    for (slot, kr), hidden in hiddens.items():
        b = slot * 2 + kr
        state = _drive(mesh_device, sp_axis, tp_axis, tt_hca, caches.hca_unified, b, hidden, chunk_size, iters_valid)
        assert state.kv_actual == total
    cache = _read_replicated(mesh_device, caches.hca_unified).float()  # [4, 1, rows, 512]
    worst_e, worst_r = 1.0, 1.0
    for (slot, kr), hidden in hiddens.items():
        b = slot * 2 + kr
        n_entries = total // 128
        with torch.no_grad():
            ref_entries, _ = ref_hca.compressor(
                hidden,
                torch.zeros(1, total, cfg.q_lora_rank),
                torch.arange(total).unsqueeze(0),
                past_key_values=None,
                layer_idx=3,
            )
        got = cache[b, 0, 128 : 128 + n_entries]
        _, pcc_e = comp_pcc(ref_entries[0].float(), got)
        ring_ref = _ring_golden(_ref_k_rows(ref_hca, rot, hidden, "compress"), total)
        _, pcc_r = comp_pcc(ring_ref, cache[b, 0, :128])
        logger.info(
            f"{name} HCA slot {slot} layer {(3, 5)[kr]} (batch {b}): entries PCC {pcc_e:.6f} ({n_entries}), ring PCC {pcc_r:.6f}"
        )
        worst_e, worst_r = min(worst_e, pcc_e), min(worst_r, pcc_r)
        # rows past the entries are still zero (nothing wrote them)
        assert cache[b, 0, 128 + -(-n_entries // 32) * 32 :].abs().max() == 0
    assert worst_e >= _ENTRY_PCC, f"HCA entries PCC {worst_e:.6f} < {_ENTRY_PCC}"
    assert worst_r >= _RING_PCC, f"HCA ring PCC {worst_r:.6f} < {_RING_PCC}"

    # ---- SWA layers 0, 1: ring only
    for (slot, kr), hidden in hiddens.items():
        b = slot * 2 + kr
        _drive(mesh_device, sp_axis, tp_axis, tt_swa, caches.swa_window, b, hidden, chunk_size, iters_valid)
    cache = _read_replicated(mesh_device, caches.swa_window).float()
    worst_r = 1.0
    for (slot, kr), hidden in hiddens.items():
        b = slot * 2 + kr
        ring_ref = _ring_golden(_ref_k_rows(ref_swa, rot, hidden, "main"), total)
        _, pcc_r = comp_pcc(ring_ref, cache[b, 0])
        logger.info(f"{name} SWA slot {slot} layer {kr} (batch {b}): ring PCC {pcc_r:.6f}")
        worst_r = min(worst_r, pcc_r)
    assert worst_r >= _RING_PCC, f"SWA ring PCC {worst_r:.6f} < {_RING_PCC}"

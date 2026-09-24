# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TtCSA (V4-Flash compressed sparse attention, path B) vs the reference DeepseekV4Attention of layer 2 with random
weights: the two-series compressor's entries (PCC >= 0.999), the indexer's top-k selection (Jaccard >= 0.9 vs the
reference block bias), and the block output single-shot / chunked (PCC >= 0.99) against an UNCHUNKED reference."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v4.attention.csa import TtCSA, TtCSACompressor

_SEED = 42
_COMPRESSOR_PCC = 0.999
_JACCARD = 0.9
_BLOCK_PCC = 0.99
_SINGLE_SHAPES = [1024, 4095]
_CHUNKED = [("chunk1024-ragged", 1024, [1024, 1024, 300]), ("chunk5120-varying", 5120, [1024, 256, 5120])]
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


def _ref_layer(cfg):
    ref = DeepseekV4Attention(cfg, layer_idx=2).eval()
    assert ref.compressor is not None and hasattr(ref.compressor, "indexer"), "layer 2 must be a CSA layer"
    with torch.no_grad():
        ref.q_a_norm.weight.uniform_(0.5, 1.5)
        ref.kv_norm.weight.uniform_(0.5, 1.5)
        ref.sinks.normal_(0.0, 1.0)
        ref.compressor.position_bias.normal_(0.0, 0.5)
        ref.compressor.kv_norm.weight.uniform_(0.5, 1.5)
        ref.compressor.indexer.position_bias.normal_(0.0, 0.5)
        ref.compressor.indexer.kv_norm.weight.uniform_(0.5, 1.5)
    return ref


def _ref_forward(ref, hidden, sw):
    total = hidden.shape[1]
    pos = torch.arange(total).unsqueeze(0)
    with torch.no_grad():
        cos, sin = ref.compressor.rotary_emb(hidden, position_ids=pos, layer_type="compress")
        i, j = torch.arange(total).view(total, 1), torch.arange(total).view(1, total)
        mask = torch.zeros(total, total).masked_fill(~((j <= i) & (i - j < sw)), float("-inf")).view(1, 1, total, total)
        out, _ = ref(hidden, {"compress": (cos, sin)}, pos, mask, past_key_values=None)
    return out


def _mapper(mesh_device, sp_axis, tp_axis):
    dims = [None, None]
    if mesh_device.shape[sp_axis] > 1:
        dims[sp_axis] = 2
    if mesh_device.shape[tp_axis] > 1:
        dims[tp_axis] = 3
    if dims == [None, None]:
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)


def _to_device(mesh_device, x, sp_axis, tp_axis):
    return ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=_mapper(mesh_device, sp_axis, tp_axis),
    )


def _to_host(mesh_device, t, sp_axis, tp_axis):
    dims = [0, 0]
    dims[sp_axis], dims[tp_axis] = 2, 3
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))
    )


def _one_chip(mesh_device, t):
    return ttnn.to_torch(
        t, mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1)))
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_csa_compressor_entries(mesh_device, device_params, sp_axis, tp_axis):
    """Two chunks through the compressor alone: the second chunk's window 0 takes the first chunk's last Ca rows."""
    torch.manual_seed(_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref = _ref_layer(cfg)
    sp = mesh_device.shape[sp_axis]
    S1, S2 = 1024, 1024
    hidden = torch.randn(1, S1 + S2, cfg.hidden_size)
    q_res = torch.randn(1, S1 + S2, cfg.q_lora_rank)
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACache

    class _Cache:
        def __init__(self, layer):
            self.layers = [None, None, layer]

    cache = _Cache(DeepseekV4CSACache(cfg))
    with torch.no_grad():
        ref.compressor(hidden[:, :S1], q_res[:, :S1], torch.arange(S1).unsqueeze(0), cache, 2)
        all_entries, _ = ref.compressor(hidden[:, S1:], q_res[:, S1:], torch.arange(S1, S1 + S2).unsqueeze(0), cache, 2)
    all_entries = all_entries[0, 0]  # [(S1+S2)/4, 512]

    tt = TtCSACompressor.from_reference(mesh_device, ref.compressor, cfg, sp_axis=sp_axis, tp_axis=tp_axis)
    cap = (S1 + S2) // 4 + 32
    tt.alloc_tables(S1 + S2, S1, cap)
    prior = tt.empty_prior()
    got = []
    for c, (lo, hi) in enumerate(((0, S1), (S1, S1 + S2))):
        x = _to_device(mesh_device, hidden[:, lo:hi].unsqueeze(1), sp_axis, tp_axis)
        entries, mask_block, prior = tt(x, hi - lo, lo // 4 * 4, prior)
        got.append(_one_chip(mesh_device, entries)[0, 0].float())
        assert tuple(mask_block.shape) == (1, 1, S1 // sp, cap)
    got = torch.cat(got, 0)
    ok, pcc = comp_pcc(all_entries.float(), got, _COMPRESSOR_PCC)
    logger.info(f"CSA compressor entries PCC {pcc} (two chunks of {S1})")
    assert ok, pcc


@pytest.mark.parametrize("seq_len", _SINGLE_SHAPES, ids=[f"seq{s}" for s in _SINGLE_SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_csa_forward(mesh_device, device_params, sp_axis, tp_axis, seq_len):
    torch.manual_seed(_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref = _ref_layer(cfg)
    sp = mesh_device.shape[sp_axis]
    hidden = torch.randn(1, seq_len, cfg.hidden_size)
    out_ref = _ref_forward(ref, hidden, cfg.sliding_window)

    tt = TtCSA.from_reference(mesh_device, ref, cfg, sp_axis=sp_axis, tp_axis=tp_axis)
    padded, real = TtCSA.prepare_input(hidden, sp)
    state = tt.alloc_state(padded.shape[1])
    out = tt(_to_device(mesh_device, padded.unsqueeze(1), sp_axis, tp_axis), seq_len_actual=real, state=state)
    out = _to_host(mesh_device, out, sp_axis, tp_axis).squeeze(1)[:, :real]
    ok, pcc = comp_pcc(out_ref.float(), out.float(), _BLOCK_PCC)
    logger.info(f"CSA single-shot seq {seq_len} mesh {tuple(mesh_device.shape)}: PCC {pcc}")

    # indexer selection vs the reference block bias (sets of selected entries per query row)
    with torch.no_grad():
        q_res = ref.q_a_norm(ref.q_a_proj(hidden))
        _, block_bias = ref.compressor(hidden, q_res, torch.arange(seq_len).unsqueeze(0), None, 2)
    ref_sel = block_bias[0, 0] == 0  # [S, T]
    sel = _to_host(mesh_device, tt.debug_last_selection, sp_axis, tp_axis)[0, 0]  # [S_pad, cap]
    sel = sel[:real, : ref_sel.shape[1]] == 0
    inter = (sel & ref_sel).sum(-1).float()
    union = (sel | ref_sel).sum(-1).float().clamp(min=1)
    jaccard = (inter / union).mean().item()
    logger.info(f"CSA indexer selection Jaccard {jaccard:.4f} (rows {real}, entries {ref_sel.shape[1]})")
    assert jaccard >= _JACCARD, jaccard
    assert ok, pcc


@pytest.mark.parametrize("name, chunk_size, iters_valid", _CHUNKED, ids=[n for n, _, _ in _CHUNKED])
@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_csa_chunked_prefill(mesh_device, device_params, sp_axis, tp_axis, name, chunk_size, iters_valid):
    torch.manual_seed(_SEED)
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    ref = _ref_layer(cfg)
    total = sum(iters_valid)
    hidden = torch.randn(1, total, cfg.hidden_size)
    out_ref = _ref_forward(ref, hidden, cfg.sliding_window)
    tt = TtCSA.from_reference(mesh_device, ref, cfg, sp_axis=sp_axis, tp_axis=tp_axis)
    state = tt.alloc_state(total, chunk_tokens=chunk_size)
    kv_actual, pccs = 0, []
    programs = mesh_device.num_program_cache_entries()
    for it, valid in enumerate(iters_valid):
        chunk = torch.zeros(1, chunk_size, cfg.hidden_size)
        chunk[:, :valid] = hidden[:, kv_actual : kv_actual + valid]
        out = tt(_to_device(mesh_device, chunk.unsqueeze(1), sp_axis, tp_axis), seq_len_actual=valid, state=state)
        out = _to_host(mesh_device, out, sp_axis, tp_axis).squeeze(1)[:, :valid]
        _, pcc = comp_pcc(out_ref[:, kv_actual : kv_actual + valid].float(), out.float())
        now = mesh_device.num_program_cache_entries()
        logger.info(
            f"  {name} iter {it} (kv_actual={kv_actual} valid={valid}): PCC {pcc:.6f}, programs +{now - programs}"
        )
        pccs.append(pcc)
        programs = now
        kv_actual += valid
    assert state.kv_actual == total and state.entry_count == sum(v // 4 for v in iters_valid)
    assert min(pccs) >= _BLOCK_PCC, f"worst chunk PCC {min(pccs):.6f} < {_BLOCK_PCC}"

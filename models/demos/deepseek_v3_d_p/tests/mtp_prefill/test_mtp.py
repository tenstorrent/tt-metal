# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for GLM-5.2 MTP, single galaxy.

Three tests, most-local first: the fused projection alone, one whole MTP module, and K levels over
that module with per-slot KV comparison. Every test carries both weight options.
"""

from __future__ import annotations

import copy

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_2.mtp import (
    fused_mtp_reference,
    glm_mtp_module_reference,
    glm_mtp_predictor_reference,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import build_weights
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.tt_mtp import TtFusedMTP, TtMTPModule, TtMTPPredictor
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import (
    eh_proj_expected_chip_shard,
    eh_proj_to_tt_layout,
    enable_mtp_indexer_slot,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc

FUSED_MTP_PCC = 0.999
# Keyed by use_pretrained.
MTP_MODULE_OUTPUT_PCC = {False: 0.98, True: 0.96}
KVPE_PCC = 0.999

SP_AXIS, TP_AXIS = 0, 1


def _accumulated_pcc(base: float, upstream_levels: int, module_pcc: float) -> float:
    """``base``'s own PCC budget plus one block's worth of drift per upstream MTP level.

    MTP is a recurrence, so device/reference disagreement is inherited rather than reset.
    """
    return 1.0 - ((1.0 - base) + upstream_levels * (1.0 - module_pcc))


_MESH_PARAMS = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(
            fabric_payload_size=GLM52Config.FABRIC_PAYLOAD_SIZE,
            worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
        ),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
]


def _shard_dims():
    dims = [None, None]
    dims[TP_AXIS] = -1
    dims[SP_AXIS] = -2
    return dims


def _to_device(t: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """Upload ``[1, seq, hidden]`` as ``[1, 1, seq, hidden]``, SP over rows and TP over columns."""
    return ttnn.from_torch(
        t.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=_shard_dims()),
    )


def _from_device(t: ttnn.Tensor, mesh_device) -> torch.Tensor:
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=_shard_dims(), mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)


def _mtp_inputs(seq_len: int, hidden: int, seed: int = 7):
    """``(embed, hidden)`` for one MTP level."""
    g = torch.Generator().manual_seed(seed)
    embed = torch.randn(1, seq_len, hidden, generator=g, dtype=torch.float32).to(torch.bfloat16)
    hid = torch.randn(1, seq_len, hidden, generator=g, dtype=torch.float32).to(torch.bfloat16)
    return embed, hid


def _glm_norm_weight(hidden, seed):
    """Same random norm gain as tests/test_prefill_block.py:743 (copied, not imported: that module
    is a collected test, and importing one under its package path double-registers it in pytest)."""
    return (torch.randn(hidden, generator=torch.Generator().manual_seed(seed)) * 0.1 + 1.0).to(torch.bfloat16)


def _glm_random_moe_weights(hidden, moe_intermediate, n_routed, seed):
    """Mirrors tests/test_prefill_block.py:756 -- see :func:`_glm_norm_weight` for why it is copied."""
    g = torch.Generator().manual_seed(seed)
    hs, ds = hidden**-0.5, moe_intermediate**-0.5

    def _expert():
        return {
            "gate_proj": (torch.randn(moe_intermediate, hidden, generator=g) * hs).to(torch.bfloat16),
            "up_proj": (torch.randn(moe_intermediate, hidden, generator=g) * hs).to(torch.bfloat16),
            "down_proj": (torch.randn(hidden, moe_intermediate, generator=g) * ds).to(torch.bfloat16),
        }

    gate_weights = {
        "weight": (torch.randn(n_routed, hidden, generator=g) * hs).to(torch.bfloat16),
        "e_score_correction_bias": (torch.randn(n_routed, generator=g) * 0.01).to(torch.float32),
    }
    return gate_weights, [_expert() for _ in range(n_routed)], _expert()


def _mtp_level_inputs(num_levels: int, seq_len: int, hidden: int, seed: int = 7):
    """``(embeds, h0)`` for a K-level predictor: K shifted-token embeddings and the trunk hidden.

    Distinct seeds per level, so a loop that reused one embedding or chained the wrong tensor cannot
    pass by symmetry.
    """
    embeds = [_mtp_inputs(seq_len, hidden, seed=seed + k)[0] for k in range(num_levels)]
    _, h0 = _mtp_inputs(seq_len, hidden, seed=seed)
    return embeds, h0


def _glm52_config_for_mtp(config_only, seq_len: int, layer_idx: int):
    """A GLM-5.2 config with the MTP layer's indexer slot declared, safe to mutate.

    ``copy.copy`` because ``config_only`` is lru_cached; GLM-5.2's own map stops at the trunk.
    """
    config = copy.copy(config_only)
    config.max_seq_len = seq_len
    enable_mtp_indexer_slot(config, layer_idx)
    return config


def _glm_layer_weights(variant, config, layer_state_dict=None):
    """Layer-78 weights -- MLA + indexer, both layernorms, and the 256-expert MoE.

    One set drives the device, the CPU reference and every level. ``layer_state_dict`` is the
    checkpoint's real layer 78; ``None`` means seeded random weights instead.
    """
    if layer_state_dict is not None:
        moe_weights = {
            k: layer_state_dict[k] for k in ("gate_weights", "routed_expert_weights", "shared_expert_weights")
        }
        return (
            layer_state_dict["mla_weights"],
            layer_state_dict["attn_norm_weight"],
            layer_state_dict["ffn_norm_weight"],
            moe_weights,
            layer_state_dict,
        )

    hidden = config.hidden_size
    mla_weights, _ = build_weights(variant, config, seed=42)
    attn_norm_w, ffn_norm_w = _glm_norm_weight(hidden, 1), _glm_norm_weight(hidden, 2)
    gate_weights, routed, shared = _glm_random_moe_weights(
        hidden, GLM52Config.MOE_INTERMEDIATE_SIZE, GLM52Config.NUM_ROUTED_EXPERTS, seed=3
    )
    moe_weights = {"gate_weights": gate_weights, "routed_expert_weights": routed, "shared_expert_weights": shared}
    layer_state_dict = {
        "attn_norm_weight": attn_norm_w,
        "mla_weights": mla_weights,
        "ffn_norm_weight": ffn_norm_w,
        **moe_weights,
    }
    return mla_weights, attn_norm_w, ffn_norm_w, moe_weights, layer_state_dict


def _mtp_device_caches(config, mesh_device, seq_len: int, num_cache_slots: int):
    """``(kvpe_cache, rope_tensors, index_kv_cache)`` for a stack of ``num_cache_slots`` MTP levels."""
    mesh_shape = list(mesh_device.shape)
    # KV dedup: both caches are striped across SP*TP, exactly like the runner allocates them. The
    # sparse indexer always dedups, so an SP-only index cache has no TP axis for it to gather over.
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        num_kvpe_cache_layers=num_cache_slots,
    )
    rope_tensors = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False).get_rope_tensors_indexed(
        cache_seq_len_global=seq_len, chunk_size_global=seq_len
    )
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        num_kvpe_cache_layers=num_full_indexer_layers(config) or 1,
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )
    return kvpe_cache, rope_tensors, index_kv_cache


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)
@pytest.mark.timeout(0)
def test_fused_mtp_pcc(mesh_device, device_params, num_links, seq_len, use_pretrained, mtp_cfg, mtp_state_dict):
    """``TtFusedMTP`` vs ``fused_mtp_reference`` -- the only new math in the feature.

    Runs on real weights too: unlike the full module this touches no MoE gate. Uses no DSA op, so it
    is not Blackhole-gated.
    """
    topology = per_axis_topology(device_params["fabric_config"])
    hidden = mtp_cfg.hidden_size
    embed, hid = _mtp_inputs(seq_len, hidden)

    tp = mesh_device.shape[TP_AXIS]
    permuted = eh_proj_to_tt_layout(mtp_state_dict["eh_proj"], tp)
    block = permuted.shape[0] // tp
    for chip in range(tp):
        assert torch.equal(
            permuted[chip * block : (chip + 1) * block],
            eh_proj_expected_chip_shard(mtp_state_dict["eh_proj"], tp, chip),
        ), f"chip {chip} holds the wrong eh_proj rows"

    logger.info(f"[fused mtp] use_pretrained={use_pretrained} seq_len={seq_len} mesh={list(mesh_device.shape)}")
    fused = TtFusedMTP(
        mesh_device,
        mtp_cfg,
        mtp_state_dict,
        tp_axis=TP_AXIS,
        num_links=num_links,
        topology=topology,
    )
    tt_out = _from_device(fused.forward(_to_device(embed, mesh_device), _to_device(hid, mesh_device)), mesh_device)

    ref = fused_mtp_reference(
        embed,
        hid,
        mtp_state_dict["enorm"],
        mtp_state_dict["hnorm"],
        mtp_state_dict["eh_proj"],
        mtp_cfg.rms_norm_eps,
    )
    _, pcc_msg = assert_with_pcc(ref.unsqueeze(0), tt_out, FUSED_MTP_PCC)
    logger.info(f"[fused mtp] PCC: {pcc_msg}")
    ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("variant", ["glm_5_2"], indirect=True, ids=["glm52"])
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_mtp_module_pcc(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    seq_len,
    use_pretrained,
    mtp_cfg,
    mtp_state_dict,
    mtp_layer_state_dict,
):
    """``TtMTPModule`` (fused projection + the MTP layer + ``shared_head.norm``) vs the reference.

    Both weight options; ``test_mtp_transformer_chunks.py`` only covers this module behind the trunk.
    """
    topology = per_axis_topology(device_params["fabric_config"])
    mesh_shape = list(mesh_device.shape)
    layer_idx = mtp_cfg.mtp_layer_idx

    config = _glm52_config_for_mtp(config_only, seq_len, layer_idx)
    hidden = config.hidden_size
    assert hidden == mtp_cfg.hidden_size

    mla_weights, attn_norm_w, ffn_norm_w, moe_weights, layer_state_dict = _glm_layer_weights(
        variant, config, mtp_layer_state_dict
    )
    module_pcc = MTP_MODULE_OUTPUT_PCC[use_pretrained]

    logger.info(
        f"[mtp module] use_pretrained={use_pretrained} module_pcc={module_pcc} "
        f"building TtMTPModule layer_idx={layer_idx} seq_len={seq_len} mesh={mesh_shape}"
    )
    module = TtMTPModule(
        mesh_device,
        config,
        GLM52Config,
        {"mtp": mtp_state_dict, "layer": layer_state_dict},
        mtp_cfg,
        seq_len=seq_len,
        layer_idx=layer_idx,
        tp_axis=TP_AXIS,
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        layer_num=1,
    )

    kvpe_cache, rope_tensors, index_kv_cache = _mtp_device_caches(config, mesh_device, seq_len, 1)

    embed, hid = _mtp_inputs(seq_len, hidden)

    logger.info("[mtp module] running device module")
    tt_x, tt_out, tt_normed, *_ = module.forward(
        _to_device(embed, mesh_device),
        _to_device(hid, mesh_device),
        rope_tensors=rope_tensors,
        kvpe_cache=kvpe_cache,
        actual_isl=seq_len,
        index_kv_cache=index_kv_cache,
    )

    logger.info("[mtp module] composing CPU reference via reference.glm_5_2.glm_mtp_module_reference")
    ref_x, ref_out, ref_normed, _ = glm_mtp_module_reference(
        config,
        mla_weights,
        mtp_state_dict,
        attn_norm_w,
        ffn_norm_w,
        embed,
        hid,
        seq_len,
        moe_weights=moe_weights,
    )

    _, msg = assert_with_pcc(ref_x.unsqueeze(0), _from_device(tt_x, mesh_device), FUSED_MTP_PCC)
    logger.info(f"[mtp module] fused projection PCC: {msg}")
    _, msg = assert_with_pcc(ref_out.unsqueeze(0), _from_device(tt_out, mesh_device), module_pcc)
    logger.info(f"[mtp module] layer output PCC: {msg}")
    _, msg = assert_with_pcc(ref_normed.unsqueeze(0), _from_device(tt_normed, mesh_device), module_pcc)
    logger.info(f"[mtp module] shared_head.norm output PCC: {msg}")
    ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("num_levels", [1, 4, 7], ids=["levels1", "levels4", "levels7"])
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("variant", ["glm_5_2"], indirect=True, ids=["glm52"])
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_mtp_predictor_pcc(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    num_levels,
    seq_len,
    use_pretrained,
    mtp_cfg,
    mtp_state_dict,
    mtp_layer_state_dict,
):
    """``TtMTPPredictor`` at K = 1 and K = 4 vs ``glm_mtp_predictor_reference``, single galaxy.

    The per-slot KV assertions are the point: a level that wrote the wrong slot still produces the
    right output single-shot. Index sharing is asserted by object identity on the returned top-k.
    """
    topology = per_axis_topology(device_params["fabric_config"])
    mesh_shape = list(mesh_device.shape)
    layer_idx = mtp_cfg.mtp_layer_idx
    config = _glm52_config_for_mtp(config_only, seq_len, layer_idx)
    hidden = config.hidden_size
    assert hidden == mtp_cfg.hidden_size

    mla_weights, attn_norm_w, ffn_norm_w, moe_weights, layer_state_dict = _glm_layer_weights(
        variant, config, mtp_layer_state_dict
    )
    module_pcc = MTP_MODULE_OUTPUT_PCC[use_pretrained]

    logger.info(
        f"[mtp predictor] use_pretrained={use_pretrained} module_pcc={module_pcc} "
        f"building TtMTPPredictor K={num_levels} layer_idx={layer_idx} mesh={mesh_shape}"
    )
    predictor = TtMTPPredictor(
        mesh_device,
        config,
        GLM52Config,
        {"mtp": mtp_state_dict, "layer": layer_state_dict},
        mtp_cfg,
        seq_len=seq_len,
        num_levels=num_levels,
        layer_idx=layer_idx,
        tp_axis=TP_AXIS,
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        layer_num=num_levels,
    )

    kvpe_cache, rope_tensors, index_kv_cache = _mtp_device_caches(config, mesh_device, seq_len, num_levels)

    embeds, h0 = _mtp_level_inputs(num_levels, seq_len, hidden)

    logger.info(f"[mtp predictor] running {num_levels} device level(s), index_share={predictor.index_share}")
    res = predictor.forward(
        lambda k, _prev: _to_device(embeds[k], mesh_device),
        _to_device(h0, mesh_device),
        rope_tensors=rope_tensors,
        kvpe_cache=kvpe_cache,
        actual_isl=seq_len,
        index_kv_cache=index_kv_cache,
        return_kv_cache=True,
        return_indexer_indices=True,
    )

    if predictor.index_share and num_levels > 1:
        assert all(
            t is res.indexer_indices[0] for t in res.indexer_indices[1:]
        ), "index_share is on but a level ran its own indexer instead of attending at level 1's top-k"
    for tensor in {id(t): t for t in res.indexer_indices if t is not None}.values():
        ttnn.deallocate(tensor)

    logger.info("[mtp predictor] composing CPU reference (one 256-expert MoE block per level)")
    ref_xs, ref_outs, ref_normeds, ref_kv = glm_mtp_predictor_reference(
        config,
        mla_weights,
        mtp_state_dict,
        attn_norm_w,
        ffn_norm_w,
        embeds,
        h0,
        seq_len,
        moe_weights=moe_weights,
        num_levels=num_levels,
        index_share=predictor.index_share,
    )

    tt_kv = res.kv_cache
    assert tt_kv is not None, "return_kv_cache=True must produce the host KVPE cache"
    assert tt_kv.shape[0] == num_levels == ref_kv.shape[0], f"{tuple(tt_kv.shape)} vs {tuple(ref_kv.shape)}"
    kv_lora_rank = config.kv_lora_rank

    for k in range(num_levels):
        lvl = k + 1
        _, msg = assert_with_pcc(
            ref_xs[k].unsqueeze(0),
            _from_device(res.x[k], mesh_device),
            _accumulated_pcc(FUSED_MTP_PCC, k, module_pcc),
        )
        logger.info(f"[mtp predictor] L{lvl} fused projection PCC: {msg}")
        _, msg = assert_with_pcc(
            ref_outs[k].unsqueeze(0),
            _from_device(res.out[k], mesh_device),
            _accumulated_pcc(module_pcc, k, module_pcc),
        )
        logger.info(f"[mtp predictor] L{lvl} layer output PCC: {msg}")
        _, msg = assert_with_pcc(
            ref_normeds[k].unsqueeze(0),
            _from_device(res.out_head_normed[k], mesh_device),
            _accumulated_pcc(module_pcc, k, module_pcc),
        )
        logger.info(f"[mtp predictor] L{lvl} shared_head.norm output PCC: {msg}")

        ref_slot, tt_slot = ref_kv[k : k + 1], tt_kv[k : k + 1]
        kv_threshold = _accumulated_pcc(KVPE_PCC, k, module_pcc)
        _, kv_pcc = comp_pcc(ref_slot[..., :kv_lora_rank].float(), tt_slot[..., :kv_lora_rank].float())
        _, pe_pcc = comp_pcc(ref_slot[..., kv_lora_rank:].float(), tt_slot[..., kv_lora_rank:].float())
        logger.info(f"[mtp predictor] L{lvl} KVPE slot {k}: kv={kv_pcc:.6f} pe={pe_pcc:.6f} (thr {kv_threshold})")
        assert kv_pcc > kv_threshold, f"L{lvl} KVPE KV PCC {kv_pcc:.6f} below {kv_threshold}"
        assert pe_pcc > kv_threshold, f"L{lvl} KVPE PE PCC {pe_pcc:.6f} below {kv_threshold}"

    ttnn.synchronize_device(mesh_device)
    logger.success(f"[mtp predictor] K={num_levels} passed")

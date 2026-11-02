# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for GLM-5.3 MTP, single galaxy.

Four tests, most-local first: the fused projection alone, one whole MTP module, K levels over
that module with per-slot KV comparison, and those K levels across a multi-turn conversation on one
cache. Every test carries both weight options.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.runners.runner_utils import MTP_PAD_TOKEN_ID, num_mtp_tokens
from models.demos.deepseek_v3_d_p.reference.glm_5_3.mtp import (
    fused_mtp_reference,
    glm_mtp_module_reference,
    glm_mtp_predictor_reference,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests._reuse import acquire_module
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.mtp_prefill.conftest import glm53_checkpoint_path
from models.demos.deepseek_v3_d_p.tests.mtp_prefill.predictor_reference import (
    compute_predictor_reference,
    load_reference,
    predictor_inputs,
    prepare_store,
    save_reference,
    stored_reference_path,
)
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import build_weights
from models.demos.deepseek_v3_d_p.tt.mla.indexer import (
    full_indexer_rank,
    normalized_hadamard_matrix,
    num_full_indexer_layers,
)
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import (
    MTPDeviceEmbedSource,
    MTPSplitChipLookahead,
    MTPUnionEmbedding,
)
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.tt_mtp import TtFusedMTP, TtMTPModule, TtMTPPredictor
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import (
    MTP_CACHE_ENV,
    MTP_CACHE_PREFIX,
    eh_proj_expected_chip_shard,
    eh_proj_to_tt_layout,
    enable_mtp_indexer_slot,
)
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import (
    build_sp_rank_tensor,
    prepare_prefill_input_tensor,
    prepare_prefill_mtp_tokens,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_natural, unrotate_cache_layer
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc

FUSED_MTP_PCC = 0.999
# Keyed by use_pretrained.
MTP_MODULE_OUTPUT_PCC = {False: 0.98, True: 0.95}
KVPE_PCC = 0.999
INDEX_K_PCC = 0.999

SP_AXIS, TP_AXIS = 0, 1


def _accumulated_pcc(base: float, upstream_levels: int, module_pcc: float) -> float:
    """``base``'s own PCC budget plus one block's worth of drift per upstream MTP level.

    MTP is a recurrence, so device/reference disagreement is inherited rather than reset.
    """
    return 1.0 - ((1.0 - base) + upstream_levels * (1.0 - module_pcc))


# Keyed by use_pretrained; levels 2-4 of test_mtp_multiturn_pcc as (x, out and normed, KVPE nope, KVPE pe), centered.
# Each is the lowest over the 8 schedules at levels7 on bh-glx-120-c02u02 (2026-10-22) minus 0.02, rounded down.
MULTITURN_LEVEL_PCC = {
    False: [
        (0.97, 0.95, 0.97, 0.97),
        (0.96, 0.93, 0.96, 0.96),
        (0.95, 0.91, 0.95, 0.95),
    ],
    True: [
        (0.96, 0.92, 0.95, 0.97),
        (0.93, 0.83, 0.89, 0.94),
        (0.86, 0.72, 0.80, 0.91),
    ],
}


def _assert_level_pcc(expected, actual, pcc: float, k: int, what: str):
    """Plain PCC at level 1 (``k == 0``). Deeper free-running levels share a large common row, which plain PCC
    rewards even in an output that only repeats it, so there each column's mean over the rows is removed first."""
    if k > 0:
        expected, actual = expected.float(), actual.float()
        expected, actual, what = expected - expected.mean(0), actual - actual.mean(0), f"{what} centered"
    _, msg = assert_with_pcc(expected, actual, pcc)
    logger.info(f"{what} PCC: {msg}")


_MESH_PARAMS = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(
            fabric_payload_size=GLM53Config.FABRIC_PAYLOAD_SIZE,
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


def _glm53_config_for_mtp(config_only, seq_len: int, layer_idx: int):
    """A GLM-5.3 config with the MTP layer's indexer slot declared, safe to mutate.

    ``copy.copy`` because ``config_only`` is lru_cached; GLM-5.3's own map stops at the trunk.
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
        hidden, GLM53Config.MOE_INTERMEDIATE_SIZE, GLM53Config.NUM_ROUTED_EXPERTS, seed=3
    )
    moe_weights = {"gate_weights": gate_weights, "routed_expert_weights": routed, "shared_expert_weights": shared}
    layer_state_dict = {
        "attn_norm_weight": attn_norm_w,
        "mla_weights": mla_weights,
        "ffn_norm_weight": ffn_norm_w,
        **moe_weights,
    }
    return mla_weights, attn_norm_w, ffn_norm_w, moe_weights, layer_state_dict


def _mtp_device_weights(variant, mesh_device, layer_idx: int, use_pretrained: bool, mtp_sd: dict, layer_sd: dict):
    """``(state_dict, weight_cache_path)`` for an MTP constructor: cache-only, as the runner builds it, when
    ``$TT_GLM53_MTP_TTNN_CACHE`` holds a complete cache for this mesh. Otherwise the weights are converted and
    nothing is written, so a read-only shared tree stays untouched."""
    state_dict = {"mtp": mtp_sd, "layer": layer_sd}
    root = os.environ.get(MTP_CACHE_ENV)
    if not (use_pretrained and root):
        return state_dict, None
    mesh_shape = list(mesh_device.shape)
    path = (
        Path(root)
        / f"{variant.name}_{'bh' if is_blackhole() else 'wh'}_{ttnn.get_num_devices()}dev"
        / f"{mesh_shape[SP_AXIS]}x{mesh_shape[TP_AXIS]}"
    )
    if path.is_dir():
        init_checker(path)
        if TtMTPPredictor.check_cache_complete(
            path,
            layer_idx,
            cache_name_prefix=MTP_CACHE_PREFIX,
            experts_per_chip=GLM53Config.NUM_ROUTED_EXPERTS // mesh_device.get_num_devices(),
            model_cfg=GLM53Config,
        ):
            logger.info(f"MTP layer {layer_idx} weights: from the cache at {path}")
            return {"mtp": {}, "layer": {}}, path
    logger.warning(f"MTP layer {layer_idx} weights: no complete cache at {path}, converting the checkpoint's")
    return state_dict, None


def _mtp_device_caches(config, mesh_device, seq_len: int, num_cache_slots: int, chunk_size: int | None = None):
    """``(kvpe_cache, rope_tensors, index_kv_cache)`` for a stack of ``num_cache_slots`` MTP levels.
    ``chunk_size`` makes ``seq_len`` a cache that chunked forwards fill, roped as the transformer does."""
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
    chunked = chunk_size is not None
    rope_tensors = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False).get_rope_tensors_indexed(
        cache_seq_len_global=seq_len, chunk_size_global=chunk_size if chunked else seq_len, tail_slack=chunked
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
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
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

    config = _glm53_config_for_mtp(config_only, seq_len, layer_idx)
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
    device_weights, weight_cache_path = _mtp_device_weights(
        variant, mesh_device, layer_idx, use_pretrained, mtp_state_dict, layer_state_dict
    )
    module = TtMTPModule(
        mesh_device,
        config,
        GLM53Config,
        device_weights,
        mtp_cfg,
        seq_len=seq_len,
        layer_idx=layer_idx,
        tp_axis=TP_AXIS,
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=weight_cache_path,
        cache_name_prefix=MTP_CACHE_PREFIX,
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

    logger.info("[mtp module] composing CPU reference via reference.glm_5_3.glm_mtp_module_reference")
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
@pytest.mark.parametrize("num_levels", [1, 4], ids=["levels1", "levels4"])
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
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
    config = _glm53_config_for_mtp(config_only, seq_len, layer_idx)
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
    device_weights, weight_cache_path = _mtp_device_weights(
        variant, mesh_device, layer_idx, use_pretrained, mtp_state_dict, layer_state_dict
    )
    predictor = TtMTPPredictor(
        mesh_device,
        config,
        GLM53Config,
        device_weights,
        mtp_cfg,
        seq_len=seq_len,
        num_levels=num_levels,
        layer_idx=layer_idx,
        tp_axis=TP_AXIS,
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=weight_cache_path,
        cache_name_prefix=MTP_CACHE_PREFIX,
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


CHUNK = 5 * 1024
CACHE_LEN = 3 * CHUNK
REFERENCE_LEN = 11520  # the longest schedule, multichunk
MAX_LEVELS = 4
INPUT_SEED = 11
VOCAB = CACHE_LEN + ttnn.TILE_SIZE

# New tokens per turn. Each turn resumes at the previous end rounded down to a tile, as the engine replays.
_MULTITURN_TURNS = [
    pytest.param([640, 5120], id="aligned_min"),  # turn 1 starts on chip 1's first row
    pytest.param([672, 5120], id="midchip_straddle"),  # split chip 1 takes its lookahead from chip 2
    pytest.param([4480, 5120], id="lastchip"),  # turn 1 starts on the last chip's first row
    pytest.param([1280, 1920, 5120], id="rot_partial"),  # three turns, each starting on a chip boundary
    pytest.param([5120, 1280, 5120], id="multichunk"),  # spans three cache chunks
    pytest.param([4512, 5100], id="split_wrap_gather"),  # split chip 7 takes chip 0's rows; ends inside a tile
    pytest.param([608, 5100], id="split_one_tile"),  # split chip 0 keeps one tile of its first run
    pytest.param([3300, 500, 542], id="replay_paths"),  # 4- then 24-row replays: no lookahead, then own slots
]


def _turn_chunks(turns) -> list[tuple[int, int]]:
    """``(start, end)`` of every forward: each turn resumes at the previous end rounded down to a tile and runs to
    its own end in ``CHUNK`` pieces."""
    chunks, end = [], 0
    for new_tokens in turns:
        resume, end = end // ttnn.TILE_SIZE * ttnn.TILE_SIZE, end + new_tokens
        chunks += [(start, min(start + CHUNK, end)) for start in range(resume, end, CHUNK)]
    return chunks


# The last reference used, kept for the next case: no test may mutate it.
_MULTITURN_REFERENCES: dict = {}


def _multiturn_reference(
    config, mtp_sd, layer_sd, use_pretrained: bool, layer_idx: int, index_share: bool, levels: int
):
    """The stored reference when this checkpoint has one, else computed once per process: every level, and stored,
    when ``$TT_GLM53_MTP_REF_CACHE`` is writable. Either has at least ``levels`` levels, and levels 1..K of a longer
    run are the K-level run."""
    key = (use_pretrained, layer_idx, index_share)
    ref = _MULTITURN_REFERENCES.get(key)
    if ref is not None and ref["x"].shape[0] >= levels:
        return ref
    args = dict(
        index_share=index_share, seed=INPUT_SEED, vocab=VOCAB, seq_len=REFERENCE_LEN, chunk=CHUNK, cache_len=CACHE_LEN
    )
    root = os.environ.get("TT_GLM53_MTP_REF_CACHE")
    path = None
    if use_pretrained and root:
        path = stored_reference_path(
            root, "glm53_mtp_multiturn_ref", glm53_checkpoint_path(), layer_idx, num_levels=MAX_LEVELS, **args
        )
    if path is not None and path.exists():
        logger.info(f"[mtp multiturn] CPU reference stored at {path}")
        ref = load_reference(path)
    else:
        store = path is not None and prepare_store(path)
        if store:
            logger.warning(f"[mtp multiturn] no stored CPU reference at {path}, computing every level to store it")
            levels = MAX_LEVELS
        elif path is not None:
            logger.warning(f"[mtp multiturn] no stored CPU reference at {path}, and {path.parent} is not writable")
        logger.info(
            f"[mtp multiturn] CPU reference, {levels} level(s) over [0, {REFERENCE_LEN}) in {CHUNK}-token chunks"
        )
        ref = compute_predictor_reference(config, mtp_sd, layer_sd, num_levels=levels, **args)
        if store:
            try:
                save_reference(ref, path)
                logger.info(f"[mtp multiturn] CPU reference stored at {path}")
            except (OSError, RuntimeError) as e:  # torch.save raises RuntimeError when it cannot write
                logger.warning(f"[mtp multiturn] could not store the CPU reference at {path}: {e}")
    _MULTITURN_REFERENCES.clear()
    _MULTITURN_REFERENCES[key] = ref
    return ref


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("turns", _MULTITURN_TURNS)
@pytest.mark.parametrize("num_levels", [1, 4], ids=["levels1", "levels4"])
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_mtp_multiturn_pcc(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    turns,
    num_levels,
    use_pretrained,
    mtp_cfg,
    mtp_state_dict,
    mtp_layer_state_dict,
):
    """``TtMTPPredictor`` over a multi-turn conversation on one cache vs a whole-conversation reference.
    Free-running: the real next tokens; each level's hidden is the level below's output, on the device and in
    the reference."""
    topology = per_axis_topology(device_params["fabric_config"])
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[SP_AXIS]
    layer_idx = mtp_cfg.mtp_layer_idx
    config = _glm53_config_for_mtp(config_only, CACHE_LEN, layer_idx)
    hidden = config.hidden_size
    assert hidden == mtp_cfg.hidden_size
    chunks = _turn_chunks(turns)
    total = chunks[-1][1]
    assert chunks[-1][0] + CHUNK <= CACHE_LEN, f"turns {turns} write past the {CACHE_LEN}-token cache"
    assert total <= REFERENCE_LEN and num_levels <= MAX_LEVELS, f"turns {turns} at K={num_levels} outgrow the reference"

    *_, layer_state_dict = _glm_layer_weights(variant, config, mtp_layer_state_dict)
    module_pcc = MTP_MODULE_OUTPUT_PCC[use_pretrained]
    logger.info(f"[mtp multiturn] use_pretrained={use_pretrained} K={num_levels} turns={turns} chunks={chunks}")
    device_weights, weight_cache_path = _mtp_device_weights(
        variant, mesh_device, layer_idx, use_pretrained, mtp_state_dict, layer_state_dict
    )
    # The turn schedule is the only axis the predictor does not depend on, so its cases share one build.
    predictor = acquire_module(
        mesh_device,
        ("TtMTPPredictor multiturn", variant.name, use_pretrained, num_levels, num_links, CHUNK, CACHE_LEN, layer_idx),
        lambda: TtMTPPredictor(
            mesh_device,
            config,
            GLM53Config,
            device_weights,
            mtp_cfg,
            seq_len=CHUNK,
            num_levels=num_levels,
            layer_idx=layer_idx,
            tp_axis=TP_AXIS,
            num_links=num_links,
            topology=topology,
            sp_axis=SP_AXIS,
            gate_fallback_mode=GateComputeMode.DEVICE_FP32,
            # Every padding row is identical, so all of them can route to one chip's experts.
            dispatch_buffer_capacity_factor=8,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=MTP_CACHE_PREFIX,
            is_chunked=True,
            max_seq_len=CACHE_LEN,
            slot_num=1,
            layer_num=num_levels,
        ),
        release=lambda p: p.module.layer.release_sub_device_managers(),
    )
    kvpe_cache, rope_tensors, index_kv_cache = _mtp_device_caches(
        config, mesh_device, CACHE_LEN, num_levels, chunk_size=CHUNK
    )
    mesh_device.enable_program_cache()

    table, h0, ids = predictor_inputs(hidden, VOCAB, REFERENCE_LEN, MAX_LEVELS, INPUT_SEED)
    # A pad id is clamped to the table's last row, which no real id may share.
    assert ids[-1] < VOCAB - 1
    embedding = TtParallelEmbedding(mesh_device, VOCAB, hidden, torch_weight=table, sp_axis=SP_AXIS, tp_axis=TP_AXIS)
    sp_rank = build_sp_rank_tensor(mesh_device, sp, mesh_shape, SP_AXIS)
    n_mtp = num_mtp_tokens(num_levels)

    def all_gather_sp(rows):
        return ttnn.all_gather(rows, dim=-2, cluster_axis=SP_AXIS, num_links=num_links, topology=topology[SP_AXIS])

    dev_chunks, done = [], 0
    for i, (start, end) in enumerate(chunks):
        stream = [ids[p] if p < end + num_levels else MTP_PAD_TOKEN_ID for p in range(start, start + CHUNK + n_mtp)]
        chunk_ids = prepare_prefill_input_tensor(
            stream[:CHUNK], mesh_device, sp, False, mesh_shape, SP_AXIS, chunk_start=start
        )
        mtp_ids = prepare_prefill_mtp_tokens(
            stream,
            mesh_device,
            sp,
            mesh_shape,
            SP_AXIS,
            num_mtp_tokens=n_mtp,
            num_levels=num_levels,
            chunk_start=start,
            chunk_end=end,
        )
        union = MTPUnionEmbedding.from_ids(
            chunk_ids, mtp_ids, lambda t: ttnn.unsqueeze_to_4D(embedding(t)), num_levels=num_levels
        )
        ttnn.deallocate(chunk_ids)
        ttnn.deallocate(mtp_ids)

        positions = torch.tensor(rotated_chip_positions(start, sp, CHUNK // sp)).flatten()
        h_rows = h0[positions.clamp(max=total - 1)]
        h_rows[positions >= end] = 0
        h_tt = _to_device(h_rows.unsqueeze(0), mesh_device)
        lookahead = MTPSplitChipLookahead.for_chunk_start(
            start, CHUNK // sp, sp, sp_rank, all_gather_sp, chunk_end=end, num_levels=num_levels
        )
        if lookahead is None:
            path = "no split-chip lookahead"
        elif lookahead.next_chip is None:
            path = "split-chip lookahead from its own slots"
        else:
            path = f"split-chip lookahead gathered from chip {lookahead.next_chip}"
        logger.info(f"[mtp multiturn] chunk {i}: [{start}, {end}) replays {max(0, done - start)} row(s); {path}")
        done = max(done, end)
        try:
            union.set_split_chip_lookahead(lookahead)
            res = predictor.forward(
                MTPDeviceEmbedSource(union, provided_levels=num_levels),
                h_tt,
                rope_tensors,
                kvpe_cache,
                actual_isl=end - start,
                actual_start=start,
                actual_end=end,
                cache_user_id=0,
                padding_side="right",
                index_kv_cache=index_kv_cache,
            )
        finally:
            if lookahead is not None:
                union.set_split_chip_lookahead(None)
                lookahead.deallocate()
        union.deallocate()
        ttnn.deallocate(h_tt)

        # Rows the next chunk replays are kept from that chunk only: its write is the one the cache keeps.
        stop = chunks[i + 1][0] if i + 1 < len(chunks) else end
        outputs = []
        for level_outputs in (res.x, res.out, res.out_head_normed):
            rows = [_from_device(t, mesh_device)[0, 0] for t in level_outputs]
            outputs.append([unrotate_cache_layer(r, positions - start, stop - start) for r in rows])
            for t in level_outputs:
                ttnn.deallocate(t)
        dev_chunks.append((start, stop, *outputs))
    ttnn.deallocate(sp_rank)

    t0 = time.monotonic()
    ref = _multiturn_reference(
        config, mtp_state_dict, layer_state_dict, use_pretrained, layer_idx, predictor.index_share, num_levels
    )
    logger.info(f"[mtp multiturn] CPU reference took {time.monotonic() - t0:.1f}s")

    limits = [(FUSED_MTP_PCC, module_pcc, KVPE_PCC, KVPE_PCC), *MULTITURN_LEVEL_PCC[use_pretrained]]
    for i, (start, end, xs, outs, normeds) in enumerate(dev_chunks):
        for k in range(num_levels):
            x_pcc, out_pcc, _, _ = limits[k]
            what = f"[mtp multiturn] chunk {i} L{k + 1}"
            _assert_level_pcc(ref["x"][k, start:end], xs[k], x_pcc, k, f"{what} fused projection")
            _assert_level_pcc(ref["out"][k, start:end], outs[k], out_pcc, k, f"{what} layer output")
            _assert_level_pcc(ref["normed"][k, start:end], normeds[k], out_pcc, k, f"{what} shared_head.norm output")

    # Whole slots, so a row a later turn failed to overwrite fails too; nothing at or past the last tile is written.
    tail = -(-total // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    kv_lora_rank = config.kv_lora_rank
    kvpe, stripes = gather_cache_natural(kvpe_cache.storage, mesh_device, tp_shard_kv=True)
    cache_positions = blockcyclic_positions(stripes, CHUNK, kvpe.shape[1])
    for k in range(num_levels):
        cache = unrotate_cache_layer(kvpe[k], cache_positions, kvpe.shape[1])
        expected = ref["kvpe"][k, :total]
        _, _, nope_pcc, pe_pcc = limits[k]
        for part, cols, pcc in (
            ("nope", slice(None, kv_lora_rank), nope_pcc),
            ("pe", slice(kv_lora_rank, None), pe_pcc),
        ):
            _assert_level_pcc(expected[:, cols], cache[:total, cols], pcc, k, f"[mtp multiturn] L{k + 1} KVPE {part}")
        assert torch.count_nonzero(cache[tail:]) == 0, f"L{k + 1} KVPE has rows written at or past {tail}"
    if predictor.index_share:
        index, stripes = gather_cache_natural(index_kv_cache, mesh_device, tp_shard_kv=True)
        raw = unrotate_cache_layer(
            index[full_indexer_rank(config, layer_idx)],
            blockcyclic_positions(stripes, CHUNK, index.shape[1]),
            index.shape[1],
        )
        index_k = raw @ normalized_hadamard_matrix(raw.shape[-1]).float()
        _, msg = assert_with_pcc(ref["index_k"][:total], index_k[:total], INDEX_K_PCC)
        logger.info(f"[mtp multiturn] L1 index-K PCC: {msg}")
        assert torch.count_nonzero(raw[tail:]) == 0, f"the index cache has rows written at or past {tail}"

    ttnn.synchronize_device(mesh_device)
    logger.success(f"[mtp multiturn] K={num_levels} turns={turns} passed")

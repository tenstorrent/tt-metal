# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for GLM-5.2 MTP (issue #53533), single galaxy.

Four tests, most-local first, so a failure localises itself:

1. ``test_fused_mtp_pcc`` — ``TtFusedMTP`` vs ``fused_mtp_reference``. The only new math in the
   feature, on both random and real weights. It opens with a host-side check of the ``eh_proj`` TP
   shard permutation, which fails per chip before any device op — end-to-end PCC alone would say
   that something is wrong but not which chip.
2. ``test_mtp_module_pcc`` — the whole ``TtMTPModule`` (fused projection + layer 78 + ``shared_head.norm``)
   vs ``glm_mtp_module_reference``. Its ``index_kv_cache`` sizing is what covers the MTP layer's
   ``indexer_types`` slot; see the comment at that call.

3. ``test_mtp_predictor_pcc`` — ``TtMTPPredictor`` at K = 1 and K = 4 vs
   ``glm_mtp_predictor_reference``. Adds what a single level cannot cover: the K-level recurrence,
   and **per-slot KV cache** comparison. The KV check is not optional here — a level that wrote the
   wrong slot still produces the right *output* (single-shot prefill reads back only what that same
   call just wrote), so nothing else catches a slot collision.
4. ``test_mtp_predictor_index_share`` — that ``index_share`` reaches the hardware, by object
   identity on the returned top-k rather than by a diluted numerical differential. Needs no CPU
   reference, so it costs device time only.

Structure follows ``tests/dflash_prefill/test_dflash.py``; the GLM device plumbing (rope, KV caches,
sharding, thresholds) follows ``tests/test_prefill_block.py::test_glm_prefill_block``.
"""

from __future__ import annotations

import copy

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_2.mtp import (
    CHAIN_FROM_NORM,
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

# Two distributed RMSNorms and one matmul — the same op class as tests/pcc/test_rmsnorm.py and
# test_ffn.py, so it earns their threshold rather than a block-level one.
FUSED_MTP_PCC = 0.999
# The MTP layer is an ordinary GLM MoE decoder block, so it earns the GLM block threshold measured in
# tests/test_prefill_block.py:729 (GLM_BLOCK_OUTPUT_PCC), not DeepSeek's PrefillBlockThresholds and
# not the ~0.93 whole-model KV floor.
MTP_MODULE_OUTPUT_PCC = 0.98
# The KVPE cache is written by the same ttMLA op whatever the model variant, so it earns the value
# tests/test_prefill_block.py:74 measured for it (PrefillBlockThresholds.kvpe_kv / kvpe_pe).
KVPE_PCC = 0.999

SP_AXIS, TP_AXIS = 0, 1


def _accumulated_pcc(base: float, upstream_levels: int) -> float:
    """``base``'s own PCC budget plus one block's worth of drift per upstream MTP level.

    MTP is a recurrence: level k's hidden input is level k-1's output, so device/reference
    disagreement is inherited, not reset. The rate is one ``MTP_MODULE_OUTPUT_PCC`` deficit per
    level, which at S = 5120 is ~2% of rows (PCC ~= 1 - wrong_rows/seq_len, and the known mechanism
    is the MoE gate picking a different top-8 for a few rows on synthetic input, which is row-local).

    This is a stated *model*, not a measurement — it is what makes a first bring-up run informative
    instead of red for a threshold reason. Every level's actual PCC is logged; replace this with the
    measured curve once there is one. ``upstream_levels = 0`` reproduces the Stage-1 thresholds
    exactly, so level 1 stays a true regression gate on ``test_mtp_module_pcc``.
    """
    return 1.0 - ((1.0 - base) + upstream_levels * (1.0 - MTP_MODULE_OUTPUT_PCC))


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
    """``(embed, hidden)`` for one MTP level, with position 0's embedding already zeroed.

    The zeroing is the caller's job on device (only the caller knows absolute positions under SP), so
    the test does it host-side before uploading and the reference sees the identical tensor.
    """
    g = torch.Generator().manual_seed(seed)
    embed = torch.randn(1, seq_len, hidden, generator=g, dtype=torch.float32).to(torch.bfloat16)
    hid = torch.randn(1, seq_len, hidden, generator=g, dtype=torch.float32).to(torch.bfloat16)
    embed[:, 0, :] = 0  # absolute position 0: nothing preceded it. Mirrors vLLM deepseek_mtp.py.
    return embed, hid


def _glm_norm_weight(hidden, seed):
    """Same random norm gain as tests/test_prefill_block.py:743 (copied, not imported: that module
    is a collected test, and importing one under its package path double-registers it in pytest)."""
    return (torch.randn(hidden, generator=torch.Generator().manual_seed(seed)) * 0.1 + 1.0).to(torch.bfloat16)


def _glm_random_moe_weights(hidden, moe_intermediate, n_routed, seed):
    """Mirrors tests/test_prefill_block.py:756 — see :func:`_glm_norm_weight` for why it is copied."""
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

    Distinct seeds per level, so a loop that reused one embedding, or chained the wrong tensor,
    cannot pass by symmetry. Position 0 is zeroed on EVERY level's embedding — vLLM zeroes at
    position 0 for all k, not just k = 1 — via :func:`_mtp_inputs`, whose docstring says why the
    caller owns it.
    """
    embeds = [_mtp_inputs(seq_len, hidden, seed=seed + k)[0] for k in range(num_levels)]
    _, h0 = _mtp_inputs(seq_len, hidden, seed=seed)
    return embeds, h0


def _glm52_config_for_mtp(config_only, seq_len: int, layer_idx: int):
    """A GLM-5.2 config with the MTP layer's indexer slot declared, safe to mutate.

    copy.copy, not the shared object: config_only is lru_cached, and enable_mtp_indexer_slot rebinds
    indexer_types — a 79-entry map left behind would follow every other GLM-5.2 test in the session.

    Declaring the slot is not optional setup: GLM-5.2's indexer_types covers layers 0..77 only, so
    without it the MTP layer's compacted index-cache slot is 21 of 21 — one past the end — and the
    index_kv_cache sized below is a slot short. mtp_indexer_types' docstring in
    tt/mtp_prefill/utils.py has the detail, including why indexer_layer_is_reused/full_indexer_rank
    look correct regardless.
    """
    config = copy.copy(config_only)
    config.max_seq_len = seq_len
    enable_mtp_indexer_slot(config, layer_idx)
    return config


def _glm_layer_weights(variant, config):
    """Random layer-78 weights — MLA + indexer, both layernorms, and the 256-expert MoE.

    One set drives the device and the CPU reference alike, and (for the predictor) every level: MTP
    is K activations over ONE weight module.
    """
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
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=num_cache_slots,
    )
    rope_tensors = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False).get_rope_tensors_indexed(
        cache_seq_len_global=seq_len, chunk_size_global=seq_len
    )
    # Strided by the compacted full-indexer count. With the MTP layer declared this is 22, and the
    # module writes slot 21; without the declaration it would be 21 and the write would run off the
    # end. ONE slot covers the whole MTP stack however many levels it has: TtIndexer._cache_slot
    # derives the slot from the block's static layer_idx, not from cache_layer_idx.
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=num_full_indexer_layers(config) or 1,
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )
    return kvpe_cache, rope_tensors, index_kv_cache


# ---------------------------------------------------------------------------
# Device: the fused projection alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)
@pytest.mark.timeout(0)
def test_fused_mtp_pcc(mesh_device, device_params, num_links, seq_len, use_pretrained, mtp_cfg, mtp_state_dict):
    """``TtFusedMTP`` vs ``fused_mtp_reference`` — the only new math in the feature.

    Runs on real weights too: unlike the full module this touches no MoE gate, so the pretrained leg
    is meaningful (see ``test_mtp_module_pcc``'s docstring for why the module test cannot).
    Uses no DSA op, so it is not Blackhole-gated.
    """
    topology = per_axis_topology(device_params["fabric_config"])
    hidden = mtp_cfg.hidden_size
    embed, hid = _mtp_inputs(seq_len, hidden)

    # The eh_proj TP shard, checked on the host before any device op. enorm/hnorm each emit the chip's
    # own slice of the *global* hidden, so chip c's concatenated activation covers two disjoint global
    # column ranges — not the contiguous block a plain dims=(None, -2) mapper hands it. Getting that
    # wrong yields a tensor of exactly the right shape holding exactly the wrong rows: no error, and a
    # collapsed PCC below that names no chip. eh_proj_expected_chip_shard slices the original [H, 2H]
    # weight rather than restating the permutation, so this is a cross-check, not a tautology.
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


# ---------------------------------------------------------------------------
# Device: the whole module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("variant", ["glm_5_2"], indirect=True, ids=["glm52"])
@pytest.mark.parametrize("use_pretrained", [False], ids=["random"], indirect=True)
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
):
    """``TtMTPModule`` (fused projection + layer 78 + ``shared_head.norm``) vs the composed CPU reference.

    **Random weights only, deliberately.** Layer 78 is a 256-expert MoE layer, and
    ``test_glm_prefill_block`` measured that a trained GLM gate driven by a synthetic input picks
    different top-8 experts on device than on CPU, collapsing isolated-block PCC to ~0.1 while the
    same layer scores ~0.995 in context. That is a real-weight-gate x synthetic-input artifact, not
    an op or weight bug, and it applies verbatim here — an MTP module's input is
    ``eh_proj(cat[...])`` of two random tensors. Real-weight coverage of the MTP-specific math is
    ``test_fused_mtp_pcc[pretrained]``; real-weight MoE coverage lives in the transformer tests.

    There is no separate KVPE-cache assertion: in single-shot prefill sparse SDPA reads the cache
    this same block just wrote, so a corrupted KV shows up in the block output compared here.
    """
    topology = per_axis_topology(device_params["fabric_config"])
    mesh_shape = list(mesh_device.shape)
    layer_idx = mtp_cfg.mtp_layer_idx

    config = _glm52_config_for_mtp(config_only, seq_len, layer_idx)
    hidden = config.hidden_size
    assert hidden == mtp_cfg.hidden_size

    # --- weights: random for device and reference alike ---
    mla_weights, attn_norm_w, ffn_norm_w, moe_weights, layer_state_dict = _glm_layer_weights(variant, config)

    # --- device module ---
    logger.info(f"[mtp module] building TtMTPModule layer_idx={layer_idx} seq_len={seq_len} mesh={mesh_shape}")
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
        # Single-block test: layer_num=1 gives the single-shot cache write a valid layer count.
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

    # Most-local first: a failure at #1 is the projection, at #2 the layer, at #3 shared_head.norm.
    _, msg = assert_with_pcc(ref_x.unsqueeze(0), _from_device(tt_x, mesh_device), FUSED_MTP_PCC)
    logger.info(f"[mtp module] fused projection PCC: {msg}")
    _, msg = assert_with_pcc(ref_out.unsqueeze(0), _from_device(tt_out, mesh_device), MTP_MODULE_OUTPUT_PCC)
    logger.info(f"[mtp module] layer output PCC: {msg}")
    _, msg = assert_with_pcc(ref_normed.unsqueeze(0), _from_device(tt_normed, mesh_device), MTP_MODULE_OUTPUT_PCC)
    logger.info(f"[mtp module] shared_head.norm output PCC: {msg}")
    ttnn.synchronize_device(mesh_device)


# ---------------------------------------------------------------------------
# Device: K levels over one module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("num_levels", [1, 4], ids=["levels1", "levels4"])
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("variant", ["glm_5_2"], indirect=True, ids=["glm52"])
@pytest.mark.parametrize("use_pretrained", [False], ids=["random"], indirect=True)
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
):
    """``TtMTPPredictor`` at K = 1 and K = 4 vs ``glm_mtp_predictor_reference``, single galaxy.

    Two legs, both cheap to justify:

    * **K = 1** is the regression gate. It runs the predictor over the exact path
      ``test_mtp_module_pcc`` already validated, so if it moves, the loop broke something rather than
      the recurrence being hard.
    * **K = 4** is the shipping config (#53533) and the only leg that can expose a slot-mapping bug.

    Random weights only, for the reason ``test_mtp_module_pcc``'s docstring gives: a trained GLM gate
    on synthetic input picks different top-8 experts on device than on CPU.

    **The KV assertions are the point of this test.** Every other comparison here is a deeper version
    of one the module test already makes, but a level that wrote the wrong KV slot still produces the
    right *output*: in single-shot prefill each level writes the full row range of its slot and reads
    that same slot back inside the same call, so a collision is invisible everywhere except in the
    cache itself. ``num_kvpe_cache_layers=num_levels`` plus ``layer_num=num_levels`` makes slot k-1
    land at batch index k-1 (``cache_batch_idx = cache_user_id * layer_num + cache_layer_idx`` with
    ``cache_user_id = 0``), which is the layout ``glm_mtp_predictor_reference`` stacks to match.

    Index sharing is on (GLM-5.2's ``index_share_for_mtp_iteration``) and the reference is told so:
    at seq 5120 with ``index_topk`` 2048 the top-k is selective on ~60% of rows, so a reference that
    recomputed per level would disagree on most of the sequence for a reason that is not a bug. That
    the flag reaches the hardware at all is ``test_mtp_predictor_index_share``'s job.
    """
    topology = per_axis_topology(device_params["fabric_config"])
    mesh_shape = list(mesh_device.shape)
    layer_idx = mtp_cfg.mtp_layer_idx
    # Pinned locally and passed to BOTH sides: a default that changed on one side only would show up
    # as a numerical failure at level 2 rather than as the configuration mismatch it is.
    chain_from = CHAIN_FROM_NORM

    config = _glm52_config_for_mtp(config_only, seq_len, layer_idx)
    hidden = config.hidden_size
    assert hidden == mtp_cfg.hidden_size

    mla_weights, attn_norm_w, ffn_norm_w, moe_weights, layer_state_dict = _glm_layer_weights(variant, config)

    logger.info(f"[mtp predictor] building TtMTPPredictor K={num_levels} layer_idx={layer_idx} mesh={mesh_shape}")
    predictor = TtMTPPredictor(
        mesh_device,
        config,
        GLM52Config,
        {"mtp": mtp_state_dict, "layer": layer_state_dict},
        mtp_cfg,
        seq_len=seq_len,
        num_levels=num_levels,
        layer_idx=layer_idx,
        chain_from=chain_from,
        tp_axis=TP_AXIS,
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        # K levels share one block, so the block's "model layer count" is the MTP stack depth: it is
        # what turns cache_layer_idx into the flat KV slot.
        layer_num=num_levels,
    )

    kvpe_cache, rope_tensors, index_kv_cache = _mtp_device_caches(config, mesh_device, seq_len, num_levels)

    embeds, h0 = _mtp_level_inputs(num_levels, seq_len, hidden)

    logger.info(f"[mtp predictor] running {num_levels} device level(s), index_share={predictor.index_share}")
    res = predictor.forward(
        [_to_device(e, mesh_device) for e in embeds],
        _to_device(h0, mesh_device),
        rope_tensors=rope_tensors,
        kvpe_cache=kvpe_cache,
        actual_isl=seq_len,
        index_kv_cache=index_kv_cache,
        return_kv_cache=True,
    )

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
        chain_from=chain_from,
    )

    tt_kv = res.kv_cache  # host torch already, [K, 1, seq, kv_lora_rank + qk_rope_head_dim]
    assert tt_kv is not None, "return_kv_cache=True must produce the host KVPE cache"
    assert tt_kv.shape[0] == num_levels == ref_kv.shape[0], f"{tuple(tt_kv.shape)} vs {tuple(ref_kv.shape)}"
    kv_lora_rank = config.kv_lora_rank

    for k in range(num_levels):
        lvl = k + 1  # 1-based, as in the recurrence
        # Most-local first within a level, and each threshold carries k levels of inherited drift.
        _, msg = assert_with_pcc(
            ref_xs[k].unsqueeze(0), _from_device(res.x[k], mesh_device), _accumulated_pcc(FUSED_MTP_PCC, k)
        )
        logger.info(f"[mtp predictor] L{lvl} fused projection PCC: {msg}")
        _, msg = assert_with_pcc(
            ref_outs[k].unsqueeze(0), _from_device(res.out[k], mesh_device), _accumulated_pcc(MTP_MODULE_OUTPUT_PCC, k)
        )
        logger.info(f"[mtp predictor] L{lvl} layer output PCC: {msg}")
        _, msg = assert_with_pcc(
            ref_normeds[k].unsqueeze(0),
            _from_device(res.out_head_normed[k], mesh_device),
            _accumulated_pcc(MTP_MODULE_OUTPUT_PCC, k),
        )
        logger.info(f"[mtp predictor] L{lvl} shared_head.norm output PCC: {msg}")

        # Slot k, split the way tests/test_prefill_block.py splits it: the latent and the RoPE halves
        # fail in different ways, and a merged PCC lets a healthy latent hide a broken k_pe.
        ref_slot, tt_slot = ref_kv[k : k + 1], tt_kv[k : k + 1]
        kv_threshold = _accumulated_pcc(KVPE_PCC, k)
        _, kv_pcc = comp_pcc(ref_slot[..., :kv_lora_rank].float(), tt_slot[..., :kv_lora_rank].float())
        _, pe_pcc = comp_pcc(ref_slot[..., kv_lora_rank:].float(), tt_slot[..., kv_lora_rank:].float())
        logger.info(f"[mtp predictor] L{lvl} KVPE slot {k}: kv={kv_pcc:.6f} pe={pe_pcc:.6f} (thr {kv_threshold})")
        assert kv_pcc > kv_threshold, f"L{lvl} KVPE KV PCC {kv_pcc:.6f} below {kv_threshold}"
        assert pe_pcc > kv_threshold, f"L{lvl} KVPE PE PCC {pe_pcc:.6f} below {kv_threshold}"

    ttnn.synchronize_device(mesh_device)
    logger.success(f"[mtp predictor] K={num_levels} passed (chain_from={chain_from})")


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("variant", ["glm_5_2"], indirect=True, ids=["glm52"])
@pytest.mark.parametrize("use_pretrained", [False], ids=["random"], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_mtp_predictor_index_share(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    seq_len,
    use_pretrained,
    mtp_cfg,
    mtp_state_dict,
):
    """``index_share`` reaches the hardware — checked by object identity, not by a PCC differential.

    ttMLA short-circuits on injected indices (``indices = indexer_indices if indexer_indices is not
    None else self._indexer.forward(...)``) and returns that same object, so with sharing on level
    2's returned top-k **is** level 1's tensor and with it off the two are distinct. That is exact:
    it cannot flake, and it costs no CPU reference.

    A numerical share/no-share differential was the obvious alternative and is a bad test here. The
    block output is ``x + mla_out + ffn_out`` with an identical residual ``x`` in both runs, so even
    a materially different top-k is diluted by the residual and the MoE path; the gap would land at
    some unknown PCC that any threshold either sleeps through or flakes on. The level-2 PCC is still
    logged below — as an observation of how much sharing changes, with no assertion on it.

    Two levels are enough: sharing is a level-1-to-everyone-else relation, and level 2 is the first
    consumer. One predictor is built and run twice, since ``index_share`` is pure runtime policy —
    nothing about the built block depends on it.
    """
    topology = per_axis_topology(device_params["fabric_config"])
    layer_idx = mtp_cfg.mtp_layer_idx
    num_levels = 2

    config = _glm52_config_for_mtp(config_only, seq_len, layer_idx)
    hidden = config.hidden_size

    _, _, _, _, layer_state_dict = _glm_layer_weights(variant, config)

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

    def _run(share: bool):
        return predictor.forward(
            [_to_device(e, mesh_device) for e in embeds],
            _to_device(h0, mesh_device),
            rope_tensors=rope_tensors,
            kvpe_cache=kvpe_cache,
            actual_isl=seq_len,
            index_kv_cache=index_kv_cache,
            index_share=share,
            return_indexer_indices=True,
        )

    logger.info("[mtp index share] run A: index_share=True")
    shared = _run(True)
    logger.info("[mtp index share] run B: index_share=False")
    unshared = _run(False)

    for name, res in (("shared", shared), ("unshared", unshared)):
        assert res.indexer_indices is not None and len(res.indexer_indices) == num_levels, name
        assert all(t is not None for t in res.indexer_indices), f"{name}: a level returned no top-k"

    assert (
        shared.indexer_indices[1] is shared.indexer_indices[0]
    ), "index_share=True: level 2 ran its own indexer instead of attending at level 1's top-k"
    assert (
        unshared.indexer_indices[1] is not unshared.indexer_indices[0]
    ), "index_share=False: level 2 reused level 1's top-k, so the flag is being ignored"

    # Level 1 computes its own top-k either way and both runs feed it identical inputs, so it is a
    # control: if it moved, the two runs differ for some reason other than the flag under test.
    _, l1_pcc = comp_pcc(
        _from_device(shared.out[0], mesh_device).float(), _from_device(unshared.out[0], mesh_device).float()
    )
    logger.info(f"[mtp index share] level 1 cross-run PCC (control, expect ~1.0): {l1_pcc:.6f}")
    assert l1_pcc > 0.999, f"level 1 differs across runs ({l1_pcc:.6f}); the two runs are not comparable"

    _, l2_pcc = comp_pcc(
        _from_device(shared.out[1], mesh_device).float(), _from_device(unshared.out[1], mesh_device).float()
    )
    logger.info(f"[mtp index share] level 2 cross-run PCC (observation only, NOT asserted): {l2_pcc:.6f}")

    for res in (shared, unshared):
        # Sharing makes entries 0 and 1 the same object; freeing it twice is a double free.
        for tensor in {id(t): t for t in res.indexer_indices}.values():
            ttnn.deallocate(tensor)
    ttnn.synchronize_device(mesh_device)
    logger.success("[mtp index share] index_share reaches the hardware in both settings")

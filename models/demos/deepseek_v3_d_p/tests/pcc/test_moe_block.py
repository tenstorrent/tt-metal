# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TtMoe's MoE blocks (tt/moe/moe_block.py) against the torch reference, per model and mesh, with seeded random weights.

Every case runs the same TtMoe (gate, shared expert, latent projections) with the block forced by ``moe_block``, so a
regression in one block shows up next to the other block's number for the same shape:

    dispatch_combine  routing setup -> dispatch -> routed expert -> combine -> post-combine reduce
    all_gather        tokens all-gathered over the mesh rows -> route plan -> flat expert (indexed) -> local reduce

Meshes: LoudBox 2 x 4 (8 chips, two mesh rows: the all-gather block's fused send-back) and Galaxy 8 x 4 (reduce-scatter
send-back). Kimi-K3 runs 512 routed experts on the LoudBox instead of 896: the flat expert holds at most 64 per chip
(896 / 8 = 112), and the all-gather block needs it. Everything else is the model's own shape at the 640-token-per-chip
chunk the production prefill feeds.
"""

import pytest

from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.tt_flat_routed_expert import resolve_routed_expert_impl
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import ROUTED_EXPERT_ACTIVATION_BY_NAME
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP

MOE_BLOCKS = ["dispatch_combine", "all_gather"]

# The flat expert's per-chip limit (tt_flat_routed_expert.FLAT_MAX_EXPERTS_PER_CHIP): K3 on 8 chips takes 512.
K3_LOUDBOX_EXPERTS = 512


def model_case(model, n_chips):
    """(variant, config, num_routed_experts, capacity factor, run_model kwargs) for a model on n_chips."""
    if model == "k2_7":
        return "kimi_k2_7", KimiK27Config, KimiK27Config.NUM_ROUTED_EXPERTS, 5, {}
    if model == "glm_5_3":
        # GLM's MoE test borrows the DeepSeek-V3 variant's gate grouping (test_ttnn_moe.test_glm_moe does the same)
        return "deepseek_v3_d_p", GLM53Config, GLM53Config.NUM_ROUTED_EXPERTS, 8, {}
    assert model == "k3", model
    experts = KimiK3Config.NUM_ROUTED_EXPERTS if n_chips >= 32 else K3_LOUDBOX_EXPERTS
    extra = dict(
        routed_emb_dim=KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        shared_hidden_dim=KimiK3Config.SHARED_EXPERT_INTERMEDIATE_SIZE,
        latent_use_norm=KimiK3Config.LATENT_MOE_USE_NORM,
        rms_norm_eps=KimiK3Config.RMS_NORM_EPS,
        final_output_pcc=0.965,
        routed_activation=ROUTED_EXPERT_ACTIVATION_BY_NAME[KimiK3Config.ROUTED_EXPERT_ACTIVATION],
        shared_activation=KimiK3Config.SHARED_EXPERT_ACTIVATION,
    )
    return "kimi_k3", KimiK3Config, experts, 5, extra


MESHES = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(fabric_payload_size=KimiK27Config.FABRIC_PAYLOAD_SIZE),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="loudbox-2x4",
    ),
    pytest.param(
        (8, 4),
        torus_xy_device_params(fabric_payload_size=KimiK27Config.FABRIC_PAYLOAD_SIZE),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="galaxy-8x4",
    ),
]

# "k2_7" / "glm_5_3" / "k3": disjoint under pytest -k (a bare "kimi" would match two models)
MODELS = ["k2_7", "glm_5_3", "k3"]


@pytest.mark.skipif(not is_blackhole(), reason="the all-gather MoE block needs the flat routed expert (Blackhole)")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params, num_links", MESHES, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("moe_block", MOE_BLOCKS)
@pytest.mark.parametrize("model", MODELS)
def test_moe_block_pcc(model, moe_block, mesh_device, device_params, num_links, request):
    """One MoE forward at the 5k-chunk shape graded against the torch reference (shared / routed / final PCC)."""
    variant_name, cfg, experts, capacity, extra = model_case(model, mesh_device.get_num_devices())
    _run(variant_name, cfg, experts, capacity, extra, moe_block, mesh_device, device_params, num_links, request)


def _run(variant_name, cfg, experts, capacity, extra, moe_block, mesh_device, device_params, num_links, request):
    from models.demos.deepseek_v3_d_p.tests.conftest import TEST_VARIANTS, _resolve_config_only

    variant = TEST_VARIANTS[variant_name]
    run_model(
        variant,
        _resolve_config_only(variant.name),
        mesh_device,
        device_params,
        PREFILL_CHUNK_TOKENS_PER_CHIP,
        cfg.EMB_SIZE,
        cfg.MOE_INTERMEDIATE_SIZE,
        experts,
        cfg.NUM_EXPERTS_PER_TOKEN,
        capacity,
        True,  # run_pcc_check
        num_links,
        per_axis_topology(device_params["fabric_config"]),
        GateComputeMode.DEVICE_FP32,
        request,
        routed_expert_impl=resolve_routed_expert_impl(cfg),
        moe_block=moe_block,
        **extra,
    )


def test_moe_block_selection(monkeypatch, expect_error):
    """Host only: the hot-swap plumbing. The env var wins over the model config, "auto" resolves by the mesh's row
    count, and an unknown name fails loudly."""
    from models.demos.deepseek_v3_d_p.tt.moe.moe_block import MOE_BLOCK_ENV, pick_moe_block, resolve_moe_block

    monkeypatch.delenv(MOE_BLOCK_ENV, raising=False)
    for cfg in (KimiK27Config, KimiK3Config, GLM53Config):
        assert resolve_moe_block(cfg) == "auto", cfg
    assert resolve_moe_block(None) == "dispatch_combine"
    monkeypatch.setenv(MOE_BLOCK_ENV, "dispatch_combine")
    assert resolve_moe_block(KimiK27Config) == "dispatch_combine"
    monkeypatch.setenv(MOE_BLOCK_ENV, "nope")
    with expect_error(ValueError, "unknown MoE block"):
        resolve_moe_block(KimiK27Config)
    assert pick_moe_block("auto", 2) == "all_gather"  # LoudBox / QuietBox
    assert pick_moe_block("auto", 8) == "dispatch_combine"  # Galaxy
    assert pick_moe_block("all_gather", 8) == "all_gather"

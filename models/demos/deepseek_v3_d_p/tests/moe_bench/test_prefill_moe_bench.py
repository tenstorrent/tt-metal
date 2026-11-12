# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill MoE layer benchmark on a Blackhole Galaxy (8x4 TorusXY), one row per model shape in moe_models.py.

Runs the production TtMoe (gate -> dispatch -> routed experts -> combine, plus the shared expert) at the 5k-token
prefill chunk (640 tokens/chip x SP 8) with placeholder weights and no PCC reference, so only device time is spent.
Each model is wired the way it runs: its own fabric payload, routed-expert / combine overlap and transport.

Run it under the device profiler (run_bench.sh does this) and post-process with roofline.py:

    python -m tracy -r -v -m pytest models/demos/deepseek_v3_d_p/tests/moe_bench/test_prefill_moe_bench.py
"""

import importlib

import pytest
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.moe_bench.moe_models import selected_models
from models.demos.deepseek_v3_d_p.tests.pcc import test_ttnn_moe
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import moe_transport_arg, routed_expert_combine_overlap_arg
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import ROUTED_EXPERT_ACTIVATION_BY_NAME
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP

_CFG_MODULES = {
    "DeepSeekV3Config": "deepseek_v3_config",
    "DeepSeekV4FlashConfig": "deepseek_v4_flash_config",
    "DeepSeekV4ProConfig": "deepseek_v4_pro_config",
    "KimiK27Config": "kimi_k2_7_config",
    "KimiK3Config": "kimi_k3_config",
    "MiniMaxM3Config": "minimax_m3_config",
    "GLM53Config": "glm_5_3_config",
}


def _model_cfg(m):
    mod = importlib.import_module(f"models.demos.deepseek_v3_d_p.reference.{_CFG_MODULES[m.prefill_cfg]}")
    cfg = getattr(mod, m.prefill_cfg)
    if m.note == "custom":
        # A custom shape rides on the DeepSeek-V3 wiring with its own hidden size.
        cfg = type(f"Custom_{m.name}", (cfg,), {"EMB_SIZE": m.hidden, "FABRIC_PAYLOAD_SIZE": m.hidden})
    return cfg


def _cases():
    # One case per model: its own variant (router config) and its own fabric payload, as production opens the mesh.
    return [
        pytest.param(
            m,
            m.prefill_variant,
            (8, 4),
            torus_xy_device_params(fabric_payload_size=moe_fabric_payload(_model_cfg(m))),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id=f"{m.name}-torus-xy-8x4",
        )
        for m in selected_models()
    ]


@pytest.mark.parametrize(
    "model, variant, mesh_device, device_params, num_links",
    _cases(),
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole Galaxy benchmark")
@pytest.mark.timeout(0)
def test_prefill_moe_bench(model, variant, config_only, mesh_device, device_params, num_links, request, monkeypatch):
    # Rows with a host gate keep DeepSeek-V3's router config: it decides which experts are picked, not how much
    # device work an expert does.
    cfg = _model_cfg(model)
    routed_activation = ROUTED_EXPERT_ACTIVATION_BY_NAME[model.routed_act]
    # run_model rejects device activations without a torch reference (swiglu_oai) even with PCC off. The
    # reference never runs here, so give it SiLU's entries; the device still runs the requested activation.
    for table in (test_ttnn_moe._TORCH_ROUTED_ACTIVATION, test_ttnn_moe._UPSTREAM_ACT):
        if routed_activation not in table:
            monkeypatch.setitem(table, routed_activation, table[ttnn.RoutedExpertActivation.Silu])
    kwargs = dict(
        routed_activation=routed_activation,
        shared_activation=getattr(cfg, "SHARED_EXPERT_ACTIVATION", "silu"),
        overlap_routed_expert_with_combine=routed_expert_combine_overlap_arg(cfg),
        moe_transport=moe_transport_arg(cfg),
        rms_norm_eps=getattr(cfg, "RMS_NORM_EPS", 1e-5),
        skip_upstream_reference=True,
    )
    if model.routed_hidden is not None:
        kwargs.update(
            routed_emb_dim=model.routed_hidden,
            latent_use_norm=getattr(cfg, "LATENT_MOE_USE_NORM", True),
        )
    if model.sh_inter != model.inter:
        kwargs["shared_hidden_dim"] = model.sh_inter
    kwargs.update(model.prefill_kwargs)

    grid = mesh_device.compute_with_storage_grid_size()
    # roofline.py splits the profiler CSV on these and reads the case parameters back from them.
    tag = (
        f"moe_bench|prefill|model={model.name}|H={model.hidden}|Hr={model.hr}|N={model.inter}|E={model.experts}"
        f"|k={model.k}|Nsh={model.sh_inter}|tokens_per_chip={PREFILL_CHUNK_TOKENS_PER_CHIP}"
        f"|mesh={mesh_device.shape[0]}x{mesh_device.shape[1]}|grid={grid.x}x{grid.y}|gate={model.prefill_gate}"
    )
    signpost(header=f"{tag}|START")
    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        PREFILL_CHUNK_TOKENS_PER_CHIP,
        model.hidden,
        model.inter,
        model.experts,
        model.k,
        model.capacity_factor,
        False,  # run_pcc_check: placeholder weights, device time only
        num_links,
        per_axis_topology(device_params["fabric_config"]),
        getattr(GateComputeMode, model.prefill_gate),
        request,
        **kwargs,
    )
    signpost(header=f"{tag}|END")

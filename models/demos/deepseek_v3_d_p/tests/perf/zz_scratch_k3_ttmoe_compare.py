# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the existing TtMoe (tt/moe/tt_moe.py, the Kimi K3 knobs of tests/pcc/test_ttnn_moe.py::test_kimi_k3_moe)
on exactly the weights and input of scratch_k3_moe_layer.py (random_k3_weights pool, the same x): dumps its output and
intermediates for the offline comparison (zz_k3_moe_accuracy.py --ttmoe). TtMoe's own layout: rows = sequence
(dispatch groups), columns = TP, so the 2560 tokens sit as [2, 1280] x 7168."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, moe_fabric_payload
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    compute_constants,
    extract_mesh_config,
    get_sp_mesh_composer,
    get_tp_mesh_composer,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_k3_token_split_moe import K3Dims, random_k3_weights
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import ROUTED_EXPERT_ACTIVATION_BY_NAME
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

N_TOK = 2560
POOL = int(os.environ.get("K3_POOL", "8"))
_PARAMS = [
    pytest.param((2, 4), fabric2d_device_params(fabric_payload_size=moe_fabric_payload(KimiK3Config)), id="2x4")
]


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(7200)
def test_ttmoe_k3_same_weights(mesh_device, device_params):
    d = K3Dims(E=int(os.environ.get("K3_E", "448")))
    w = random_k3_weights(d, pool_size=POOL)
    x = torch.randn(N_TOK, d.H, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    mc = extract_mesh_config(mesh_device)
    dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
    seq = N_TOK // dgs
    epc, meta, max_buf, max_per_e = compute_constants(seq, d.E, d.K, mesh_device.get_num_devices(), dgs, 5)
    logger.info(f"TtMoe K3: dgs {dgs} x seq {seq}, {ndg} dispatch groups, {epc} experts per chip")
    routed = [
        {"gate_proj": w.pool[g % POOL][0], "up_proj": w.pool[g % POOL][1], "down_proj": w.pool[g % POOL][2]}
        for g in range(d.E)
    ]
    moe = TtMoe(
        mesh_device=mesh_device,
        dispatch_group_size=dgs,
        num_dispatch_groups=ndg,
        experts_per_chip=epc,
        num_routed_experts=d.E,
        num_experts_per_tok=d.K,
        metadata_len=meta,
        max_dispatched_tokens_per_expert=max_per_e,
        max_dispatch_buffer_token_size=max_buf,
        seq_len_per_chip=seq,
        emb_dim=d.H,
        hidden_dim=d.I,
        num_links=2,
        topology=per_axis_topology(device_params["fabric_config"]),
        routed_expert_weights=routed,
        shared_expert_weights={"gate_proj": w.sh_gate, "up_proj": w.sh_up, "down_proj": w.sh_down},
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        routed_expert_activation=ROUTED_EXPERT_ACTIVATION_BY_NAME[KimiK3Config.ROUTED_EXPERT_ACTIVATION],
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        shared_expert_activation=KimiK3Config.SHARED_EXPERT_ACTIVATION,
        shared_expert_situ_beta=d.beta,
        shared_expert_situ_linear_beta=d.linear_beta,
        gate_weights={"weight": w.gate, "e_score_correction_bias": w.bias},
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        n_expert_groups=1,
        n_limited_groups=1,
        route_scale=d.route_scale,
        routed_emb_dim=d.latent,
        shared_hidden_dim=d.I_shared,
        latent_weights={"down_proj": w.lat_down, "up_proj": w.lat_up, "norm": w.lat_norm},
        latent_use_norm=True,
        rms_norm_eps=d.eps,
    )
    ttnn.synchronize_device(mesh_device)
    tt_x = ttnn.from_torch(
        x.reshape(dgs, seq, d.H),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    y, it = moe(tt_x, return_intermediates=True)
    ttnn.synchronize_device(mesh_device)
    tp, sp = get_tp_mesh_composer(mesh_device), get_sp_mesh_composer(mesh_device)
    host = lambda t, comp: ttnn.to_torch(t, mesh_composer=comp, dtype=torch.float32)  # noqa: E731
    dump = dict(
        x=x,
        y=host(y, tp).reshape(N_TOK, d.H),
        ys=host(it.shared_output, tp).reshape(N_TOK, d.H),
        yr=host(it.routed_output, tp).reshape(N_TOK, d.H),
        routed=host(it.latent_routed_output, tp).reshape(N_TOK, d.latent),
        lat=host(it.latent_input, sp).reshape(N_TOK, d.latent),
        gi=ttnn.to_torch(it.gate_indices, mesh_composer=sp, dtype=torch.int32).reshape(N_TOK, -1)[:, : d.K].long(),
    )
    dump["gw"] = ttnn.to_torch(it.gate_scores, mesh_composer=sp, dtype=torch.float32).reshape(N_TOK, -1)[:, : d.K]
    if it.gate_logits is not None:
        try:
            dump["logits"] = ttnn.to_torch(it.gate_logits, mesh_composer=sp, dtype=torch.float32).reshape(N_TOK, d.E)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"TTMOE logits not dumped: {e}")
    for k, v in dump.items():
        logger.info(f"TTMOE dump {k}: {tuple(v.shape)}")
    path = os.environ.get("K3_DUMP_TTMOE")
    if path:
        torch.save(dict(dump, E=d.E, pool=POOL), path)
        logger.info(f"TTMOE dump: {path}")

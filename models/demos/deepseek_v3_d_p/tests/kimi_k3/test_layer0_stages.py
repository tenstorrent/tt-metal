# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where does Kimi-K3's layer 0 diverge from torch? One PCC per stage.

`test_block_layer0.py` reports a single number for the whole layer, which is the right gate and the
wrong debugger: at 0.03 it says everything and nothing. This walks the same layer one stage at a
time against the same torch reference — round-trip, attention norm, KDA, FFN norm, dense FFN — so
the first stage that drops names the bug instead of the last one.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kda.layer import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext, build_attention
from models.demos.deepseek_v3_d_p.tt.kimi_k3.block import TtKimiK3Block
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_layer_state_dict

SP_AXIS, TP_AXIS = 0, 1
SEQ_LEN = 5120

PLACEMENTS = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE, l1_small_size=1152),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    )
]


def _rms_norm(hidden, weight):
    scale = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + KimiK3Config.RMS_NORM_EPS)
    return hidden * scale * weight


def _shard(mesh_device, hidden):
    dims = [None, None]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    return ttnn.from_torch(
        hidden.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    )


def _compose(mesh_device, tensor):
    dims = [0, 0]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    ).reshape(-1, KimiK3Config.EMB_SIZE)[:SEQ_LEN]


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_layer0_stage_by_stage(mesh_device, device_params):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    state_dict = load_layer_state_dict(Path(checkpoint), 0)
    hidden = trace.decoder_input(0, SEQ_LEN)
    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    schedule = KimiK3LayerSchedule.build(KimiK3Config, 0, 1)

    results = {}
    states = None
    try:
        # 0. Does the sharding round-trip at all? Everything downstream is meaningless if not.
        placed = _shard(mesh_device, hidden)
        results["round_trip"] = comp_pcc(hidden, _compose(mesh_device, placed), 0.99)[1]

        attention = build_attention(
            mesh_device,
            config,
            KimiK3Config,
            state_dict,
            layer_idx=0,
            schedule=schedule,
            seq_len=SEQ_LEN,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
        )
        states = KdaStateCache({0: attention.kda})
        attention.bind_state_cache(states)
        block = TtKimiK3Block(
            mesh_device,
            config,
            KimiK3Config,
            state_dict,
            layer_idx=0,
            local_idx=0,
            attention=attention,
            seq_len=SEQ_LEN,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
        )

        # 1. attn_norm
        normed = block.attn_norm(placed)
        want_normed = _rms_norm(hidden.float(), state_dict["attn_norm_weight"].float())
        results["attn_norm"] = comp_pcc(want_normed, _compose(mesh_device, normed), 0.99)[1]

        # The trace records this exact tensor, so it is also checkable against the model itself.
        results["attn_norm_vs_golden"] = comp_pcc(
            trace.rows("kda", "kda_input_layer_0", 0, SEQ_LEN), _compose(mesh_device, normed), 0.99
        )[1]

        # 2. KDA
        attn_out = attention.forward(normed, K3AttnContext())
        want_attn, _ = kda_forward_reference(want_normed.unsqueeze(0), state_dict["kda_weights"], kimi_k3_kda_config())
        results["kda"] = comp_pcc(want_attn.squeeze(0), _compose(mesh_device, attn_out), 0.99)[1]
        results["kda_vs_golden"] = comp_pcc(
            trace.rows("kda", "kda_output_layer_0", 0, SEQ_LEN), _compose(mesh_device, attn_out), 0.99
        )[1]

        # 3. residual + ffn_norm
        residual_t = hidden.float() + want_attn.squeeze(0).float()
        residual_tt = ttnn.add(placed, attn_out)
        results["residual"] = comp_pcc(residual_t, _compose(mesh_device, residual_tt), 0.99)[1]

        ffn_normed = block.ffn_norm(residual_tt)
        want_ffn_in = _rms_norm(residual_t, state_dict["ffn_norm_weight"].float())
        results["ffn_norm"] = comp_pcc(want_ffn_in, _compose(mesh_device, ffn_normed), 0.99)[1]

        # 4. dense FFN
        ffn_out = block._ffn_path(ffn_normed, actual_isl=None, padding_side="right", ctx=K3AttnContext())
        results["ffn_shape"] = str(tuple(ffn_out.shape))
    finally:
        if states is not None:
            states.deallocate()

    for stage, message in results.items():
        logger.info(f"STAGE {stage}: {message}")

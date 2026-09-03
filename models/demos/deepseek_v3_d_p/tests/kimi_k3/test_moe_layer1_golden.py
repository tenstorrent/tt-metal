# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3's LatentMoE at layer 1, fed the model's own MoE input.

The 2-layer ladder puts layer 0 at 0.9998514 and layer 1 at -0.0016, so the stack runs and layer 1's
output is wrong. Layer 1 adds exactly one thing over layer 0: LatentMoE in place of the dense FFN.
The 100k trace records `moe_io/moe_input_layer_1` and `moe_output_layer_1`, so the MoE can be scored
in complete isolation — real input, real weights, real output, nothing of ours upstream of it.

`test_ttnn_moe.py::test_kimi_k3_moe` already gates this module at 0.965, but on **seeded random**
weights against a torch reference built from the same seeds. That measures the device against
itself. This measures it against the checkpoint, which is where a weight-mapping mistake lives:
w1/w3/w2 -> gate/up/down, the latent pair's orientation, and the router bias dtype are all things
random weights cannot catch.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_layer_state_dict, load_routed_expert_weights
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock

SP_AXIS, TP_AXIS = 0, 1
SEQ_LEN = 5120
LAYER = 1

# The package's own K3 MoE bar is 0.965 at 8x4 (896 experts over 32 chips accumulate in bf8 latent
# space). Against the real model rather than a seeded reference, so the same bar applies.
MOE_PCC = 0.965

PLACEMENTS = [
    pytest.param(
        (8, 4),
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 1152},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    )
]


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
def test_moe_layer1_matches_golden(mesh_device, device_params):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    checkpoint = Path(checkpoint)
    state_dict = load_layer_state_dict(checkpoint, LAYER)
    logger.info(f"layer {LAYER}: reading {KimiK3Config.NUM_ROUTED_EXPERTS} routed experts (~59 GB)")
    state_dict["routed_expert_weights"] = load_routed_expert_weights(checkpoint, LAYER, KimiK3Config.NUM_ROUTED_EXPERTS)

    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    topology = per_axis_topology()

    moe = TtPrefillBlock._build_moe(
        mesh_device=mesh_device,
        model_cfg=KimiK3Config,
        config=config,
        state_dict=state_dict,
        seq_len=SEQ_LEN,
        sp_axis=SP_AXIS,
        emb_dim=KimiK3Config.EMB_SIZE,
        num_links=1,
        topology=topology,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        dispatch_buffer_capacity_factor=2,
        layer_idx=LAYER,
    )

    moe_in = trace.rows("moe_io", f"moe_input_layer_{LAYER}", 0, SEQ_LEN)
    placed = ttnn.squeeze(_shard(mesh_device, moe_in), dim=0)
    got, _ = moe(placed, return_intermediates=False, actual_isl=None, padding_side="right")
    got = _compose(mesh_device, ttnn.unsqueeze(got, dim=0))

    want = trace.rows("moe_io", f"moe_output_layer_{LAYER}", 0, SEQ_LEN)
    passed, message = comp_pcc(want, got, MOE_PCC)
    logger.info(f"K3 LatentMoE layer {LAYER} vs golden moe_output_layer_{LAYER}, T={SEQ_LEN}: {message}")
    assert passed, f"K3 LatentMoE != the model's own MoE: {message}"

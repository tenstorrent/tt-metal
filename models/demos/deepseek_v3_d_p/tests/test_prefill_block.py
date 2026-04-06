# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TtPrefillBlock — verifies composition of norm → MLA → residual → norm → FFN/MoE → residual.

Validates output shapes and PCC against torch reference.

Uses HF DeepseekV3Model layer as the reference: creates a model with random weights,
extracts those weights into our TT state_dict format, and compares forward passes.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import create_hf_model, extract_layer_state_dict
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD_DENSE = 0.999
PCC_THRESHOLD_MOE_GATE_HOST = 0.993
PCC_THRESHOLD_MOE_GATE_DEVICE = 0.980


@pytest.mark.parametrize("pcc_validation", [True, False], ids=["pcc", "smoke"])
@pytest.mark.parametrize("isl_total", [1024])
@pytest.mark.parametrize(
    "layer_type, gate_fallback_mode",
    [
        ("dense", None),
        ("moe", GateComputeMode.HOST_ALL),
        ("moe", GateComputeMode.DEVICE),
    ],
    ids=["dense", "moe-gate_host", "moe-gate_device"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_prefill_block(
    config_only,
    mesh_device,
    device_params,
    isl_total,
    layer_type,
    gate_fallback_mode,
    num_links,
    topology,
    pcc_validation,
):
    profiler.clear()
    profiler.start("total_test_time")
    config = config_only
    config.max_seq_len = isl_total

    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    sp_factor = mesh_shape[sp_axis]
    tp_factor = mesh_shape[tp_axis]
    emb_dim = config.hidden_size
    isl_per_chip = isl_total // sp_factor

    # layer_idx=0 for dense (< NUM_DENSE_LAYERS=3), layer_idx=3 for MoE (>= 3)
    layer_idx = 0 if layer_type == "dense" else DeepSeekV3Config.NUM_DENSE_LAYERS

    logger.info(f"mesh_shape={mesh_shape}, sp_factor={sp_factor}, tp_factor={tp_factor}")
    logger.info(
        f"isl_total={isl_total}, isl_per_chip={isl_per_chip}, "
        f"layer_type={layer_type}, layer_idx={layer_idx}, gate_fallback_mode={gate_fallback_mode}"
    )

    # --- Build HF reference model and extract weights ---
    profiler.start("weights_creation")
    torch.manual_seed(42)
    num_layers = layer_idx + 1
    hf_model = create_hf_model(config, num_layers)
    hf_sd = hf_model.state_dict()
    state_dict = extract_layer_state_dict(hf_sd, layer_idx, hf_model.layers[layer_idx])
    profiler.end("weights_creation")

    # --- Create input ---
    torch.manual_seed(123)
    torch_input = torch.randn(1, isl_total, emb_dim, dtype=torch.bfloat16)

    # --- Torch reference (only when pcc_validation is enabled) ---
    torch_output = None
    if pcc_validation:
        profiler.start("torch_reference")
        logger.info("Running torch reference forward...")
        position_ids = torch.arange(isl_total, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.zeros(1, 1, isl_total, isl_total, dtype=torch.bfloat16)
        with torch.no_grad():
            layer_out = hf_model.layers[layer_idx](
                torch_input, attention_mask=attention_mask, position_ids=position_ids
            )
            torch_output = layer_out[0]
        logger.info(f"Torch reference output shape: {torch_output.shape}")
        profiler.end("torch_reference")

    # --- TT block ---
    profiler.start("tt_block_creation")
    block_kwargs = dict(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        layer_idx=layer_idx,
        seq_len=isl_total,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
    )
    if gate_fallback_mode is not None:
        block_kwargs["gate_fallback_mode"] = gate_fallback_mode

    block = TtPrefillBlock(**block_kwargs)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_block_creation")

    # Shard input to device: [1, 1, isl_total, emb_dim] → [1, 1, isl_per_chip, emb_dim/tp]
    tt_input_4d = torch_input.unsqueeze(0)  # [1, 1, isl_total, emb_dim]
    tt_input = ttnn.from_torch(
        tt_input_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(-2, -1)),
    )

    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope_setup.get_rope_tensors(isl_total)

    profiler.start("tt_forward")
    logger.info("Running TtPrefillBlock forward...")
    tt_output = block(tt_input, rope_tensors)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")
    logger.info("Forward pass completed successfully")

    # --- Validate output shape ---
    expected_per_device_shape = [1, 1, isl_per_chip, emb_dim // tp_factor]
    output_shape = list(tt_output.shape)
    assert (
        output_shape == expected_per_device_shape
    ), f"Output shape mismatch: got {output_shape}, expected {expected_per_device_shape}"
    logger.info(f"Output shape: {output_shape} (matches expected)")

    # --- PCC check ---
    if torch_output is not None:
        profiler.start("pcc_validation")
        tt_output_host = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)
        # Remove leading batch dim: [1, 1, isl_total, emb_dim] → [1, isl_total, emb_dim]
        tt_output_host = tt_output_host.squeeze(0)

        if layer_type == "dense":
            pcc_threshold = PCC_THRESHOLD_DENSE
        else:
            if gate_fallback_mode == GateComputeMode.DEVICE:
                pcc_threshold = PCC_THRESHOLD_MOE_GATE_DEVICE
            else:
                pcc_threshold = PCC_THRESHOLD_MOE_GATE_HOST

        _, pcc = comp_pcc(torch_output.float(), tt_output_host.float())
        profiler.end("pcc_validation")
        logger.info(f"PCC: {pcc:.6f} (threshold: {pcc_threshold})")
        assert pcc > pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"
        logger.success(
            f"TtPrefillBlock test passed "
            f"(layer_type={layer_type}, gate_fallback_mode={gate_fallback_mode}, PCC={pcc:.4f})"
        )
    else:
        logger.success(
            f"TtPrefillBlock smoke test passed (layer_type={layer_type}, gate_fallback_mode={gate_fallback_mode})"
        )
    profiler.end("total_test_time")

    # --- Timing report ---
    logger.info(f"\n{'='*60}")
    logger.info("Timing Report")
    logger.info(f"{'='*60}")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Hang/smoke test: 6 prefill layers on a single Galaxy host (8x4, 32 chips) with
num_dispatch_subgroups=4 partitioning the SP axis into four 2x4 subgroups.

Goal: exercise the full prefill block (attn norm + MLA + ffn norm + dense FFN
or MoE) for a stack of layers in subgroup mode and confirm the pipeline runs to
completion without hangs. Weights are random; correctness is not validated —
only output shape and finiteness are checked.

DeepSeek V3 has first_k_dense_replace=3, so 6 layers exercises 3 dense + 3 MoE.

Run command (set TT_MESH_GRAPH_DESC_PATH before invoking):
    METAL_HOME=/data/ianastasijevic/workspaces/main/tt-metal
    TT_MESH_GRAPH_DESC_PATH=$METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto \\
      source $METAL_HOME/python_env/bin/activate && cd $METAL_HOME && \\
      pytest models/demos/deepseek_v3_d_p/tests/test_8x4_subgroups_prefill.py -v
"""

import gc

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import create_hf_model, extract_tt_state_dict


@pytest.mark.skipif(not is_blackhole(), reason="Requires Blackhole.")
@pytest.mark.parametrize(
    "num_layers, isl_total, n_routed_experts, capacity_factor, gate_fallback_mode",
    [
        pytest.param(6, 1024, 64, 4, GateComputeMode.HOST_ALL, id="6L-isl1024-e64-cf4-host"),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology, num_dispatch_subgroups, dispatch_group_size",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            1,
            ttnn.Topology.Linear,
            4,
            2,
            id="subgroups-4x2-mesh-8x4-linear-1link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_subgroups_prefill_8x4(
    config_only,
    mesh_device,
    device_params,
    num_layers,
    isl_total,
    n_routed_experts,
    capacity_factor,
    gate_fallback_mode,
    num_links,
    topology,
    num_dispatch_subgroups,
    dispatch_group_size,
):
    """Stack of TtPrefillBlock layers on 8x4 mesh with 2x4 dispatch subgroups (random weights)."""
    torch.manual_seed(42)

    profiler.clear()
    profiler.start("test_subgroups_prefill_8x4")

    config = config_only
    config.max_seq_len = isl_total

    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    sp_factor = mesh_shape[sp_axis]
    tp_factor = mesh_shape[tp_axis]
    emb_dim = config.hidden_size
    isl_per_chip = isl_total // sp_factor

    assert tuple(mesh_device.shape) == (8, 4), f"Expected (8, 4) mesh, got {mesh_device.shape}"
    assert sp_factor == dispatch_group_size * num_dispatch_subgroups, (
        f"mesh row axis ({sp_factor}) must equal "
        f"dispatch_group_size ({dispatch_group_size}) * num_dispatch_subgroups ({num_dispatch_subgroups})"
    )

    logger.info(
        f"Running {num_layers}-layer prefill on mesh={mesh_shape}, isl_total={isl_total}, "
        f"isl_per_chip={isl_per_chip}, n_routed_experts={n_routed_experts}, "
        f"num_dispatch_subgroups={num_dispatch_subgroups}, dispatch_group_size={dispatch_group_size}"
    )
    signpost(
        f"Subgroups prefill 8x4 — {num_layers} layers, mesh={mesh_shape}, "
        f"isl_total={isl_total}, experts={n_routed_experts}"
    )

    # --- Monkeypatch n_routed_experts for the duration of the test ---
    orig_num_routed_experts = DeepSeekV3Config.NUM_ROUTED_EXPERTS
    DeepSeekV3Config.NUM_ROUTED_EXPERTS = n_routed_experts

    try:
        # --- Build HF model with random weights, extract per-layer state dicts ---
        logger.info(f"Creating HF model with {num_layers} layers (random weights)...")
        profiler.start("hf_model_creation")
        hf_model = create_hf_model(config, num_layers, n_routed_experts=n_routed_experts)
        tt_state = extract_tt_state_dict(hf_model)

        # Embed initial random tokens with the HF embedding (cheap and avoids env-var-gated TtPrefillTransformer).
        token_ids = torch.randint(0, config.vocab_size, (1, isl_total), dtype=torch.int64)
        with torch.no_grad():
            h0 = hf_model.embed_tokens(token_ids).to(torch.bfloat16)  # [1, isl_total, emb_dim]
        del hf_model
        gc.collect()
        profiler.end("hf_model_creation")
        logger.info(f"Initial embedding shape: {h0.shape}")

        # --- Build TT blocks ---
        profiler.start("tt_blocks_construction")
        blocks = []
        for layer_idx in range(num_layers):
            logger.info(f"Building TtPrefillBlock layer {layer_idx}/{num_layers - 1}...")
            block_kwargs = dict(
                mesh_device=mesh_device,
                config=config,
                state_dict=tt_state["layers"][layer_idx],
                layer_idx=layer_idx,
                seq_len=isl_total,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                num_dispatch_subgroups=num_dispatch_subgroups,
            )
            is_dense = layer_idx < config.first_k_dense_replace
            if not is_dense:
                block_kwargs["gate_fallback_mode"] = gate_fallback_mode
                block_kwargs["capacity_factor"] = capacity_factor
                block_kwargs["routed_expert_activations_dtype"] = ttnn.bfloat8_b
                block_kwargs["routed_expert_weights_dtype"] = ttnn.bfloat8_b
                block_kwargs["shared_expert_activations_dtype"] = ttnn.bfloat16
                block_kwargs["shared_expert_weights_dtype"] = ttnn.bfloat8_b
            blocks.append(TtPrefillBlock(**block_kwargs))
        # Free torch state dicts now that TT weights are constructed.
        del tt_state
        gc.collect()
        ttnn.synchronize_device(mesh_device)
        profiler.end("tt_blocks_construction")
        logger.info("All TtPrefillBlocks constructed")

        # --- KVPE cache (one slot per layer) and RoPE tensors ---
        kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank
        tt_kvpe_cache = init_kvpe_cache(
            kvpe_cache_head_dim=kvpe_cache_head_dim,
            mesh_device=mesh_device,
            seq_len=isl_total,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=num_layers,
        )
        rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
        rope_tensors = rope_setup.get_rope_tensors(isl_total)

        # --- Shard initial hidden state to mesh: [1, 1, isl_total, emb_dim] -> per chip [1, 1, isl_per_chip, emb/tp] ---
        h_tt = ttnn.from_torch(
            h0.unsqueeze(0),  # [1, 1, isl_total, emb_dim]
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(-2, -1)),
        )
        del h0

        # --- Forward through layers ---
        profiler.start("tt_forward")
        for layer_idx, block in enumerate(blocks):
            signpost(f"forward_layer_{layer_idx}_start")
            logger.info(f"Layer {layer_idx}/{num_layers - 1} forward...")
            h_tt, _ = block(h_tt, rope_tensors, tt_kvpe_cache, cache_layer_idx=layer_idx)
            ttnn.synchronize_device(mesh_device)
            signpost(f"forward_layer_{layer_idx}_end")
        profiler.end("tt_forward")
        logger.info("All layers completed without hangs")

        # --- Validate output shape and finiteness ---
        expected_per_device_shape = [1, 1, isl_per_chip, emb_dim // tp_factor]
        output_shape = list(h_tt.shape)
        assert (
            output_shape == expected_per_device_shape
        ), f"Output shape mismatch: got {output_shape}, expected {expected_per_device_shape}"
        logger.info(f"Final output per-device shape: {output_shape} (matches expected)")

        h_host = ttnn.to_torch(
            h_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        ).to(torch.float32)
        finite_ratio = torch.isfinite(h_host).float().mean().item()
        logger.info(f"finite_ratio={finite_ratio} (over {h_host.numel()} elements)")
        assert finite_ratio == 1.0, f"Output contains non-finite values (finite_ratio={finite_ratio})"

        logger.success(
            f"Smoke test passed: {num_layers} layers, mesh={mesh_shape}, "
            f"num_dispatch_subgroups={num_dispatch_subgroups}, dispatch_group_size={dispatch_group_size}"
        )
    finally:
        DeepSeekV3Config.NUM_ROUTED_EXPERTS = orig_num_routed_experts

    profiler.end("test_subgroups_prefill_8x4")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")

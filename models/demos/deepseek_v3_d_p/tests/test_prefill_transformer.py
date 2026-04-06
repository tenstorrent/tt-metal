# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TtPrefillTransformer — verifies composition of embed -> [block x N] -> norm.

Validates output shapes and PCC against torch reference.

Uses HF DeepseekV3Model as the reference: creates a model with random or pretrained
weights, extracts those weights into our TT state_dict format, and compares forward passes.

Parametrized over:
- use_pretrained: real pretrained weights from DeepSeek-R1-0528 vs random weights
- input_source: "random", "json_prompts", or InfiniteBench subset (passkey, kv_retrieval, etc.)
- pcc_validation: per-stage PCC check (via return_intermediates) vs shape-only smoke test
- n_routed_experts / capacity_factor / gate_fallback_mode: MoE configurations
"""

import json

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    PROMPTS_PATH,
    create_hf_model,
    create_hf_model_with_weights,
    download_infinitebench_subset,
    extract_tt_state_dict,
    get_or_compute_host_reference,
    tokenize_infinitebench_to_isl,
    tokenize_prompts_to_isl,
    tt_state_dict_to_hf_state_dict,
)
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD = 0.99

# Input sources: "random" = random token IDs, "json_prompts" = test_prompts_1024.json,
# or any InfiniteBench subset name (downloaded on first use via infinitebench_prompt fixture).
INFINITEBENCH_SUBSET_NAMES = {"passkey", "kv_retrieval", "longdialogue_qa_eng", "longbook_qa_eng"}


@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize(
    "input_source",
    ["json_prompts", "random", "passkey", "kv_retrieval", "longdialogue_qa_eng", "longbook_qa_eng"],
)
@pytest.mark.parametrize("pcc_validation", [True, False], ids=["pcc", "smoke"])
@pytest.mark.parametrize("isl_total", [1024, 6400])
@pytest.mark.parametrize("num_layers", [6])
@pytest.mark.parametrize(
    "n_routed_experts, capacity_factor, gate_fallback_mode",
    [
        (64, 4, GateComputeMode.HOST_ALL),
        (256, 32, GateComputeMode.HOST_ALL),
        (256, 32, GateComputeMode.DEVICE),
    ],
    ids=["e64_cf4_host", "e256_cf32_host", "e256_cf32_device"],
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
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_prefill_transformer(
    config_only,
    mesh_device,
    device_params,
    isl_total,
    num_layers,
    n_routed_experts,
    capacity_factor,
    gate_fallback_mode,
    num_links,
    topology,
    pcc_validation,
    input_source,
    use_pretrained,
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    request,
):
    torch.manual_seed(42)

    # Skip invalid pretrained combinations
    if use_pretrained and n_routed_experts != 256:
        pytest.skip("Pretrained weights only available for 256 experts")

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

    weight_type = "pretrained" if use_pretrained else "random"

    # Only enable weight caching for pretrained runs
    if use_pretrained and weight_cache_path is not None:
        rows, cols = mesh_shape
        effective_cache_path = weight_cache_path / f"{rows}x{cols}"
        effective_cache_path.mkdir(parents=True, exist_ok=True)
    else:
        effective_cache_path = None

    logger.info(f"mesh_shape={mesh_shape}, sp_factor={sp_factor}, tp_factor={tp_factor}")
    logger.info(
        f"isl_total={isl_total}, isl_per_chip={isl_per_chip}, "
        f"num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
        f"capacity_factor={capacity_factor}, gate_fallback_mode={gate_fallback_mode}, "
        f"input_source={input_source}, pcc_validation={pcc_validation}, "
        f"weights={weight_type}"
    )

    # --- Monkeypatch n_routed_experts ---
    orig_num_routed_experts = DeepSeekV3Config.NUM_ROUTED_EXPERTS
    DeepSeekV3Config.NUM_ROUTED_EXPERTS = n_routed_experts

    # --- Build HF reference model and extract weights ---
    profiler.start("weights_creation")
    if use_pretrained:
        _, state_dict = request.getfixturevalue("pretrained_transformer_weights")
        hf_sd = tt_state_dict_to_hf_state_dict(state_dict)
        hf_model = create_hf_model_with_weights(config, num_layers, hf_sd)
    else:
        hf_model = create_hf_model(config, num_layers, n_routed_experts=n_routed_experts)
        state_dict = extract_tt_state_dict(hf_model)
    profiler.end("weights_creation")

    # --- Create input ---
    if input_source == "random":
        token_ids = torch.randint(0, config.vocab_size, (1, isl_total), dtype=torch.int64)
    elif input_source == "json_prompts":
        profiler.start("tokenization")
        tok = request.getfixturevalue("tokenizer")
        token_ids = tokenize_prompts_to_isl(tok, PROMPTS_PATH, isl_total, sp_factor)
        profiler.end("tokenization")
        logger.info(f"Tokenized input shape: {token_ids.shape}, first 10 tokens: {token_ids[0, :10].tolist()}")
    elif input_source in INFINITEBENCH_SUBSET_NAMES:
        profiler.start("tokenization")
        tok = request.getfixturevalue("tokenizer")
        cached_path = download_infinitebench_subset(input_source)
        with open(cached_path) as f:
            prompt_text = json.load(f)["prompt"]
        token_ids = tokenize_infinitebench_to_isl(tok, prompt_text, isl_total, sp_factor)
        profiler.end("tokenization")
        logger.info(
            f"Tokenized InfiniteBench [{input_source}] shape: {token_ids.shape}, first 10 tokens: {token_ids[0, :10].tolist()}"
        )
    else:
        raise ValueError(f"Unknown input_source: {input_source}")

    # --- TT transformer ---
    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        num_layers=num_layers,
        seq_len=isl_total,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=gate_fallback_mode,
        capacity_factor=capacity_factor,
        weight_cache_path=effective_cache_path,
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_transformer_creation")

    # --- Shard token_ids to device ---
    # Reshape [1, isl_total] -> [sp_factor, 1, isl_per_chip] for SP sharding
    token_ids_reshaped = token_ids.reshape(sp_factor, 1, isl_per_chip)

    tt_tokens = ttnn.from_torch(
        token_ids_reshaped,
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
    )

    # --- Forward ---
    profiler.start("tt_forward")
    logger.info("Running TtPrefillTransformer forward...")
    result = transformer(tt_tokens, return_intermediates=pcc_validation)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")
    logger.info("Forward pass completed successfully")

    if pcc_validation:
        tt_output, tt_snapshots = result
    else:
        tt_output = result

    # --- Validate output shape ---
    expected_per_device_shape = [1, 1, isl_per_chip, emb_dim // tp_factor]
    output_shape = list(tt_output.shape)
    assert (
        output_shape == expected_per_device_shape
    ), f"Output shape mismatch: got {output_shape}, expected {expected_per_device_shape}"
    logger.info(f"Output shape: {output_shape} (matches expected)")

    # --- PCC check ---
    if pcc_validation:
        profiler.start("pcc_validation")
        if use_pretrained and input_source != "random":
            threshold = 0.97
        elif use_pretrained:
            threshold = 0.95
        elif n_routed_experts < 256:
            threshold = 0.985
        else:
            threshold = PCC_THRESHOLD  # 0.99

        cache_key = f"{weight_type}_{input_source}_isl{isl_total}" f"_layers{num_layers}_experts{n_routed_experts}"
        is_ci = is_ci_env or is_ci_v2_env
        ref_snapshots = get_or_compute_host_reference(hf_model, token_ids, num_layers, cache_key, is_ci)

        # Per-stage PCC comparison
        pcc_results = []
        for (label, tt_host), ref_host in zip(tt_snapshots, ref_snapshots):
            try:
                _, pcc = comp_pcc(ref_host.float(), tt_host.float())
                logger.info(f"{label:<20s}  PCC = {pcc:.6f}")
                pcc_results.append((label, pcc))
            except Exception as e:
                logger.error(f"{label:<20s}  PCC comparison failed: {e}")
                pcc_results.append((label, -1.0))

        profiler.end("pcc_validation")

        # Summary table
        logger.info(f"\n{'='*50}")
        logger.info(f"{'Stage':<20s}  {'PCC':>10s}  {'Status':>8s}")
        logger.info(f"{'-'*50}")
        failures = []
        for label, pcc in pcc_results:
            status = "PASS" if pcc > threshold else ("FAIL" if pcc >= 0 else "ERROR")
            logger.info(f"{label:<20s}  {pcc:>10.6f}  {status:>8s}")
            if pcc <= threshold:
                failures.append((label, pcc))
        logger.info(f"{'='*50}")

        if failures:
            msg = "; ".join(f"{label}: {pcc:.6f}" for label, pcc in failures)
            pytest.fail(f"PCC below {threshold} at: {msg}")

        logger.success(
            f"TtPrefillTransformer PCC test passed "
            f"(num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
            f"capacity_factor={capacity_factor}, gate_fallback_mode={gate_fallback_mode}, "
            f"weights={weight_type})"
        )
    else:
        logger.success(
            f"TtPrefillTransformer smoke test passed "
            f"(num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
            f"capacity_factor={capacity_factor}, gate_fallback_mode={gate_fallback_mode}, "
            f"weights={weight_type})"
        )

    profiler.end("total_test_time")

    # --- Timing report ---
    logger.info(f"\n{'='*60}")
    logger.info("Timing Report")
    logger.info(f"{'='*60}")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")

    # Restore original config
    DeepSeekV3Config.NUM_ROUTED_EXPERTS = orig_num_routed_experts

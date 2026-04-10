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

import gc
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
    download_infinitebench_subset,
    extract_tt_state_dict,
    load_and_compute_layer_by_layer,
    load_reference_cache,
    save_reference_cache,
    tokenize_infinitebench_to_isl,
    tokenize_prompts_to_isl,
)
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD = 0.99

# Input sources: "random" = random token IDs, "json_prompts" = test_prompts_1024.json,
# or any InfiniteBench subset name (downloaded on first use via infinitebench_prompt fixture).
INFINITEBENCH_SUBSET_NAMES = {"passkey", "kv_retrieval", "longdialogue_qa_eng", "longbook_qa_eng"}


@pytest.mark.parametrize("return_kv_cache", [True], ids=["kv_cache"])
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize(
    "input_source",
    ["json_prompts", "random", "passkey", "kv_retrieval", "longdialogue_qa_eng", "longbook_qa_eng"],
)
@pytest.mark.parametrize("pcc_validation", [True, False], ids=["pcc", "smoke"])
@pytest.mark.parametrize("isl_total", [1024, 2048, 6400])
@pytest.mark.parametrize("num_layers", [12])
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
    return_kv_cache,
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

    # --- Cache-aware loading strategy ---
    profiler.start("cache_check")

    from models.demos.deepseek_v3_d_p.utils.cache_utils import check_reference_cache_exists, check_ttnn_cache_complete

    # Check cache states
    experts_per_chip = 256 // (mesh_shape[0] * mesh_shape[1]) if use_pretrained else 8
    ttnn_cache_complete = (
        check_ttnn_cache_complete(effective_cache_path, num_layers, experts_per_chip, tuple(mesh_shape))
        if effective_cache_path
        else False
    )

    cache_key = f"{weight_type}_{input_source}_isl{isl_total}_layers{num_layers}_experts{n_routed_experts}"
    ref_cache_exists = check_reference_cache_exists(cache_key) if pcc_validation else False

    logger.info(f"Cache status: TTNN={ttnn_cache_complete}, Reference={ref_cache_exists}")

    # Determine what we need to load
    need_to_load_weights = not ttnn_cache_complete
    need_to_compute_reference = pcc_validation and not ref_cache_exists
    need_hf_model = need_to_load_weights or need_to_compute_reference

    logger.info(
        f"Loading strategy: need_weights={need_to_load_weights}, "
        f"need_reference={need_to_compute_reference}, "
        f"need_hf_model={need_hf_model}"
    )

    profiler.end("cache_check")

    # --- Create input (needed early for reference computation) ---
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

    # --- Build HF model and/or extract weights based on cache state ---
    profiler.start("weights_creation")

    state_dict = None
    ref_snapshots = None
    ref_kvpe_list = None

    if use_pretrained:
        model_path = request.getfixturevalue("model_path")
        logger.debug(f"{model_path=}")
        if need_hf_model:
            # Use unified loader with flags
            logger.info("Processing layers with unified loader...")
            result = load_and_compute_layer_by_layer(
                model_path=model_path,
                config=config,
                num_layers=num_layers,
                token_ids=token_ids,
                compute_reference=need_to_compute_reference,
                build_ttnn_cache=need_to_load_weights,
                weight_cache_path=effective_cache_path,
                mesh_device=mesh_device,
                seq_len=isl_total,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                capacity_factor=capacity_factor,
                gate_fallback_mode=gate_fallback_mode,
            )

            # state_dict is always None (cache built to disk)
            state_dict = {}
            ref_snapshots = result.ref_snapshots
            ref_kvpe_list = result.ref_kvpe_list

            # Save reference cache if computed
            if need_to_compute_reference and ref_snapshots is not None:
                save_reference_cache(cache_key, ref_snapshots, ref_kvpe_list)
                logger.info("Reference cached")
        else:
            # Both caches exist - skip loading entirely
            logger.info("Both caches exist, skipping weight loading")
            state_dict = {}
    else:
        # Random weights - always create HF model
        logger.info("Creating HF model with random weights...")
        hf_model = create_hf_model(config, num_layers, n_routed_experts=n_routed_experts)
        state_dict = extract_tt_state_dict(hf_model)
        del hf_model
        gc.collect()

    profiler.end("weights_creation")

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

    # --- Free memory immediately after transformer creation ---
    del state_dict
    gc.collect()
    logger.info("State dict freed after transformer creation")

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
    do_return_kv = pcc_validation and return_kv_cache
    result = transformer(tt_tokens, return_intermediates=pcc_validation, return_kv_cache=do_return_kv)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")
    logger.info("Forward pass completed successfully")

    if pcc_validation:
        if do_return_kv:
            tt_output, tt_snapshots, tt_kv_caches = result
        else:
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
        logger.info(f"PCC threshold: {threshold}")

        # Load reference from cache if not already computed
        if ref_snapshots is None:
            logger.info("Loading reference from cache...")
            ref_snapshots, ref_kvpe_list = load_reference_cache(cache_key)
        # else: already computed by load_and_compute_layer_by_layer()

        # Per-stage PCC comparison
        pcc_results = []
        for (label, tt_host), ref_host in zip(tt_snapshots, ref_snapshots):
            try:
                _, pcc = comp_pcc(ref_host.float(), tt_host.float())
                logger.debug(f"{label:<20s}  PCC = {pcc:.6f}")
                pcc_results.append((label, pcc))
            except Exception as e:
                logger.error(f"{label:<20s}  PCC comparison failed: {e}")
                pcc_results.append((label, -1.0))

        # Per-layer KVPE PCC comparison
        if do_return_kv:
            kv_lora_rank = config.kv_lora_rank
            for (label, tt_kvpe), ref_kvpe in zip(tt_kv_caches, ref_kvpe_list):
                try:
                    _, kv_pcc = comp_pcc(
                        ref_kvpe[:, :, :, :kv_lora_rank].float(), tt_kvpe[:, :, :, :kv_lora_rank].float()
                    )
                    _, pe_pcc = comp_pcc(
                        ref_kvpe[:, :, :, kv_lora_rank:].float(), tt_kvpe[:, :, :, kv_lora_rank:].float()
                    )
                    logger.info(f"{label:<20s}  KV PCC = {kv_pcc:.6f}, PE PCC = {pe_pcc:.6f}")
                    pcc_results.append((f"{label}_kv", kv_pcc))
                    pcc_results.append((f"{label}_pe", pe_pcc))
                except Exception as e:
                    logger.error(f"{label:<20s}  KVPE PCC comparison failed: {e}")
                    pcc_results.append((f"{label}_kv", -1.0))
                    pcc_results.append((f"{label}_pe", -1.0))

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

        # Store failures for deferred check (after timing report)
        has_pcc_failures = len(failures) > 0

        if not has_pcc_failures:
            logger.success(
                f"TtPrefillTransformer PCC test passed "
                f"(num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
                f"capacity_factor={capacity_factor}, gate_fallback_mode={gate_fallback_mode}, "
                f"weights={weight_type})"
            )
        else:
            pcc_failure_msg = "; ".join(f"{label}: {pcc:.6f}" for label, pcc in failures)
            logger.error(
                f"TtPrefillTransformer PCC test has failures " f"(num_layers={num_layers}, failures={len(failures)})"
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

    # Deferred PCC failure check (after timing report)
    if pcc_validation and has_pcc_failures:
        pytest.fail(f"PCC below {threshold} at: {pcc_failure_msg}")

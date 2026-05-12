# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
- n_routed_experts / gate_fallback_mode: MoE configurations
"""

import gc
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from conftest import is_galaxy
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    create_balanced_chunk_order,
    reorder_tensor_chunks,
    reverse_reorder_tensor_chunks,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    create_kv_chunk_address_table,
    init_kvpe_cache,
)
from models.demos.deepseek_v3_d_p.utils.test_utils import save_intermediate_output
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ABC_1K_PATH,
    ABC_SHORT_PATH,
    P64TOK_PATH,
    P960TOK_PATH,
    PIE960_PATH,
    PROMPTS_PATH,
    ReferenceCacheKey,
    check_reference_cache_exists,
    create_hf_model,
    download_infinitebench_subset,
    extract_tt_state_dict,
    load_and_compute_layer_by_layer,
    load_debug_trace,
    load_reference_cache,
    save_reference_cache,
    slice_non_padded,
    tokenize_prompt_to_isl,
)
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD = 0.99
TRACE_PCC_THRESHOLD = 0.97

TRACE_DIR_BASE = Path(os.getenv("DEEPSEEK_V3_TRACE_DIR", "/data/ddjekic/bit_sculpt/results/deepseek-r1-0528"))
ILLIAD_1024_TRACE = TRACE_DIR_BASE / "illiad_prefill_fa2"
ILLIAD_25024_TRACE = TRACE_DIR_BASE / "illiad_prefill_fa2_25024"
ABC_1k_PADD_RIGHT_1024 = TRACE_DIR_BASE / "ABC_1k_prefill_padd_right_1024"
ABC_1k_PADD_LEFT_1024 = TRACE_DIR_BASE / "ABC_1k_prefill_padd_left_1024"
LONGBOOK_QA_ENG_25024 = TRACE_DIR_BASE / "longbook_qa_eng_25088"

# Input sources: "random" = random token IDs, "json_prompts" = test_prompts_1024.json,
# or any InfiniteBench subset name (downloaded on first use via infinitebench_prompt fixture).
INFINITEBENCH_SUBSET_NAMES = {"passkey", "kv_retrieval", "longdialogue_qa_eng", "longbook_qa_eng"}
SEQ_LEN_1K = 1024
SEQ_LEN_25K = 25 * 1024


@pytest.mark.skipif(not is_blackhole(), reason="Requires Blackhole.")
@pytest.mark.parametrize("tokenizer", ["right", "left"], indirect=True, ids=["right_pad", "left_pad"])
@pytest.mark.parametrize("temperature", [[0.5]], ids=["temp_sweep"])
@pytest.mark.parametrize("return_kv_cache", [True], ids=["kv_cache"])
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize(
    "input_source",
    [
        "json_prompts",
        "abc_1k",
        "abc_short",
        "p64tok",
        "p960tok",
        "pie960",
        "random",
        "passkey",
        "kv_retrieval",
        "longdialogue_qa_eng",
        "longbook_qa_eng",
    ],
)
@pytest.mark.parametrize("pcc_validation", [True, False], ids=["pcc", "smoke"])
@pytest.mark.parametrize("is_balanced", [True, False], ids=["balanced", "regular"])
@pytest.mark.parametrize(
    "isl_total, dispatch_buffer_capacity_factor",
    [(SEQ_LEN_1K, 8), (SEQ_LEN_25K, 8)],
)
@pytest.mark.parametrize(
    "num_layers",
    [
        5,
        12,
        pytest.param(61, marks=pytest.mark.skipif(not is_galaxy(), reason="Testing entire-prefill only on Galaxy")),
    ],
    ids=["5_layers", "12_layers", "61_layers"],
)
@pytest.mark.parametrize(
    "n_routed_experts, gate_fallback_mode",
    [
        (64, GateComputeMode.HOST_ALL),
        (256, GateComputeMode.HOST_ALL),
        (256, GateComputeMode.DEVICE),
        (256, GateComputeMode.DEVICE_FP32),
    ],
    ids=["e64_host", "e256_host", "e256_device", "e256_device_fp32"],
)
@pytest.mark.parametrize("num_iterations", [1, 25, 2000], ids=["iter1", "iter25", "iter2000"])
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
    is_balanced,
    isl_total,
    dispatch_buffer_capacity_factor,
    num_layers,
    n_routed_experts,
    gate_fallback_mode,
    num_links,
    topology,
    pcc_validation,
    num_iterations,
    input_source,
    use_pretrained,
    return_kv_cache,
    temperature,
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    tokenizer,
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
        f"dispatch_buffer_capacity_factor={dispatch_buffer_capacity_factor}, "
        f"gate_fallback_mode={gate_fallback_mode}, "
        f"input_source={input_source}, pcc_validation={pcc_validation}, "
        f"weights={weight_type}"
    )

    # --- Monkeypatch n_routed_experts ---
    orig_num_routed_experts = DeepSeekV3Config.NUM_ROUTED_EXPERTS
    DeepSeekV3Config.NUM_ROUTED_EXPERTS = n_routed_experts

    padding_side = tokenizer.padding_side

    # --- Cache-aware loading strategy ---
    profiler.start("cache_check")

    # Check cache states
    experts_per_chip = 256 // (mesh_shape[0] * mesh_shape[1]) if use_pretrained else 8
    ttnn_cache_complete = (
        TtPrefillTransformer.check_cache_complete(effective_cache_path, num_layers, experts_per_chip)
        if effective_cache_path
        else False
    )

    cache_key = ReferenceCacheKey(
        weight_type=weight_type,
        input_source=input_source,
        isl_total=isl_total,
        num_layers=num_layers,
        n_routed_experts=n_routed_experts,
        padding_side=padding_side,
    )
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

    # Report cache check timing breakdown
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import report_and_clear

    report_and_clear()

    # --- Create input (needed early for reference computation) ---
    if input_source == "random":
        token_ids = torch.randint(0, config.vocab_size, (1, isl_total), dtype=torch.int64)
        attention_mask = torch.ones(1, isl_total, dtype=torch.int64)
    else:
        profiler.start("tokenization")
        tok = tokenizer
        if input_source == "json_prompts":
            from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

            prompt_text = load_prompts_from_json(str(PROMPTS_PATH))
            prompt_text = prompt_text[0] if isinstance(prompt_text, list) else prompt_text
        elif input_source == "abc_1k":
            from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

            prompt_text = load_prompts_from_json(str(ABC_1K_PATH))
        elif input_source == "abc_short":
            from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

            prompt_text = load_prompts_from_json(str(ABC_SHORT_PATH))
        elif input_source == "p64tok":
            from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

            prompt_text = load_prompts_from_json(str(P64TOK_PATH))
        elif input_source == "p960tok":
            from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

            prompt_text = load_prompts_from_json(str(P960TOK_PATH))
        elif input_source == "pie960":
            from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

            prompt_text = load_prompts_from_json(str(PIE960_PATH))
        elif input_source in INFINITEBENCH_SUBSET_NAMES:
            cached_path = download_infinitebench_subset(input_source)
            with open(cached_path) as f:
                prompt_text = json.load(f)["prompt"]
        else:
            raise ValueError(f"Unknown input_source: {input_source}")
        token_ids, attention_mask, tokens = tokenize_prompt_to_isl(tok, max_isl=isl_total, prompt_text=prompt_text)
        profiler.end("tokenization")
        logger.info(
            f"Tokenized {input_source} input shape: {token_ids.shape}, first 10 tokens: {token_ids[0, :10].tolist()}, last 10 tokens: {token_ids[0, -10:].tolist()}"
        )

    number_of_non_padded_tokens = attention_mask.sum().item()  # should be returned by tokenize..
    logger.info(f"Number of non-padded tokens is: {number_of_non_padded_tokens}")

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
                attention_mask=attention_mask,
                compute_reference=need_to_compute_reference,
                build_ttnn_cache=need_to_load_weights,
                weight_cache_path=effective_cache_path,
                mesh_device=mesh_device,
                seq_len=isl_total,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
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
    # Log program cache size BEFORE creation
    cache_entries_before = mesh_device.num_program_cache_entries()
    logger.info(f"Program cache entries BEFORE transformer creation: {cache_entries_before}")

    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        num_layers=num_layers,
        seq_len=isl_total,
        is_balanced=is_balanced,
        padding_side=padding_side,
        dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=gate_fallback_mode,
        weight_cache_path=effective_cache_path,
    )
    ttnn.ReadDeviceProfiler(mesh_device)
    ttnn.synchronize_device(mesh_device)

    # Log program cache size AFTER creation
    cache_entries_after = mesh_device.num_program_cache_entries()
    logger.info(f"Program cache entries AFTER transformer creation: {cache_entries_after}")
    logger.info(f"Program cache entries ADDED during creation: {cache_entries_after - cache_entries_before}")

    # --- Free memory immediately after transformer creation ---
    del state_dict
    gc.collect()
    logger.info("State dict freed after transformer creation")
    profiler.end("tt_transformer_creation")

    # --- Create external KVPE cache ---
    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
        mesh_device=mesh_device,
        seq_len=isl_total,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
    )

    # create kv_cache dissagregation table
    CHUNK_SIZE_BYTES = 19584  # [1, 1, 32, 576] bfp8
    lookup_table_config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    lookup_table_config.num_layers = num_layers
    lookup_table_config.max_sequence_length = isl_total
    lookup_table_config.num_slots = 1
    lookup_table_config.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    lookup_table_config.chunk_size_bytes = CHUNK_SIZE_BYTES

    # just create atm for demo purposes, don't actually use it
    lookup_table = create_kv_chunk_address_table(
        config=lookup_table_config,
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        seq_len=isl_total,
        sp_axis=sp_axis,
        tt_kvpe_cache=tt_kvpe_cache,
        chunk_size_bytes=CHUNK_SIZE_BYTES,
    )

    # --- Shard token_ids to device ---
    # Reshape [1, isl_total] -> [sp_factor, 1, isl_per_chip] for SP sharding
    if is_balanced == True:
        chunk_order = create_balanced_chunk_order(sp_factor) if is_balanced else None
        token_ids = (
            reorder_tensor_chunks(token_ids.unsqueeze(1).unsqueeze(-1), chunk_order, seq_dim=2).squeeze(1).squeeze(-1)
        )

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
    for i in range(num_iterations):
        logger.info(f"Starting iteration: {i}")
        first_token_id, first_token_prob, tt_intermediates = transformer(
            tt_tokens,
            tt_kvpe_cache,
            number_of_non_padded_tokens=number_of_non_padded_tokens,
            return_intermediates=pcc_validation,
            read_profiler=True,
            temperature=temperature,
        )
        logger.info(f"Starting completion sync on iteration: {i}")
        ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")
    logger.info(f"Forward pass completed. First token: ID={first_token_id}, prob={first_token_prob:.4f}")

    # --- Save intermediate outputs ---

    if pcc_validation:
        assert tt_intermediates is not None, "Expected intermediates dict"
        test_params = {
            "mesh_shape": mesh_shape,
            "isl_total": isl_total,
            "isl_per_chip": isl_per_chip,
            "num_layers": num_layers,
            "n_routed_experts": n_routed_experts,
            "dispatch_buffer_capacity_factor": dispatch_buffer_capacity_factor,
            "gate_fallback_mode": gate_fallback_mode,
            "use_pretrained": use_pretrained,
            "input_source": input_source,
            "topology": str(topology),
            "num_links": num_links,
            "emb_dim": emb_dim,
            "sp_factor": sp_factor,
            "tp_factor": tp_factor,
        }

        assert "norm" in tt_intermediates, "Expected 'norm' in intermediates"
        save_intermediate_output(
            tensor=tt_intermediates["norm"],
            name="norm",
            test_params=test_params,
        )

        assert "lm_head" in tt_intermediates, "Expected 'lm_head' in intermediates"
        save_intermediate_output(
            tensor=tt_intermediates["lm_head"],
            name="lm_head",
            test_params=test_params,
        )

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
        # ref_snapshots is a list of tensors: [embed, layer_0, ..., layer_N, norm, lm_head]
        # tt_intermediates is a dict with keys: "embed", "layer_0", ..., "layer_N", "norm", "lm_head", "first_token"
        pcc_results = []
        ref_labels = ["embed"] + [f"layer_{i}" for i in range(num_layers)] + ["norm", "lm_head"]
        for label, ref_host in zip(ref_labels, ref_snapshots):
            if label not in tt_intermediates:
                logger.error(f"{label:<20s}  Missing from TT intermediates")
                pcc_results.append((label, -1.0))
                continue
            tt_host = tt_intermediates[label]
            try:
                # ignore padded tokens in comparison
                _, pcc = comp_pcc(
                    slice_non_padded(ref_host, number_of_non_padded_tokens, padding_side).float(),
                    slice_non_padded(tt_host, number_of_non_padded_tokens, padding_side).float(),
                )
                logger.debug(f"{label:<20s}  PCC = {pcc:.6f}")
                pcc_results.append((label, pcc))
            except Exception as e:
                logger.error(f"{label:<20s}  PCC comparison failed: {e}")
                pcc_results.append((label, -1.0))

        # Per-layer KVPE PCC comparison — read back from external cache
        if do_return_kv:
            tt_kvpe_all = ttnn.to_torch(
                tt_kvpe_cache,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
            ).to(torch.bfloat16)
            # Shape: [num_layers, tp_factor, seq_total, head_dim] — take first TP replica
            tt_kvpe_all_layers = tt_kvpe_all[:, :1, :, :]
            if is_balanced:
                tt_kvpe_all_layers = reverse_reorder_tensor_chunks(tt_kvpe_all_layers, chunk_order, seq_dim=2)
            kv_lora_rank = config.kv_lora_rank
            for i, ref_kvpe in enumerate(ref_kvpe_list):
                tt_kvpe_layer = tt_kvpe_all_layers[i : i + 1, :, :, :]
                label = f"layer_{i}_kvpe"
                try:
                    # ignore padded tokens in comparison
                    _, kv_pcc = comp_pcc(
                        slice_non_padded(
                            ref_kvpe[..., :kv_lora_rank], number_of_non_padded_tokens, padding_side
                        ).float(),
                        slice_non_padded(
                            tt_kvpe_layer[..., :kv_lora_rank], number_of_non_padded_tokens, padding_side
                        ).float(),
                    )
                    # ignore padded tokens in comparison
                    _, pe_pcc = comp_pcc(
                        slice_non_padded(
                            ref_kvpe[..., kv_lora_rank:], number_of_non_padded_tokens, padding_side
                        ).float(),
                        slice_non_padded(
                            tt_kvpe_layer[..., kv_lora_rank:], number_of_non_padded_tokens, padding_side
                        ).float(),
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

        # Log first token info (returned value is for first temperature in list)
        tok = tokenizer

        # Decode token to string
        token_text = tok.decode([first_token_id]) if tok else "N/A"
        first_temp = temperature[0] if isinstance(temperature, list) else temperature
        logger.info(
            f"First Token: ID={first_token_id} [{repr(token_text)}] prob={first_token_prob*100:.1f}% temp={first_temp}"
        )

        # Log all temperature results from intermediates
        if tt_intermediates and "first_token" in tt_intermediates:
            for result in tt_intermediates["first_token"]:
                tid = result["token_id"]
                tprob = result["probability"]
                ttemp = result["temperature"]
                ttext = tok.decode([tid]) if tok else "N/A"
                logger.debug(f"First Token: ID={tid} [{repr(ttext)}] prob={tprob*100:.1f}% temp={ttemp}")
                # Print top5
                if "top5" in result:
                    for i, t5 in enumerate(result["top5"]):
                        t5_id = t5["token_id"]
                        t5_prob = t5["probability"]
                        t5_text = tok.decode([t5_id]) if tok else "N/A"
                        logger.debug(f"  top{i+1}: ID={t5_id} [{repr(t5_text)}] prob={t5_prob*100:.1f}%")

        # Store failures for deferred check (after timing report)
        has_pcc_failures = len(failures) > 0

        if not has_pcc_failures:
            logger.success(
                f"TtPrefillTransformer PCC test passed "
                f"(num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
                f"gate_fallback_mode={gate_fallback_mode}, "
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
            f"gate_fallback_mode={gate_fallback_mode}, "
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


# ---------------------------------------------------------------------------
# Trace-based PCC test
# ---------------------------------------------------------------------------


def plot_pcc_results(
    pcc_results: list[tuple[str, float]],
    trace_dir: Path,
    dataset_name: str = "",
    gate_fallback_mode: str = "",
    threshold: float = TRACE_PCC_THRESHOLD,
    filename: str = "pcc_results.png",
    title_prefix: str = "",
    annotation: str = "",
) -> None:
    """Save a bar chart of per-layer and logits PCC values to disk next to the trace directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filtered = [
        (label, pcc)
        for label, pcc in pcc_results
        if label.startswith("layer_")
        and "_kvpe" not in label
        and "_kv" not in label
        and "_pe" not in label
        or label in ("lm_head", "logits")
    ]
    if not filtered:
        logger.warning("No layer/logits PCC results to plot")
        return

    labels = [label for label, _ in filtered]
    values = [pcc for _, pcc in filtered]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.35), 6))
    colors = ["#2ecc71" if v > threshold else "#e74c3c" for v in values]
    ax.bar(range(len(values)), values, color=colors, edgecolor="none", width=0.8)

    ax.axhline(y=threshold, color="orange", linestyle="--", linewidth=1, label=f"threshold={threshold}")
    ax.set_ylabel("PCC")
    ax.set_xlabel("Layer")
    title = f"{dataset_name}    gate_mode = {gate_fallback_mode}"
    if title_prefix:
        title = f"{title_prefix}\n{title}"
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(min(0, min(values) - 0.02), 1.01)
    ax.legend(loc="lower left")
    if annotation:
        ax.text(
            0.99,
            0.02,
            annotation,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )
    fig.tight_layout()

    out_path = trace_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"PCC plot saved to {out_path}")


@pytest.mark.skipif(not is_blackhole(), reason="Requires Blackhole.")
@pytest.mark.parametrize(
    "trace_dir, max_seq_len, pad_left",
    [
        # (ILLIAD_1024_TRACE, None, False),
        (LONGBOOK_QA_ENG_25024, None, False),
        # (ABC_1k_PADD_RIGHT_1024, 1024, False),
        # (ABC_1k_PADD_LEFT_1024, 1024, True),
    ],
    ids=["longbook_25k_qa_eng"],
)
@pytest.mark.parametrize(
    "num_layers",
    [
        61,
    ],
)
@pytest.mark.parametrize(
    "n_routed_experts, capacity_factor, gate_fallback_mode",
    [
        (256, 2, GateComputeMode.HOST_ALL),
        (256, 2, GateComputeMode.DEVICE),
    ],
    ids=["e256_cf32_host", "e256_cf32_device"],
)
@pytest.mark.parametrize(
    "dequant_method",
    ["tt"],
    ids=["dequant_tt"],
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
def test_prefill_transformer_from_trace(
    config_only,
    mesh_device,
    device_params,
    trace_dir,
    max_seq_len,
    pad_left,
    num_layers,
    n_routed_experts,
    capacity_factor,
    gate_fallback_mode,
    dequant_method,
    num_links,
    topology,
    weight_cache_path,
    request,
):
    """
    PCC test using pre-computed reference tensors from a bit_sculpt debug trace.

    Loads real intermediate hidden states (per-layer decoder outputs) and KVPE
    from safetensors files, runs the TT model on the same input tokens, and
    compares per-layer PCC.

    TTNN weight cache is built automatically if missing (requires pretrained
    weights accessible via model_path fixture).
    """
    torch.manual_seed(42)

    profiler.clear()
    profiler.start("total_test_time")

    config = config_only
    trace = load_debug_trace(trace_dir, num_layers=num_layers)

    n_real_tokens = trace.token_ids.shape[1]

    if max_seq_len is not None and max_seq_len > n_real_tokens:
        tok = request.getfixturevalue("tokenizer")
        pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        pad_len = max_seq_len - n_real_tokens
        padding = torch.full((1, pad_len), pad_token_id, dtype=torch.int64)

        if pad_left:
            token_ids = torch.cat([padding, trace.token_ids], dim=1)
        else:
            token_ids = torch.cat([trace.token_ids, padding], dim=1)

        isl_total = max_seq_len
        logger.info(
            f"Padded {n_real_tokens} tokens to {max_seq_len} "
            f"({'left' if pad_left else 'right'}-padded, pad_token_id={pad_token_id})"
        )
    else:
        token_ids = trace.token_ids
        isl_total = n_real_tokens

    config.max_seq_len = isl_total

    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    sp_factor = mesh_shape[sp_axis]
    tp_factor = mesh_shape[tp_axis]
    emb_dim = config.hidden_size
    isl_per_chip = isl_total // sp_factor

    logger.info(
        f"Trace-based test: trace={trace_dir.name}, isl={isl_total}, "
        f"num_layers={num_layers}, mesh={mesh_shape}, "
        f"n_routed_experts={n_routed_experts}, capacity_factor={capacity_factor}, "
        f"gate_fallback_mode={gate_fallback_mode}"
    )

    orig_num_routed_experts = DeepSeekV3Config.NUM_ROUTED_EXPERTS
    DeepSeekV3Config.NUM_ROUTED_EXPERTS = n_routed_experts

    # --- Weight cache: build if missing, reuse if present ---
    if weight_cache_path is not None:
        rows, cols = mesh_shape
        cache_subdir = f"v2/{rows}x{cols}" if dequant_method == "hf" else f"{rows}x{cols}"
        effective_cache_path = weight_cache_path / cache_subdir
        effective_cache_path.mkdir(parents=True, exist_ok=True)
    else:
        pytest.skip("weight_cache_path is None (pretrained weights unavailable)")

    profiler.start("cache_check")
    experts_per_chip = 256 // (mesh_shape[0] * mesh_shape[1])
    ttnn_cache_complete = TtPrefillTransformer.check_cache_complete(effective_cache_path, num_layers, experts_per_chip)
    profiler.end("cache_check")

    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import report_and_clear

    report_and_clear()

    logger.info(f"Cache status: TTNN={ttnn_cache_complete}, dequant_method={dequant_method}")
    logger.info(f"Cache lookup path: {effective_cache_path}")

    if not ttnn_cache_complete:
        logger.info(f"TTNN weight cache incomplete — building from HF weights (dequant={dequant_method})...")
        model_path = request.getfixturevalue("model_path")
        profiler.start("cache_build")
        load_and_compute_layer_by_layer(
            model_path=model_path,
            config=config,
            num_layers=num_layers,
            token_ids=None,
            attention_mask=None,
            compute_reference=False,
            build_ttnn_cache=True,
            weight_cache_path=effective_cache_path,
            mesh_device=mesh_device,
            seq_len=isl_total,
            num_links=num_links,
            topology=topology,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            gate_fallback_mode=gate_fallback_mode,
            dequant_method=dequant_method,
        )
        gc.collect()
        profiler.end("cache_build")
        logger.info(f"TTNN weight cache built at {effective_cache_path}")
    else:
        logger.info(f"TTNN weight cache found at {effective_cache_path}")

    # --- TT transformer ---
    cache_entries_before = mesh_device.num_program_cache_entries()
    logger.info(f"Program cache entries BEFORE transformer creation: {cache_entries_before}")

    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        state_dict={},
        num_layers=num_layers,
        seq_len=isl_total,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=gate_fallback_mode,
        dispatch_buffer_capacity_factor=capacity_factor,
        weight_cache_path=effective_cache_path,
    )
    ttnn.ReadDeviceProfiler(mesh_device)
    ttnn.synchronize_device(mesh_device)

    cache_entries_after = mesh_device.num_program_cache_entries()
    logger.info(f"Program cache entries AFTER transformer creation: {cache_entries_after}")
    logger.info(f"Program cache entries ADDED during creation: {cache_entries_after - cache_entries_before}")
    profiler.end("tt_transformer_creation")

    # --- Create external KVPE cache ---
    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
        mesh_device=mesh_device,
        seq_len=isl_total,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
    )

    # --- Shard token_ids to device ---
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
    first_token_id, first_token_prob, tt_intermediates = transformer(
        tt_tokens,
        tt_kvpe_cache,
        number_of_non_padded_tokens=isl_total,
        return_intermediates=True,
        read_profiler=True,
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")
    logger.info("Forward pass completed successfully")

    # --- PCC check: TT intermediates vs trace references ---
    profiler.start("pcc_validation")
    pcc_results = []

    for label, tt_host in tt_intermediates.items():
        if label not in trace.ref_snapshots:
            logger.debug(f"Skipping {label} (no trace reference)")
            continue
        ref_host = trace.ref_snapshots[label]
        try:
            _, pcc = comp_pcc(ref_host.float(), tt_host.float())
            logger.debug(f"{label:<20s}  PCC = {pcc:.6f}")
            pcc_results.append((label, pcc))
        except Exception as e:
            logger.error(f"{label:<20s}  PCC comparison failed: {e}")
            pcc_results.append((label, -1.0))

    # --- Logits PCC check ---
    # tt_intermediates["lm_head"] has shape [sp_factor, 1, TILE_SIZE, vocab_size]
    # trace.logits is [1, vocab_size] — last real token only.
    # Extract the matching last-token logits using the same device/offset logic as the LM head.
    if trace.logits is not None and "lm_head" in tt_intermediates:
        from models.demos.deepseek_v3_d_p.tt.mla.utils import global_to_local_token_id

        try:
            tt_logits_full = tt_intermediates["lm_head"]
            global_token_id = isl_total - 1
            device_id, local_token_id = global_to_local_token_id(
                global_token_id,
                sp_factor,
                isl_total,
                is_balanced=False,
            )
            token_offset = local_token_id % ttnn.TILE_SIZE
            tt_last_token_logits = tt_logits_full[device_id, 0, token_offset, :].unsqueeze(0)
            logger.debug(
                f"Logits extraction: full={list(tt_logits_full.shape)}, "
                f"device_id={device_id}, local_token_id={local_token_id}, token_offset={token_offset}, "
                f"extracted={list(tt_last_token_logits.shape)}, trace={list(trace.logits.shape)}"
            )
            _, logits_pcc = comp_pcc(trace.logits.float(), tt_last_token_logits.float())
            logger.info(f"{'logits':<20s}  PCC = {logits_pcc:.6f}")
            pcc_results.append(("logits", logits_pcc))
        except Exception as e:
            logger.error(f"{'logits':<20s}  PCC comparison failed: {e}")
            pcc_results.append(("logits", -1.0))

    # --- First token comparison ---
    # next_token_id/text may be in metadata.json or output_metadata.json
    ref_token_id = trace.metadata.get("next_token_id")
    ref_token_text = trace.metadata.get("next_token_text")
    if ref_token_id is None or ref_token_text is None:
        output_meta_path = trace_dir / "output_metadata.json"
        if output_meta_path.exists():
            import json

            with open(output_meta_path) as f:
                output_meta = json.load(f)
            ref_token_id = ref_token_id or output_meta.get("next_token_id")
            ref_token_text = ref_token_text or output_meta.get("next_token_text")
    if ref_token_text is None:
        ref_token_text = "N/A"
    token_match = first_token_id == ref_token_id if ref_token_id is not None else None
    logger.info(
        f"First token: TT={first_token_id} (prob={first_token_prob:.4f}), "
        f"Trace={ref_token_id} [{repr(ref_token_text)}], "
        f"Match={'YES' if token_match else 'NO' if token_match is not None else 'N/A'}"
    )

    profiler.end("pcc_validation")

    # --- Summary table ---
    logger.info(f"\n{'='*50}")
    logger.info(f"{'Stage':<20s}  {'PCC':>10s}  {'Status':>8s}")
    logger.info(f"{'-'*50}")
    failures = []
    for label, pcc in pcc_results:
        status = "PASS" if pcc > TRACE_PCC_THRESHOLD else ("FAIL" if pcc >= 0 else "ERROR")
        logger.info(f"{label:<20s}  {pcc:>10.6f}  {status:>8s}")
        if pcc <= TRACE_PCC_THRESHOLD:
            failures.append((label, pcc))
    logger.info(f"{'='*50}")

    profiler.end("total_test_time")

    # --- Timing report ---
    logger.info(f"\n{'='*60}")
    logger.info("Timing Report")
    logger.info(f"{'='*60}")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")

    first_token_annotation = (
        f"First token: TT={first_token_id} (prob={first_token_prob:.4f}), "
        f"Trace={ref_token_id} [{repr(ref_token_text)}], "
        f"Match={'YES' if token_match else 'NO' if token_match is not None else 'N/A'}"
    )
    gate_suffix = "_device_gate" if gate_fallback_mode == GateComputeMode.DEVICE else "_host_gate"
    plot_pcc_results(
        pcc_results,
        trace_dir,
        dataset_name=trace_dir.name,
        gate_fallback_mode=str(gate_fallback_mode),
        filename=f"pcc_results{gate_suffix}.png",
        annotation=first_token_annotation,
    )

    DeepSeekV3Config.NUM_ROUTED_EXPERTS = orig_num_routed_experts

    if failures:
        pcc_failure_msg = "; ".join(f"{label}: {pcc:.6f}" for label, pcc in failures)
        pytest.fail(f"PCC below {TRACE_PCC_THRESHOLD} at: {pcc_failure_msg}")

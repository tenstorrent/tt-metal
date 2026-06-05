# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TtPrefillTransformer — verifies composition of embed -> [block x N] -> norm.

Validates output shapes and PCC against torch reference.

Reference sources are checked in priority order:
1. Debug trace on disk (pre-computed safetensors from a known-good run)
2. Reference cache (previously computed and cached PyTorch outputs)
3. HF model computation (creates HF DeepseekV3Model and runs forward on the fly)

Parametrized over:
- use_pretrained: real pretrained weights from DeepSeek-R1-0528 vs random weights
- input_source: "random", "json_prompts", or InfiniteBench subset (passkey, kv_retrieval, etc.)
- pcc_validation: per-stage PCC check (via return_intermediates) vs shape-only smoke test
- n_routed_experts / gate_fallback_mode: MoE configurations
"""

import gc
import json
import os
import time

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
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    create_kv_chunk_address_table,
    init_kvpe_cache,
)
from models.demos.deepseek_v3_d_p.utils.pcc_plot_utils import generate_pcc_plots, write_pcc_summary
from models.demos.deepseek_v3_d_p.utils.test_utils import save_intermediate_output
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ABC_1K_PATH,
    ABC_SHORT_PATH,
    P64TOK_PATH,
    P960TOK_PATH,
    PIE960_PATH,
    PROMPT_1K_PATH,
    ReferenceCacheKey,
    check_first_token_match,
    check_reference_cache_exists,
    create_hf_model,
    download_infinitebench_subset,
    extract_tt_state_dict,
    find_trace_dir,
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
TRACE_PCC_THRESHOLD_HOST = 0.96
TRACE_PCC_THRESHOLD_DEVICE_BF16 = 0.88
TRACE_PCC_THRESHOLD_DEVICE_FP32 = 0.95
# Determinism: every iteration is expected to be (near-)bit-identical to iter 0.
DETERMINISM_PCC_THRESHOLD = 0.9999

# Input sources: "random" = random token IDs, "json_prompts" = test_prompts_1024.json,
# or any InfiniteBench subset name (downloaded on first use via infinitebench_prompt fixture).
INFINITEBENCH_SUBSET_NAMES = {"passkey", "kv_retrieval", "longdialogue_qa_eng", "longbook_qa_eng"}
SEQ_LEN_1K = 1024
SEQ_LEN_25K = 25600


def _compare_intermediate_pcc(reference_items, tt_intermediates, number_of_non_padded_tokens, padding_side):
    pcc_results = []
    for label, ref_host in reference_items:
        if label not in tt_intermediates:
            logger.error(f"{label:<20s}  Missing from TT intermediates")
            pcc_results.append((label, -1.0))
            continue

        tt_host = tt_intermediates[label]
        try:
            _, pcc = comp_pcc(
                slice_non_padded(ref_host, number_of_non_padded_tokens, padding_side).float(),
                slice_non_padded(tt_host, number_of_non_padded_tokens, padding_side).float(),
            )
            logger.debug(f"{label:<20s}  PCC = {pcc:.6f}")
            pcc_results.append((label, pcc))
        except Exception as e:
            logger.error(f"{label:<20s}  PCC comparison failed: {e}")
            pcc_results.append((label, -1.0))
    return pcc_results


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
@pytest.mark.parametrize("determinism_check", [False, True], ids=["no_determinism", "with_determinism"])
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
@pytest.mark.parametrize("num_iterations", [1, 5, 25, 2000], ids=["iter1", "iter5", "iter25", "iter2000"])
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
    determinism_check,
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

    # determinism_check and pcc_validation are mutually exclusive validation modes:
    # determinism compares iter N against iter 0; pcc_validation compares against host.
    if determinism_check and pcc_validation:
        pytest.skip("determinism_check and pcc_validation are mutually exclusive — pick one validation mode")

    # Determinism check needs at least 2 iterations (iter 0 is the baseline)
    if determinism_check and num_iterations < 2:
        pytest.skip("determinism_check requires num_iterations >= 2 (iter 0 is the baseline)")

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

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(DeepSeekV3Config, "NUM_ROUTED_EXPERTS", n_routed_experts)

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

    # Priority 1: debug trace on disk
    trace = None
    trace_dir = (
        find_trace_dir(input_source, isl_total, padding_side, use_pretrained, n_routed_experts)
        if pcc_validation
        else None
    )
    if trace_dir is not None:
        trace = load_debug_trace(trace_dir, num_layers=num_layers)
        logger.info(
            f"Loaded debug trace from {trace_dir} "
            f"(trace n_layers={trace.metadata.get('n_layers')}, test num_layers={num_layers})"
        )

    cache_key = ReferenceCacheKey(
        weight_type=weight_type,
        input_source=input_source,
        isl_total=isl_total,
        num_layers=num_layers,
        n_routed_experts=n_routed_experts,
        padding_side=padding_side,
    )
    ref_cache_exists = check_reference_cache_exists(cache_key) if (pcc_validation and trace is None) else False

    logger.info(
        f"Cache status: TTNN={ttnn_cache_complete}, Trace={'YES' if trace else 'NO'}, Reference={ref_cache_exists}"
    )

    # Determine what we need to load
    need_to_load_weights = not ttnn_cache_complete
    need_to_compute_reference = pcc_validation and trace is None and not ref_cache_exists
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
    if trace is not None:
        # When a trace is selected, the TT model must run on the exact token IDs the
        # trace was generated from, otherwise PCC compares two different inputs.
        token_ids = trace.token_ids
        assert (
            token_ids.shape[1] == isl_total
        ), f"Trace token count {token_ids.shape[1]} does not match isl_total {isl_total}"
        attention_mask = torch.ones_like(token_ids)
        logger.info(f"Using {isl_total} tokens from trace (skipping tokenization)")
    elif input_source == "random":
        token_ids = torch.randint(0, config.vocab_size, (1, isl_total), dtype=torch.int64)
        attention_mask = torch.ones(1, isl_total, dtype=torch.int64)
    else:
        profiler.start("tokenization")
        tok = tokenizer
        if input_source == "json_prompts":
            from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

            prompt_text = load_prompts_from_json(str(PROMPT_1K_PATH))
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
        lm_head_is_column_parallel=True,
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

    # --- Forward (with per-iteration validation, buffered) ---
    # Two validation modes are supported, both producing per-iteration PCC numbers:
    #   * pcc_validation: compare each iter against host reference (trace/cache).
    #   * determinism_check: compare iter N>=1 against the iter-0 baseline.
    need_intermediates = pcc_validation or determinism_check
    do_return_kv = need_intermediates and return_kv_cache

    # Pre-compute validation setup once before the iteration loop:
    # threshold + reference items. This avoids re-loading per iteration.
    threshold = None
    reference_items_list = None
    trace_full_model = False
    # Determinism-mode baselines captured from iter 0 (None until then).
    baseline_logits = None
    baseline_first_token_id = None
    if determinism_check:
        threshold = DETERMINISM_PCC_THRESHOLD
        logger.info(f"Determinism check threshold: {threshold} (baseline = iter 0)")
    elif pcc_validation:
        if trace is not None:
            if gate_fallback_mode == GateComputeMode.DEVICE:
                threshold = TRACE_PCC_THRESHOLD_DEVICE_BF16
            elif gate_fallback_mode == GateComputeMode.DEVICE_FP32:
                threshold = TRACE_PCC_THRESHOLD_DEVICE_FP32
            elif gate_fallback_mode == GateComputeMode.HOST_ALL:
                threshold = TRACE_PCC_THRESHOLD_HOST
            else:
                threshold = TRACE_PCC_THRESHOLD
        elif use_pretrained and input_source != "random":
            threshold = 0.97
        elif use_pretrained:
            threshold = 0.95
        elif n_routed_experts < 256:
            threshold = 0.985
        else:
            threshold = PCC_THRESHOLD  # 0.99
        logger.info(f"PCC threshold: {threshold} (ref_source={'trace' if trace else 'host'})")

        if trace is not None:
            reference_items_list = list(trace.ref_snapshots.items())
        else:
            if ref_snapshots is None:
                logger.info("Loading reference from cache...")
                ref_snapshots, ref_kvpe_list = load_reference_cache(cache_key)
            ref_labels = ["embed"] + [f"layer_{li}" for li in range(num_layers)] + ["norm", "lm_head"]
            reference_items_list = list(zip(ref_labels, ref_snapshots))

        trace_full_model = trace is not None and num_layers == trace.metadata.get("n_layers")
        if trace is not None and not trace_full_model:
            logger.info(
                f"Skipping trace logits/first-token checks: "
                f"num_layers={num_layers} != trace n_layers={trace.metadata.get('n_layers')}"
            )

    profiler.start("tt_forward")
    logger.info("Running TtPrefillTransformer forward...")
    # Buffer of per-iteration PCC results: list of (iter_idx, list[(label, pcc)]).
    # PCC is computed every iteration but assertions are deferred until after
    # all iterations complete and the full summary has been printed.
    per_iter_pcc_buffer = []
    for i in range(num_iterations):
        logger.info(f"Starting iteration: {i}")
        start_time = time.time()
        first_token_id, first_token_prob, tt_intermediates = transformer(
            tt_tokens,
            tt_kvpe_cache,
            number_of_non_padded_tokens=number_of_non_padded_tokens,
            return_intermediates=need_intermediates,
            read_profiler=False,
            temperature=temperature,
        )
        logger.info(f"Starting completion sync on iteration: {i}")
        ttnn.synchronize_device(mesh_device)
        end_time = time.time()
        logger.info(f"Iteration {i}: {end_time - start_time} seconds")

        if need_intermediates:
            assert tt_intermediates is not None, "Expected intermediates dict"
            profiler.start("pcc_validation")
            iter_pcc = []

            # Determinism iter 0: snapshot baselines (intermediates, KVPE cache, logits,
            # first-token) for later iterations to compare against. No PCC this iter.
            if determinism_check and i == 0:
                excluded = {"first_token", "logits"}
                reference_items_list = [
                    (label, val.clone().detach())
                    for label, val in tt_intermediates.items()
                    if isinstance(val, torch.Tensor) and label not in excluded
                ]
                if do_return_kv:
                    tt_kvpe_all = ttnn.to_torch(
                        tt_kvpe_cache,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
                    ).to(torch.bfloat16)
                    tt_kvpe_all_layers = tt_kvpe_all[:, :1, :, :]
                    if is_balanced:
                        tt_kvpe_all_layers = reverse_reorder_tensor_chunks(tt_kvpe_all_layers, chunk_order, seq_dim=2)
                    ref_kvpe_list = [
                        tt_kvpe_all_layers[layer_idx : layer_idx + 1, :, :, :].clone()
                        for layer_idx in range(num_layers)
                    ]
                if "logits" in tt_intermediates and isinstance(tt_intermediates["logits"], torch.Tensor):
                    baseline_logits = tt_intermediates["logits"].clone().detach()
                baseline_first_token_id = first_token_id
                logger.info(
                    f"Iteration {i}: captured baseline for determinism check "
                    f"({len(reference_items_list)} intermediate tensors"
                    + (f", {num_layers} KVPE layers" if do_return_kv else "")
                    + (", logits" if baseline_logits is not None else "")
                    + ")"
                )
                per_iter_pcc_buffer.append((i, iter_pcc))
                profiler.end("pcc_validation")
                continue

            iter_pcc.extend(
                _compare_intermediate_pcc(
                    reference_items_list,
                    tt_intermediates,
                    number_of_non_padded_tokens,
                    padding_side,
                )
            )

            if do_return_kv and ref_kvpe_list is not None:
                tt_kvpe_all = ttnn.to_torch(
                    tt_kvpe_cache,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
                ).to(torch.bfloat16)
                tt_kvpe_all_layers = tt_kvpe_all[:, :1, :, :]
                if is_balanced:
                    tt_kvpe_all_layers = reverse_reorder_tensor_chunks(tt_kvpe_all_layers, chunk_order, seq_dim=2)
                kv_lora_rank = config.kv_lora_rank
                for layer_idx, ref_kvpe in enumerate(ref_kvpe_list):
                    tt_kvpe_layer = tt_kvpe_all_layers[layer_idx : layer_idx + 1, :, :, :]
                    label = f"layer_{layer_idx}_kvpe"
                    try:
                        _, kv_pcc = comp_pcc(
                            slice_non_padded(
                                ref_kvpe[..., :kv_lora_rank], number_of_non_padded_tokens, padding_side
                            ).float(),
                            slice_non_padded(
                                tt_kvpe_layer[..., :kv_lora_rank], number_of_non_padded_tokens, padding_side
                            ).float(),
                        )
                        _, pe_pcc = comp_pcc(
                            slice_non_padded(
                                ref_kvpe[..., kv_lora_rank:], number_of_non_padded_tokens, padding_side
                            ).float(),
                            slice_non_padded(
                                tt_kvpe_layer[..., kv_lora_rank:], number_of_non_padded_tokens, padding_side
                            ).float(),
                        )
                        logger.debug(f"iter {i} {label:<20s}  KV PCC = {kv_pcc:.6f}, PE PCC = {pe_pcc:.6f}")
                        iter_pcc.append((f"{label}_kv", kv_pcc))
                        iter_pcc.append((f"{label}_pe", pe_pcc))
                    except Exception as e:
                        logger.error(f"iter {i} {label:<20s}  KVPE PCC comparison failed: {e}")
                        iter_pcc.append((f"{label}_kv", -1.0))
                        iter_pcc.append((f"{label}_pe", -1.0))

            # Logits PCC — host trace logits in pcc_validation mode, iter-0 logits in determinism mode.
            if determinism_check:
                if baseline_logits is not None and "logits" in tt_intermediates:
                    try:
                        _, logits_pcc = comp_pcc(baseline_logits.float(), tt_intermediates["logits"].float())
                        logger.debug(f"iter {i} {'logits':<20s}  PCC = {logits_pcc:.6f}")
                        iter_pcc.append(("logits", logits_pcc))
                    except Exception as e:
                        logger.error(f"iter {i} {'logits':<20s}  PCC comparison failed: {e}")
                        iter_pcc.append(("logits", -1.0))
                # First-token determinism: ID must exactly match the baseline. Use 1.0/-1.0
                # so it surfaces in the per-iter PASS/ERROR table alongside PCC entries.
                if first_token_id == baseline_first_token_id:
                    iter_pcc.append(("first_token_id", 1.0))
                else:
                    logger.error(
                        f"iter {i} first_token mismatch: {first_token_id} (baseline {baseline_first_token_id})"
                    )
                    iter_pcc.append(("first_token_id", -1.0))
            elif trace_full_model and trace.logits is not None and "logits" in tt_intermediates:
                try:
                    _, logits_pcc = comp_pcc(trace.logits.float(), tt_intermediates["logits"].float())
                    logger.debug(f"iter {i} {'logits':<20s}  PCC = {logits_pcc:.6f}")
                    iter_pcc.append(("logits", logits_pcc))
                except Exception as e:
                    logger.error(f"iter {i} {'logits':<20s}  PCC comparison failed: {e}")
                    iter_pcc.append(("logits", -1.0))

            per_iter_pcc_buffer.append((i, iter_pcc))
            profiler.end("pcc_validation")

    profiler.end("tt_forward")
    logger.info(f"Forward pass completed. First token: ID={first_token_id}, prob={first_token_prob:.4f}")

    # --- Save intermediate outputs (from last iteration) ---

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

    # --- PCC summary (print buffered per-iteration results, then defer assertion) ---
    if need_intermediates:
        ref_source = "iter0_baseline" if determinism_check else ("trace" if trace else "host")
        failures = []  # list of (iter_idx, label, pcc)
        for iter_idx, iter_pcc in per_iter_pcc_buffer:
            logger.info(f"\n{'='*60}")
            if determinism_check and iter_idx == 0:
                logger.info(f"Iteration {iter_idx} (baseline — no comparison)")
                logger.info(f"{'='*60}")
                continue
            logger.info(f"Iteration {iter_idx} PCC results (ref={ref_source})")
            logger.info(f"{'Stage':<20s}  {'PCC':>10s}  {'Status':>8s}")
            logger.info(f"{'-'*60}")
            for label, pcc in iter_pcc:
                status = "PASS" if pcc > threshold else ("FAIL" if pcc >= 0 else "ERROR")
                logger.info(f"{label:<20s}  {pcc:>10.6f}  {status:>8s}")
                if pcc <= threshold:
                    failures.append((iter_idx, label, pcc))
            logger.info(f"{'='*60}")

        # Use last iteration's PCC for CI summary (preserves existing downstream behavior).
        pcc_results = per_iter_pcc_buffer[-1][1] if per_iter_pcc_buffer else []

        # --- First token info ---
        tok = tokenizer
        token_text = tok.decode([first_token_id]) if tok else "N/A"
        first_temp = temperature[0] if isinstance(temperature, list) else temperature
        logger.info(
            f"First Token: ID={first_token_id} [{repr(token_text)}] prob={first_token_prob*100:.1f}% temp={first_temp}"
        )

        # First-token match against trace metadata (full-layer trace only — host PCC mode)
        if pcc_validation and trace_full_model:
            token_match = check_first_token_match(trace, trace_dir, first_token_id, first_token_prob)
            if token_match is False:
                failures.append((num_iterations - 1, "first_token_match", -1.0))

        # Log all temperature results from intermediates
        if tt_intermediates and "first_token" in tt_intermediates:
            for result in tt_intermediates["first_token"]:
                tid = result["token_id"]
                tprob = result["probability"]
                ttemp = result["temperature"]
                ttext = tok.decode([tid]) if tok else "N/A"
                logger.debug(f"First Token: ID={tid} [{repr(ttext)}] prob={tprob*100:.1f}% temp={ttemp}")
                if "top5" in result:
                    for i, t5 in enumerate(result["top5"]):
                        t5_id = t5["token_id"]
                        t5_prob = t5["probability"]
                        t5_text = tok.decode([t5_id]) if tok else "N/A"
                        logger.debug(f"  top{i+1}: ID={t5_id} [{repr(t5_text)}] prob={t5_prob*100:.1f}%")

        has_pcc_failures = len(failures) > 0
        mode_label = "determinism" if determinism_check else "PCC"

        if not has_pcc_failures:
            logger.success(
                f"TtPrefillTransformer {mode_label} test passed across {num_iterations} iteration(s) "
                f"(num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
                f"gate_fallback_mode={gate_fallback_mode}, "
                f"weights={weight_type}, ref_source={ref_source})"
            )
        else:
            pcc_failure_msg = "; ".join(f"iter {it} {label}: {pcc:.6f}" for it, label, pcc in failures)
            logger.error(
                f"TtPrefillTransformer {mode_label} test has failures "
                f"(num_layers={num_layers}, failures={len(failures)} across {num_iterations} iteration(s))"
            )
    else:
        pcc_results = []
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

    # --- CI PCC summary (Mermaid charts + markdown table for $GITHUB_STEP_SUMMARY) ---
    if pcc_results:
        output_pcc = {}
        kvpe_kv_pcc = {}
        kvpe_pe_pcc = {}
        for label, pcc in pcc_results:
            if "_kv" in label:
                kvpe_kv_pcc[label] = pcc
            elif "_pe" in label:
                kvpe_pe_pcc[label] = pcc
            else:
                output_pcc[label] = pcc

        summary_result = {
            "pcc": (output_pcc, kvpe_kv_pcc, kvpe_pe_pcc),
            "num_layers": num_layers,
            "isl_total": isl_total,
            "weight_type": weight_type,
            "input_source": trace_dir.name if trace_dir else input_source,
            "mesh_shape": mesh_shape,
            "n_routed_experts": n_routed_experts,
            "capacity_factor": dispatch_buffer_capacity_factor,
            "gate_fallback_mode": gate_fallback_mode,
            "threshold": threshold,
        }
        write_pcc_summary(summary_result, threshold=threshold)
        if not os.getenv("GITHUB_ACTIONS") and trace_dir is not None:
            generate_pcc_plots(summary_result, output_dir=str(trace_dir))

    # Deferred PCC failure check (after timing report)
    if need_intermediates and has_pcc_failures:
        pytest.fail(f"PCC below {threshold} at: {pcc_failure_msg}")

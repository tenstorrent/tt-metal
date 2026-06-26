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
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tests.conftest import FABRIC_2D_PREFILL_BLOCK_MESH_PARAMS
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
    create_kv_chunk_address_table_ds,
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
    PROMPT_25K_PATH,
    ReferenceCacheKey,
    check_first_token_match,
    check_first_token_match_host_ref,
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
from tests.ttnn.utils_for_testing import assert_equal, comp_pcc

PCC_THRESHOLD = 0.99
TRACE_PCC_THRESHOLD = 0.97
TRACE_PCC_THRESHOLD_HOST = 0.96
TRACE_PCC_THRESHOLD_DEVICE_BF16 = 0.88
TRACE_PCC_THRESHOLD_DEVICE_FP32 = 0.95
# Determinism: every iteration is expected to match the iter-0 baseline near-bit-exactly.
DETERMINISM_PCC_THRESHOLD = 1.0

# Input sources: "random" = random token IDs, "json_prompts" = test_prompts_1024.json,
# or any InfiniteBench subset name (downloaded on first use via infinitebench_prompt fixture).
INFINITEBENCH_SUBSET_NAMES = {"passkey", "kv_retrieval", "longdialogue_qa_eng", "longbook_qa_eng"}
SEQ_LEN_1K = 1024
SEQ_LEN_5K = 5120
SEQ_LEN_25K = 25600


def _compare_intermediate_pcc(reference_items, tt_intermediates, number_of_non_padded_tokens, padding_side):
    pcc_results = []
    for label, ref_host in reference_items:
        # For lm_head TT only emits logits at the next-token position, not the full sequence.
        # Compare the single meaningful position against the same slice of the full-seq reference.
        if label == "lm_head":
            tt_host = tt_intermediates.get("logits")
            if tt_host is None:
                logger.error(f"{label:<20s}  Missing 'logits' single-position extract in TT intermediates")
                pcc_results.append((label, -1.0))
                continue
            last_token_idx = number_of_non_padded_tokens - 1 if padding_side == "right" else ref_host.shape[-2] - 1
            try:
                ref_slice = ref_host.narrow(-2, last_token_idx, 1)
                _, pcc = comp_pcc(ref_slice.float(), tt_host.float())
                logger.debug(f"{label:<20s}  PCC = {pcc:.6f}")
                pcc_results.append((label, pcc))
            except Exception as e:
                logger.error(f"{label:<20s}  PCC comparison failed: {e}")
                pcc_results.append((label, -1.0))
            continue

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


def run_model(
    variant,
    config,
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

    if use_pretrained and not variant.supports_pretrained:
        pytest.skip(f"{variant.name}: pretrained weights not wired")

    profiler.clear()
    profiler.start("total_test_time")
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

    padding_side = tokenizer.padding_side

    # --- Cache-aware loading strategy ---
    profiler.start("cache_check")

    # Check cache states
    experts_per_chip = n_routed_experts // (mesh_shape[0] * mesh_shape[1]) if use_pretrained else 8
    ttnn_cache_complete = (
        TtPrefillTransformer.check_cache_complete(
            effective_cache_path,
            num_layers,
            experts_per_chip,
            first_k_dense=variant.model_config.NUM_DENSE_LAYERS,
        )
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
    ref_cache_exists = check_reference_cache_exists(variant, cache_key) if (pcc_validation and trace is None) else False

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
        elif input_source == "prompt_25k":
            from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

            prompt_text = load_prompts_from_json(str(PROMPT_25K_PATH))
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
                variant=variant,
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
                save_reference_cache(variant, cache_key, ref_snapshots, ref_kvpe_list)
                logger.info("Reference cached")
        else:
            # Both caches exist - skip loading entirely
            logger.info("Both caches exist, skipping weight loading")
            state_dict = {}
    else:
        # Random weights - always create HF model
        logger.info("Creating HF model with random weights...")
        hf_model = create_hf_model(variant, config, num_layers, n_routed_experts=n_routed_experts)
        state_dict = extract_tt_state_dict(variant, hf_model)
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
        model_cfg=variant.model_config,
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
    lookup_table = create_kv_chunk_address_table_ds(
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

    # --- Determinism check (isolated from the pcc_validation path below) ---
    # Run num_iterations forwards on identical input and compare every iteration's per-stage
    # intermediates + final logits + sampled token against the iter-0 baseline.
    if determinism_check:
        if pcc_validation:
            pytest.skip("determinism_check and pcc_validation are mutually exclusive — pick one")
        if num_iterations < 2:
            pytest.skip("determinism_check requires num_iterations >= 2 (iter 0 is the baseline)")
        threshold = DETERMINISM_PCC_THRESHOLD
        logger.info(f"Determinism check (threshold={threshold}, baseline=iter0)")
        profiler.start("tt_forward")
        baseline_items = baseline_logits = baseline_first_token_id = None
        det_failures = []
        for i in range(num_iterations):
            logger.info(f"Determinism iteration: {i}")
            # Seed the host sampler so identical (bit-exact) logits sample the same token
            # -> first_token_id reflects only device determinism.
            torch.manual_seed(0)
            first_token_id, _, tt_intermediates = transformer(
                tt_tokens,
                tt_kvpe_cache,
                number_of_non_padded_tokens=number_of_non_padded_tokens,
                return_intermediates=True,
                read_profiler=False,
                temperature=temperature,
            )
            ttnn.synchronize_device(mesh_device)
            if i == 0:
                # lm_head is a fixed 32-row tile (not the full sequence) -> exclude it from the
                # per-stage slicer; the "logits" comparison below covers the LM-head output.
                excluded = {"first_token", "logits", "lm_head"}
                baseline_items = [
                    (k, v.clone().detach())
                    for k, v in tt_intermediates.items()
                    if isinstance(v, torch.Tensor) and k not in excluded
                ]
                _bl = tt_intermediates.get("logits")
                baseline_logits = _bl.clone().detach() if isinstance(_bl, torch.Tensor) else None
                baseline_first_token_id = first_token_id
                logger.info(f"Determinism: captured iter0 baseline ({len(baseline_items)} tensors)")
                continue
            iter_pcc = _compare_intermediate_pcc(
                baseline_items, tt_intermediates, number_of_non_padded_tokens, padding_side
            )
            if baseline_logits is not None and isinstance(tt_intermediates.get("logits"), torch.Tensor):
                try:
                    _, lp = comp_pcc(baseline_logits.float(), tt_intermediates["logits"].float())
                    iter_pcc.append(("logits", lp))
                except Exception as e:
                    logger.error(f"logits PCC comparison failed: {e}")
                    iter_pcc.append(("logits", -1.0))
            iter_pcc.append(("first_token_id", 1.0 if first_token_id == baseline_first_token_id else -1.0))
            logger.info(f"\n--- Determinism iter {i} vs iter0 ---")
            for label, pcc in iter_pcc:
                status = "PASS" if pcc >= threshold else ("FAIL" if pcc >= 0 else "ERROR")
                logger.info(f"{label:<20s}  {pcc:>10.6f}  {status:>8s}")
                if pcc < threshold:
                    det_failures.append((i, label, pcc))
        profiler.end("tt_forward")
        profiler.end("total_test_time")
        if det_failures:
            msg = "; ".join(f"iter {it} {label}: {pcc:.6f}" for it, label, pcc in det_failures)
            pytest.fail(f"Determinism PCC below {threshold}: {msg}")
        logger.success(
            f"TtPrefillTransformer determinism test passed across {num_iterations} iteration(s) "
            f"(num_layers={num_layers}, n_routed_experts={n_routed_experts}, gate_fallback_mode={gate_fallback_mode})"
        )
        return

    # --- Forward ---
    profiler.start("tt_forward")
    logger.info("Running TtPrefillTransformer forward...")
    do_return_kv = pcc_validation and return_kv_cache
    for i in range(num_iterations):
        start = time.time()
        logger.info(f"Starting iteration: {i}")
        first_token_id, first_token_prob, tt_intermediates = transformer(
            tt_tokens,
            tt_kvpe_cache,
            number_of_non_padded_tokens=number_of_non_padded_tokens,
            return_intermediates=pcc_validation,
            read_profiler=False,
            temperature=temperature,
        )
        logger.info(f"Starting completion sync on iteration: {i}")
        ttnn.synchronize_device(mesh_device)
        end = time.time()
        logger.info(f"Iteration {i} completed in {end - start} seconds.")
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

    logger.info(
        f"Params: pcc_validation={pcc_validation}, return_kv_cache={return_kv_cache}, do_return_kv={do_return_kv} is_balanced={is_balanced} ref_kvpe_list={ref_kvpe_list is not None}"
    )

    # --- PCC check ---
    if pcc_validation:
        profiler.start("pcc_validation")

        # --- Determine threshold based on reference source ---
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

        # --- Load reference snapshots (priority: trace > cache > already computed) ---
        pcc_results = []
        if trace is not None:
            reference_items = trace.ref_snapshots.items()
        else:
            if ref_snapshots is None:
                logger.info("Loading reference from cache...")
                ref_snapshots, ref_kvpe_list = load_reference_cache(variant, cache_key)

            ref_labels = ["embed"] + [f"layer_{i}" for i in range(num_layers)] + ["norm", "lm_head"]
            reference_items = zip(ref_labels, ref_snapshots)

        pcc_results.extend(
            _compare_intermediate_pcc(
                reference_items,
                tt_intermediates,
                number_of_non_padded_tokens,
                padding_side,
            )
        )

        # Per-layer KVPE PCC comparison — read back from external cache
        if do_return_kv and ref_kvpe_list is not None:
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

            # Per-layer chunk readback via the address table — verify the table
            # maps to the same bytes as the gathered cache, chunk by chunk.
            # Only meaningful when `is_balanced` so `tt_kvpe_all_layers` is
            # position-continuous (matching the lookup table's position index).
            if is_balanced:
                logger.info(f"Starting KV cache table validity check")
                chunk_shape = [1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, kvpe_cache_head_dim]
                for layer in range(num_layers):
                    for position in range(0, isl_total, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                        raw_bytes = lookup_table.read_device_chunk(layer=layer, position=position, slot=0)
                        chunk_tt = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes(raw_bytes, chunk_shape)
                        chunk_torch = ttnn.to_torch(chunk_tt).to(torch.bfloat16)
                        expected_chunk = tt_kvpe_all_layers[
                            layer : layer + 1, :, position : position + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, :
                        ]
                        assert_equal(chunk_torch, expected_chunk)
                logger.info(f"KV cache table validity check passed!")

        # --- Logits PCC check (last-token logits vs trace reference) ---
        # Trace logits / next-token are products of the full traced model. They are
        # only meaningful when the TT model ran the same number of layers as the trace.
        trace_full_model = trace is not None and num_layers == trace.metadata.get("n_layers")
        if trace_full_model and trace.logits is not None and "logits" in tt_intermediates:
            try:
                _, logits_pcc = comp_pcc(trace.logits.float(), tt_intermediates["logits"].float())
                logger.info(f"{'logits':<20s}  PCC = {logits_pcc:.6f}")
                pcc_results.append(("logits", logits_pcc))
            except Exception as e:
                logger.error(f"{'logits':<20s}  PCC comparison failed: {e}")
                pcc_results.append(("logits", -1.0))
        elif trace is not None and not trace_full_model:
            logger.info(
                f"Skipping trace logits/first-token checks: "
                f"num_layers={num_layers} != trace n_layers={trace.metadata.get('n_layers')}"
            )

        profiler.end("pcc_validation")

        # --- Summary table ---
        logger.info(f"\n{'='*50}")
        logger.info(f"{'Stage':<20s}  {'PCC':>10s}  {'Status':>8s}")
        logger.info(f"{'-'*50}")
        failures = []
        for label, pcc in pcc_results:
            status = "PASS" if pcc >= threshold else ("FAIL" if pcc >= 0 else "ERROR")
            logger.info(f"{label:<20s}  {pcc:>10.6f}  {status:>8s}")
            if pcc < threshold:
                failures.append((label, pcc))
        logger.info(f"{'='*50}")

        # --- First token info ---
        tok = tokenizer
        token_text = tok.decode([first_token_id]) if tok else "N/A"
        first_temp = temperature[0] if isinstance(temperature, list) else temperature
        logger.info(
            f"First Token: ID={first_token_id} [{repr(token_text)}] prob={first_token_prob*100:.1f}% temp={first_temp}"
        )

        # First-token cross-check against the reference
        if trace is not None and num_layers == trace.metadata.get("n_layers"):
            token_match = check_first_token_match(trace, trace_dir, first_token_id, first_token_prob)
            if token_match is False:
                failures.append(("first_token_match", -1.0))
        elif trace is None and num_layers == config.num_hidden_layers:
            hf_match = check_first_token_match_host_ref(
                ref_snapshots, number_of_non_padded_tokens, padding_side, first_token_id, tok
            )
            if hf_match is False:
                failures.append(("first_token_match", -1.0))
        else:
            logger.debug("Skipping first token check")

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

        if not has_pcc_failures:
            logger.success(
                f"TtPrefillTransformer PCC test passed "
                f"(num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
                f"gate_fallback_mode={gate_fallback_mode}, "
                f"weights={weight_type}, ref_source={'trace' if trace else 'host'})"
            )
        else:
            pcc_failure_msg = "; ".join(f"{label}: {pcc:.6f}" for label, pcc in failures)
            logger.error(
                f"TtPrefillTransformer PCC test has failures " f"(num_layers={num_layers}, failures={len(failures)})"
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
    if pcc_validation and has_pcc_failures:
        pytest.fail(f"PCC below {threshold} at: {pcc_failure_msg}")


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
        "prompt_25k",
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
# iter2000 is the long-running stability soak (program-cache growth, semaphore
# desync, leaks). Kept opt-in via -k iter2000; CI selectors normally pick iter1.
@pytest.mark.parametrize("determinism_check", [False, True], ids=["no_determinism", "with_determinism"])
@pytest.mark.parametrize("num_iterations", [1, 2, 5, 25, 2000], ids=["iter1", "iter2", "iter5", "iter25", "iter2000"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
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
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
        # FABRIC_2D variants — shared list defined in conftest.py (also used by
        # test_prefill_block_loop.py). Covers (4,2) BH LoudBox, (2,4) asymmetric, (8,4) BH Galaxy.
        *FABRIC_2D_PREFILL_BLOCK_MESH_PARAMS,
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.timeout(0)
def test_ds_prefill_transformer(
    variant,
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
    run_model(
        variant,
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
    )


# ════════════════════════════════════════════════════════════════════════════════════════════════
# GLM-5.1 chained per-layer PCC vs the GPU trace (single-shot, 2x4 sp2×tp4 Blackhole)
# ════════════════════════════════════════════════════════════════════════════════════════════════
# Builds the WHOLE TtPrefillTransformer (embed → [block × N] → norm → lm_head) with REAL GLM-5.1
# weights and runs ONE single-shot forward at isl=5120 on the (2,4) mesh, snapshotting every layer's
# residual-stream output (return_intermediates) and PCC-ing it against the GPU trace's
# decoder_output_layer_{i}. Because the device runs its OWN embedding + chained blocks, intermediates
# ["layer_i"] IS the chained comparison; only the TARGET comes from the trace. This pins WHERE the
# chained PCC drops (~0.88 mid-stack) and ISOLATES the device MoE gate as the cause by sweeping
# gate_fallback_mode ∈ {HOST_ALL (fp32 host gate), DEVICE_FP32 (on-device gate kernel)} and dumping
# the device top-8 routing to compare against the trace's expert_ids.
#
# WEIGHTS (streaming, host-RAM-bounded). A single MoE layer's 256 experts are ~30 GB on host, so all
# 78 layers cannot be materialized at once. Pass 1 streams each layer's real fp8→bf16 weights and
# writes the TTNN .tensorbin cache via the no-device-copy `build_ttnn_cache` helpers, freeing host
# after each layer; Pass 2 builds the transformer straight from that cache. On a cache hit ttMLA keys
# `_has_indexer` off the state_dict (the cache holds NO indexer stems), so Pass 2 injects the 5 tiny
# `indexer.*` stems per sparse layer — else the cached model silently runs DENSE MLA (no DSA).
#
# Run (THIS machine, 2x4 = sp2×tp4 Blackhole; device is exclusive → run serially):
#   TT_METAL_HOME=/localdev/nmilicevic/tt-metal PYTHONPATH=/localdev/nmilicevic/tt-metal \
#   GLM51_REPO=zai-org/GLM-5.1-FP8 \
#   TT_DS_PREFILL_TTNN_CACHE=/localdev/nmilicevic/glm_tp4_cache/ttnn \
#   TT_DS_PREFILL_HOST_REF_CACHE=/localdev/nmilicevic/glm_tp4_cache/host_ref \
#   /localdev/nmilicevic/tt-metal/python_env/bin/python -m pytest -svq \
#   models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py \
#   -k "glm_chained_vs_trace and L6 and host_all"   # then ...and L6 and device_fp32, then L78
# ════════════════════════════════════════════════════════════════════════════════════════════════

# Trace bundle (decoder_io/, routing/, metadata.json). Override with GLM51_PREFILL_TRACE_DIR.
GLM_TRACE_DIR = "/localdev/nmilicevic/tt-metal/bit_sculpt/results/glm-51-traces/vllm-glm51-sdpa-5k-trace"
# Per-layer PCC is RECORD-ONLY while we investigate the drop — never hard-fail deep layers; only the
# functional sanity gate (first-token match) and the decisive routing-collapse summary are asserted.
GLM_FIRST_TOKEN = 19264  # metadata.completion_token_ids[0]


def _glm_repo() -> str:
    return os.environ.get("GLM51_REPO", "zai-org/GLM-5.1-FP8")


def _glm_trace_dir():
    from pathlib import Path

    return Path(os.environ.get("GLM51_PREFILL_TRACE_DIR", GLM_TRACE_DIR))


def _glm_load_token_ids(trace_dir, isl: int) -> torch.Tensor:
    """token_ids[:isl] from the trace metadata — the EXACT sequence the trace was generated from
    (PCC compares two runs of the same input)."""
    with open(trace_dir / "metadata.json") as f:
        md = json.load(f)
    ids = torch.tensor(md["token_ids"][:isl], dtype=torch.int64)
    assert ids.shape[0] == isl, f"trace has {ids.shape[0]} tokens, need {isl}"
    return ids


def _glm_ref_layer(trace_dir, layer: int) -> torch.Tensor:
    """decoder_output_layer_{layer} [5120, 6144] (bf16→fp32) — concat the row shards (chunked_group_a_v1)."""
    from safetensors import safe_open

    tdir = trace_dir / "decoder_io" / f"decoder_output_layer_{layer}"
    key = f"decoder_output_layer_{layer}"
    parts = []
    for shard in sorted(tdir.glob("rows_*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            parts.append(f.get_slice(key)[:].to(torch.float32))
    assert parts, f"no row shards in {tdir}"
    return torch.cat(parts, dim=0)


def _glm_ref_expert_ids(trace_dir, layer: int):
    """routing/expert_ids_layer_{layer} [5120, 8] int32 (GPU top-8 per token), or None if absent
    (layers < first_k_dense are dense → no routing)."""
    from safetensors import safe_open

    tdir = trace_dir / "routing" / f"expert_ids_layer_{layer}"
    if not tdir.exists():
        return None
    key = f"expert_ids_layer_{layer}"
    parts = []
    for shard in sorted(tdir.glob("rows_*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            parts.append(f.get_slice(key)[:].to(torch.int64))
    return torch.cat(parts, dim=0) if parts else None


def _glm_mla_weights(layer: int) -> dict:
    """GLM layer-`layer` MLA (+ 5 indexer stems) in the v3 `mla_weights` dict shape, via the cheap
    attention-only MLACPU/WEIGHT_NAME_MAP path (loads only the small self_attn shard — seconds)."""
    from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import glm_model_args
    from models.demos.deepseek_v32.reference_cpu.model import MLACPU
    from models.demos.deepseek_v32.reference_cpu.weights import initialize_weights
    from models.demos.deepseek_v32.tests.test_mla import WEIGHT_NAME_MAP

    mla = MLACPU(glm_model_args(max_seq=SEQ_LEN_5K), simulate_fp8=False).eval()
    mla.indexer.use_fp8_path = False
    initialize_weights(mla, layer=layer, repo=_glm_repo(), local_files_only=True)
    sd = mla.state_dict()
    return {v3: sd[cpu].clone() for cpu, v3 in WEIGHT_NAME_MAP.items()}


def _glm_indexer_stems(layer: int) -> dict:
    """Just the 5 `indexer.*` stems, nested as `mla_weights` — the per-sparse-layer injection that
    keeps ttMLA's `_has_indexer` True on a cache hit (else the cached model runs DENSE MLA)."""
    mw = _glm_mla_weights(layer)
    return {"mla_weights": {k: v for k, v in mw.items() if k.startswith("indexer.")}}


def _glm_block_state(layer: int, first_k_dense: int) -> dict:
    """Full TtPrefillBlock state_dict for a GLM layer: MLA(+indexer) + decoder norms + FFN/MoE."""
    from models.demos.deepseek_v32.reference_cpu.weights import load_dense_block_weights, load_moe_block_weights

    state = {"mla_weights": _glm_mla_weights(layer)}
    repo = _glm_repo()
    if layer < first_k_dense:
        state.update(load_dense_block_weights(layer, repo=repo, local_files_only=True))
    else:
        state.update(load_moe_block_weights(layer, repo=repo, local_files_only=True))
    return state


def _glm_embed_norm_lmhead():
    """model.embed_tokens.weight / model.norm.weight / lm_head.weight (bf16) from the GLM HF repo."""
    import json as _json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    repo = _glm_repo()
    idx = hf_hub_download(repo, "model.safetensors.index.json", local_files_only=True)
    wm = _json.load(open(idx))["weight_map"]
    out = {}
    for key, name in [
        ("model.embed_tokens.weight", "embed_weight"),
        ("model.norm.weight", "norm_weight"),
        ("lm_head.weight", "lm_head_weight"),
    ]:
        path = hf_hub_download(repo, wm[key], local_files_only=True)
        with safe_open(path, framework="pt", device="cpu") as f:
            out[name] = f.get_tensor(key).to(torch.float32 if name == "embed_weight" else torch.bfloat16)
    return out


# Precision levers (env-gated) for matching the GPU's fp32-accum / bf16-weight profile. See
# _glm_routed_expert_dtype + the lever notes in test_glm_chained_vs_trace. GLM_MLA_FP32_ACC and
# GLM_MLA_WUV_BF16 are read inside the model (mla.py); GLM_MOE_EXPERTS_BF8 is wired here because the
# routed-expert weight dtype is a build_ttnn_cache / TtPrefillTransformer argument, not a model env.
def _glm_routed_expert_dtype():
    """bf8 routed-expert weights when GLM_MOE_EXPERTS_BF8 is set (GPU uses fp8), else the bf4 default."""
    import ttnn as _ttnn

    return _ttnn.bfloat8_b if os.environ.get("GLM_MOE_EXPERTS_BF8") else _ttnn.bfloat4_b


def _glm_recache_mla_wuv_bf16(mesh_device, config, cache_dir, num_layers, seq_len, sp_axis, tp_axis):
    """Lever #2 helper: ensure each layer's bf16 W_UV (wkv_b2) cache exists. The wildcard block-level
    completeness check treats the pre-existing bf8 wkv_b2 as 'complete', so a normal build pass would
    skip it. Here we rebuild ONLY the MLA cache per layer (cheap: _glm_mla_weights loads just the small
    self_attn shard). ttMLA.build_ttnn_cache writes the new bf16 wkv_b2 (env GLM_MLA_WUV_BF16=1) while
    the other MLA weights hit their existing dtype-specific cache files and are merely re-loaded."""
    from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA

    for i in range(num_layers):
        prefix = f"layer_{i}.mla"
        bf16_wuv = list(cache_dir.glob(f"{prefix}.wkv_b2_dtype_BFLOAT16_layout_TILE.tensorbin"))
        if bf16_wuv:
            continue  # bf16 W_UV already cached for this layer
        logger.info(f"[glm cache] layer {i}: MLA recache for bf16 W_UV (wkv_b2)")
        mla_state = _glm_mla_weights(i)  # cheap self_attn-only shard
        ttMLA.build_ttnn_cache(
            state_dict=mla_state,
            cache_path=cache_dir,
            mesh_device=mesh_device,
            config=config,
            layer_idx=i,
            seq_len=seq_len,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
        )
        del mla_state
        gc.collect()


def _glm_build_cache(
    mesh_device,
    config,
    cache_dir,
    num_layers,
    seq_len,
    sp_axis,
    tp_axis,
    gate_fallback_mode,
    routed_expert_weights_dtype=None,
):
    """Pass 1: stream each layer's real weights → write the TTNN cache (no device copy), freeing host
    per layer. Skips layers already cached. embed/norm/lm_head cached once at the end."""
    import ttnn as _ttnn
    from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
    from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
    from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
    from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

    if routed_expert_weights_dtype is None:
        routed_expert_weights_dtype = _ttnn.bfloat4_b
    first_k_dense = GLM51Config.NUM_DENSE_LAYERS
    experts_per_chip = GLM51Config.NUM_ROUTED_EXPERTS // mesh_device.get_num_devices()
    init_checker(cache_dir)

    # Lever #3 (GLM_MOE_EXPERTS_BF8): the block-completeness check matches routed-expert caches with a
    # `*` wildcard, so it treats the pre-existing bf4 experts as complete and would skip the bf8 rebuild
    # — but the live load then can't find the bf8 files and would rebuild from empty placeholder weights.
    # Detect the dtype-specific routed-expert cache directly and force a (real-weight) rebuild when the
    # requested dtype's files are missing. The other components hit their existing dtype-keyed caches and
    # are merely re-loaded, so only the routed-expert tensors are genuinely rebuilt from streamed weights.
    dtype_tag = str(routed_expert_weights_dtype).rsplit(".", 1)[-1].upper()  # e.g. BFLOAT8_B / BFLOAT4_B

    def _routed_expert_dtype_cached(layer_idx: int) -> bool:
        # The first local expert's gate file is representative of the whole routed-expert set.
        return bool(
            list(
                cache_dir.glob(f"layer_{layer_idx}.routed_expert.local_0_gate_dtype_{dtype_tag}_layout_TILE.tensorbin")
            )
        )

    for i in range(num_layers):
        is_dense = i < first_k_dense
        block_complete = TtPrefillBlock.check_cache_complete(cache_dir, i, is_dense, experts_per_chip)
        moe_dtype_ok = is_dense or _routed_expert_dtype_cached(i)
        if block_complete and moe_dtype_ok:
            logger.info(f"[glm cache] layer {i}: HIT, skip build")
            continue
        reason = "MISS" if not block_complete else f"routed-expert dtype {dtype_tag} missing"
        logger.info(f"[glm cache] layer {i}: {reason} → stream real weights + write cache")
        state = _glm_block_state(i, first_k_dense)
        TtPrefillBlock.build_ttnn_cache(
            state_dict=state,
            layer_idx=i,
            cache_path=cache_dir,
            mesh_device=mesh_device,
            config=config,
            model_cfg=GLM51Config,
            seq_len=seq_len,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            gate_fallback_mode=gate_fallback_mode,
            routed_expert_weights_dtype=routed_expert_weights_dtype,
        )
        del state
        gc.collect()

    # Lever #2 (GLM_MLA_WUV_BF16): the wildcard block check above would treat a pre-existing bf8 wkv_b2
    # as complete and skip it, so add a targeted MLA-only recache that writes the bf16 W_UV per layer.
    if os.environ.get("GLM_MLA_WUV_BF16"):
        _glm_recache_mla_wuv_bf16(mesh_device, config, cache_dir, num_layers, seq_len, sp_axis, tp_axis)

    # embed / norm / lm_head (model-global, not layer-indexed)
    if not (
        TtParallelEmbedding.check_cache_complete(cache_dir)
        and TtDistributedRmsNorm.check_cache_complete(cache_dir, "norm")
        and TtLMHead.check_cache_complete(cache_dir)
    ):
        logger.info("[glm cache] building embed/norm/lm_head cache")
        enl = _glm_embed_norm_lmhead()
        TtParallelEmbedding.build_ttnn_cache(
            torch_weight=enl["embed_weight"],
            vocab_size=config.vocab_size,
            emb_dim=config.hidden_size,
            mesh_device=mesh_device,
            cache_path=cache_dir,
            tp_axis=tp_axis,
        )
        TtDistributedRmsNorm.build_ttnn_cache(
            torch_weight=enl["norm_weight"],
            emb_dim=config.hidden_size,
            mesh_device=mesh_device,
            cache_path=cache_dir,
            cache_name_prefix="norm",
        )
        TtLMHead.build_ttnn_cache(
            torch_weight=enl["lm_head_weight"],
            vocab_size=config.vocab_size,
            emb_dim=config.hidden_size,
            mesh_device=mesh_device,
            cache_path=cache_dir,
            is_column_parallel=True,
        )
        del enl
        gc.collect()


def _glm_routing_overlap(gate_dir, trace_dir, layers, num_experts):
    """Per-layer mean top-8 set-overlap (|ours ∩ gpu|/8) + unique-experts-used (ours vs gpu) for the
    dumped device routing vs the trace. gate_indices_layer_{L}.pt is [seq, 8]; expert_ids is [seq, 8]."""
    from pathlib import Path

    rows = []
    for L in layers:
        gpath = Path(gate_dir) / f"gate_indices_layer_{L}.pt"
        ref = _glm_ref_expert_ids(trace_dir, L)
        if not gpath.exists() or ref is None:
            continue
        ours = torch.load(gpath).to(torch.int64)  # [seq, 8]
        n = min(ours.shape[0], ref.shape[0])
        ours, ref = ours[:n], ref[:n]
        inter = torch.tensor([len(set(ours[r].tolist()) & set(ref[r].tolist())) for r in range(n)], dtype=torch.float32)
        rows.append(
            {
                "layer": L,
                "overlap": (inter / 8.0).mean().item(),
                "uniq_ours": int(torch.unique(ours).numel()),
                "uniq_gpu": int(torch.unique(ref).numel()),
            }
        )
    return rows


def _run_glm_chained_vs_trace(mesh_device, num_layers, gate_fallback_mode, num_links, topology):
    from pathlib import Path

    from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config, glm_hf_config

    torch.manual_seed(42)
    trace_dir = _glm_trace_dir()
    if not (trace_dir / "metadata.json").exists():
        pytest.skip(f"GLM trace not found: {trace_dir} (set GLM51_PREFILL_TRACE_DIR)")

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp_factor, tp_factor = mesh_shape[sp_axis], mesh_shape[tp_axis]
    isl = SEQ_LEN_5K
    isl_per_chip = isl // sp_factor
    assert isl % (sp_factor * tp_factor) == 0 and isl_per_chip % 32 == 0, f"isl {isl} not divisible on {mesh_shape}"
    assert GLM51Config.INDEX_N_HEADS % tp_factor == 0, "index heads must split over tp"

    config = glm_hf_config(max_seq=isl)
    config.vocab_size = GLM51Config.VOCAB_SIZE
    config.num_hidden_layers = GLM51Config.NUM_LAYERS
    first_k_dense = GLM51Config.NUM_DENSE_LAYERS

    # --- TTNN cache: $TT_DS_PREFILL_TTNN_CACHE/glm_5_1_bh_<ndev>dev, then a sp×tp subdir. ---
    cache_root = os.environ.get("TT_DS_PREFILL_TTNN_CACHE")
    assert cache_root, "set TT_DS_PREFILL_TTNN_CACHE to a writable dir"
    cache_dir = Path(cache_root) / f"glm_5_1_bh_{ttnn.get_num_devices()}dev" / f"{sp_factor}x{tp_factor}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    token_ids = _glm_load_token_ids(trace_dir, isl)

    logger.info(
        f"[glm chained] mesh={mesh_shape} (sp{sp_factor}×tp{tp_factor}) isl={isl} num_layers={num_layers} "
        f"gate={gate_fallback_mode} cache={cache_dir}"
    )

    # Precision lever #3 (GLM_MOE_EXPERTS_BF8): bf8 routed-expert weights to match the GPU's fp8 experts
    # (default is bf4). The dtype is baked into the cache filename, so the bf8 cache is a *separate* set
    # of files from the bf4 one — flipping the env rebuilds only the routed-expert tensors.
    routed_expert_weights_dtype = _glm_routed_expert_dtype()
    logger.info(
        f"[glm chained] precision levers: fp32_acc={bool(os.environ.get('GLM_MLA_FP32_ACC'))} "
        f"wuv_bf16={bool(os.environ.get('GLM_MLA_WUV_BF16'))} "
        f"routed_expert_dtype={routed_expert_weights_dtype}"
    )

    # ---- Pass 1: build the streaming weight cache (free host per layer). ----
    profiler.clear()
    profiler.start("glm_cache_build")
    _glm_build_cache(
        mesh_device,
        config,
        cache_dir,
        num_layers,
        isl,
        sp_axis,
        tp_axis,
        gate_fallback_mode,
        routed_expert_weights_dtype=routed_expert_weights_dtype,
    )
    profiler.end("glm_cache_build")

    # ---- Pass 2: build the transformer FROM the cache + inject the per-sparse-layer indexer stems. ----
    # The cache holds no indexer stems and ttMLA keys `_has_indexer` off the state_dict, so a sparse
    # layer needs {"mla_weights": {indexer stems}} or it silently runs DENSE MLA (the GLM landmine).
    layers_state = [({} if i < first_k_dense else _glm_indexer_stems(i)) for i in range(num_layers)]
    state_dict = {"layers": layers_state}  # embed/norm/lm_head load from cache (torch_weight=None)

    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=GLM51Config,
        state_dict=state_dict,
        num_layers=num_layers,
        seq_len=isl,
        is_balanced=False,
        padding_side="right",
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=gate_fallback_mode,
        routed_expert_weights_dtype=routed_expert_weights_dtype,
        weight_cache_path=cache_dir,
        lm_head_is_column_parallel=True,
    )
    ttnn.synchronize_device(mesh_device)
    del state_dict, layers_state
    gc.collect()
    profiler.end("tt_transformer_creation")

    # External KVPE cache (one slot per layer).
    kvpe_head_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_head_dim,
        mesh_device=mesh_device,
        seq_len=isl,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
    )

    # SP-shard token_ids [1, isl] → [sp, 1, isl_per_chip] (single-shot, non-balanced → natural order).
    tt_tokens = ttnn.from_torch(
        token_ids.reshape(sp_factor, 1, isl_per_chip),
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
    )

    # ---- Forward (single-shot, return per-layer intermediates). The MoE routing dump fires per layer
    # when TT_DS_PREFILL_DUMP_GATE_INDICES is set (tt_moe.py). ----
    profiler.start("tt_forward")
    first_token_id, first_token_prob, intermediates = transformer(
        tt_tokens,
        tt_kvpe_cache,
        number_of_non_padded_tokens=isl,
        return_intermediates=True,
        read_profiler=False,
        temperature=0.0,
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")
    logger.info(f"[glm chained] first token id={first_token_id} prob={first_token_prob:.4f} (trace={GLM_FIRST_TOKEN})")

    # ---- Per-layer chained PCC vs the trace (RECORD-ONLY). ----
    pcc_rows = []
    for i in range(num_layers):
        key = f"layer_{i}"
        if key not in intermediates:
            continue
        ours = intermediates[key]  # [1, isl, emb] bf16 (SP-seq + TP-hidden concatenated, natural order)
        ours = ours.reshape(isl, -1).float()
        ref = _glm_ref_layer(trace_dir, i)  # [isl, emb]
        _, pcc = comp_pcc(ref, ours)
        pcc_rows.append((i, pcc))
        logger.info(f"[glm chained] layer {i:2d} ({'dense' if i < first_k_dense else 'moe '}) PCC={pcc:.5f}")

    logger.info(f"\n{'='*46}\n[glm chained] PER-LAYER PCC  gate={gate_fallback_mode}\n{'-'*46}")
    logger.info(f"{'layer':>5s} {'kind':>5s} {'PCC':>10s}")
    for i, pcc in pcc_rows:
        logger.info(f"{i:>5d} {'dense' if i < first_k_dense else 'moe':>5s} {pcc:>10.5f}")
    logger.info(f"{'='*46}")

    # ---- Routing overlap vs the trace (focus L3 = first MoE layer + a few more). ----
    gate_dir = os.environ.get("TT_DS_PREFILL_DUMP_GATE_INDICES")
    route_rows = []
    if gate_dir:
        # L3 (first MoE) is the decisive device-gate probe; sample a spread + always the deepest layer.
        focus = sorted(
            {L for L in (3, 4, 5, 8, 12, 20, 30, 40, 50, 62, 77, num_layers - 1) if first_k_dense <= L < num_layers}
        )
        route_rows = _glm_routing_overlap(gate_dir, trace_dir, focus, GLM51Config.NUM_ROUTED_EXPERTS)
        logger.info(f"\n{'='*64}\n[glm chained] ROUTING vs trace  gate={gate_fallback_mode}\n{'-'*64}")
        logger.info(f"{'layer':>5s} {'overlap':>9s} {'uniq_ours':>10s} {'uniq_gpu':>9s}")
        for r in route_rows:
            logger.info(f"{r['layer']:>5d} {r['overlap']:>9.4f} {r['uniq_ours']:>10d} {r['uniq_gpu']:>9d}")
        logger.info(f"{'='*64}")
    else:
        logger.warning("[glm chained] TT_DS_PREFILL_DUMP_GATE_INDICES not set → no routing comparison")

    # ---- Timing ----
    for k in profiler.times:
        logger.info(f"  {k}: {profiler.get(k) * 1000:.2f} ms")

    return {
        "first_token_id": first_token_id,
        "pcc_rows": pcc_rows,
        "route_rows": route_rows,
        "first_token_match": (first_token_id == GLM_FIRST_TOKEN),
    }


@pytest.mark.skipif(not is_blackhole(), reason="GLM requires Blackhole")
@pytest.mark.parametrize(
    "num_layers",
    [6, 40, 78],
    ids=["L6", "L40", "L78"],
)
@pytest.mark.parametrize(
    "gate_fallback_mode",
    [GateComputeMode.HOST_ALL, GateComputeMode.DEVICE_FP32],
    ids=["host_all", "device_fp32"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=GLM51Config.FABRIC_PAYLOAD_SIZE),
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
def test_glm_chained_vs_trace(mesh_device, device_params, num_layers, gate_fallback_mode, num_links, topology):
    """GLM-5.1 whole-transformer chained per-layer PCC vs the GPU trace on 2x4 (sp2×tp4) Blackhole.

    Record-only per-layer PCC (investigating the ~0.88 mid-stack drop); asserts only the routing
    verdict: DEVICE_FP32 must NOT collapse L3 routing worse than HOST_ALL (the device-gate-bug probe).
    """
    res = _run_glm_chained_vs_trace(mesh_device, num_layers, gate_fallback_mode, num_links, topology)

    # Decisive probe: at L3 (first MoE layer), DEVICE_FP32 routing should not be dramatically worse than
    # HOST_ALL. Record-only across modes (the cross-mode verdict is read from the two runs' logs), but
    # we DO assert HOST_ALL routes sanely so a real regression in the host gate trips CI.
    l3 = next((r for r in res["route_rows"] if r["layer"] == 3), None)
    if gate_fallback_mode == GateComputeMode.HOST_ALL and l3 is not None:
        assert l3["overlap"] >= 0.5 and l3["uniq_ours"] >= 128, (
            f"HOST_ALL L3 routing looks collapsed (overlap={l3['overlap']:.3f}, "
            f"uniq_ours={l3['uniq_ours']}) — host fp32 gate regressed"
        )
    logger.success(
        f"[glm chained] done gate={gate_fallback_mode} num_layers={num_layers} "
        f"first_token={'MATCH' if res['first_token_match'] else 'MISS'}"
    )


@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
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
        "prompt_25k",
        "random",
        "passkey",
        "kv_retrieval",
        "longdialogue_qa_eng",
        "longbook_qa_eng",
    ],
)
@pytest.mark.parametrize("pcc_validation", [True, False], ids=["pcc", "smoke"])
@pytest.mark.parametrize("is_balanced", [False], ids=["non_balanced"])
@pytest.mark.parametrize(
    "isl_total, dispatch_buffer_capacity_factor",
    [(SEQ_LEN_1K, 8), (SEQ_LEN_5K, 8), (SEQ_LEN_25K, 8)],
    ids=["1k", "5k", "25k"],
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
    [(384, GateComputeMode.DEVICE)],
    ids=["e384_device"],
)
@pytest.mark.parametrize("determinism_check", [False, True], ids=["no_determinism", "with_determinism"])
@pytest.mark.parametrize("num_iterations", [1, 2, 5, 25, 2000], ids=["iter1", "iter2", "iter5", "iter25", "iter2000"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi"])
@pytest.mark.timeout(0)
def test_kimi_prefill_transformer(
    variant,
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
    run_model(
        variant,
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
    )

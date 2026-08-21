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
- (input_source, pcc_validation, use_pretrained): one coupled axis. A golden is a reference only
  for the pretrained weights it was captured from, so PCC runs pair a single source per variant
  with pretrained weights; everything else is smoke-only.
- n_routed_experts / gate_fallback_mode: MoE configurations
"""

import gc
import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from conftest import is_galaxy
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.mistral_small_4 import mistral4_decoder_layer_reference, rms_norm
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_119b_config import Mistral4Small119BConfig
from models.demos.deepseek_v3_d_p.tests.conftest import FABRIC_2D_PREFILL_BLOCK_MESH_PARAMS
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers, resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    create_balanced_chunk_order,
    reorder_tensor_chunks,
    reverse_reorder_tensor_chunks,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.pcc_plot_utils import generate_pcc_plots, write_pcc_summary
from models.demos.deepseek_v3_d_p.utils.test_utils import save_intermediate_output
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    PROMPT_1K_PATH,
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
    mla_kvpe_width,
    save_reference_cache,
    slice_debug_trace,
    slice_non_padded,
    tokenize_prompt_to_isl,
)
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD = 0.99
TRACE_PCC_THRESHOLD = 0.97
TRACE_PCC_THRESHOLD_HOST = 0.96
TRACE_PCC_THRESHOLD_DEVICE_BF16 = 0.88
TRACE_PCC_THRESHOLD_DEVICE_FP32 = 0.95
# Determinism: every iteration is expected to match the iter-0 baseline near-bit-exactly.
DETERMINISM_PCC_THRESHOLD = 1.0

# Only the subset that is still a parametrized input_source; downloaded on first use.
INFINITEBENCH_SUBSET_NAMES = {"longbook_qa_eng"}
# input_source meaning "this variant's own golden" — naming it after a prompt would go stale.
VARIANT_DEFAULT_TRACE = "variant_default"
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

    # Priority 1: debug trace on disk. A golden is captured from the full pretrained model, so it is
    # a reference only for a run with those weights and that expert count.
    trace = None
    trace_dir = None
    trace_sliced = False
    trace_eligible = pcc_validation and use_pretrained and n_routed_experts == variant.model_config.NUM_ROUTED_EXPERTS
    trace_match = find_trace_dir(input_source, isl_total, padding_side) if trace_eligible else None
    if trace_match is not None:
        trace_dir, trace_isl = trace_match
        trace = load_debug_trace(trace_dir, num_layers=num_layers)
        if trace_isl > isl_total:
            trace = slice_debug_trace(trace, isl_total)
            trace_sliced = True
            logger.info(f"Sliced trace {trace_dir.name} from native isl={trace_isl} to requested isl={isl_total}")
        logger.info(
            f"Loaded debug trace from {trace_dir} "
            f"(trace n_layers={trace.metadata.get('n_layers')}, test num_layers={num_layers}, "
            f"native_isl={trace_isl}, sliced={trace_sliced})"
        )
    # Explicitly asked for this variant's own golden (TRACE_LOOKUP is longbook/R1-only). Not a
    # fallback: only this input_source lands here, so no other row gets handed someone else's golden.
    elif trace_eligible and input_source == VARIANT_DEFAULT_TRACE:
        _pinned = getattr(variant, "test_prefill_trace_default", None)
        assert _pinned and os.path.exists(os.path.join(_pinned, "metadata.json")), (
            f"{variant.name}: input_source={VARIANT_DEFAULT_TRACE} needs a usable "
            f"test_prefill_trace_default, got {_pinned}"
        )
        trace_dir = Path(_pinned)
        trace = load_debug_trace(trace_dir, num_layers=num_layers, isl=isl_total)
        # load_debug_trace(isl=...) chops the per-row tensors, but the stored logits/next_token_id stay
        # full-sequence. Mark a chopped golden sliced so the later full-model checks are skipped instead
        # of comparing this shorter prefill against the golden's final-token logits.
        native_isl = len(trace.metadata.get("token_ids", []))
        if native_isl > isl_total:
            trace_sliced = True
        logger.info(
            f"Loaded {variant.name} variant golden from {trace_dir} "
            f"(num_layers={num_layers}, isl={isl_total}, native_isl={native_isl}, sliced={trace_sliced})"
        )

    cache_key = ReferenceCacheKey(
        weight_type=weight_type,
        input_source=input_source,
        isl_total=isl_total,
        num_layers=num_layers,
        n_routed_experts=n_routed_experts,
        padding_side=padding_side,
    )
    # A cache written before the compressed-line fix holds expanded per-head keys, and without the
    # width check still reports as reusable.
    kvpe_width = mla_kvpe_width(config)
    ref_cache_exists = pcc_validation and trace is None and check_reference_cache_exists(variant, cache_key, kvpe_width)

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

            # The file holds two prompts; one prefill takes one.
            prompt_text = load_prompts_from_json(str(PROMPT_1K_PATH), max_prompts=1)[0]
        elif input_source in INFINITEBENCH_SUBSET_NAMES:
            cached_path = download_infinitebench_subset(input_source)
            with open(cached_path) as f:
                prompt_text = json.load(f)["prompt"]
        else:
            raise ValueError(
                f"No tokens for input_source={input_source}: it has no prompt file, and "
                f"variant.test_prefill_trace_default ({getattr(variant, 'test_prefill_trace_default', None)}) "
                f"did not resolve"
            )
        token_ids, attention_mask, _ = tokenize_prompt_to_isl(tok, max_isl=isl_total, prompt_text=prompt_text)
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
    has_indexer = resolve_has_indexer(config)
    cache_format = MlaKvCacheFormat.BF16_RM if has_indexer else MlaKvCacheFormat.BFP8_TILE
    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=isl_total,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
    )

    # Sparse single-shot is folded onto the block-cyclic path, so (like chunked) it needs the caller-owned,
    # user-major layer-stacked indexer key cache [num_users*index_cache_layers, 1, T, D_idx]. Unlike the
    # per-layer KVPE cache, the indexer stride is the COMPACTED full-indexer count (num_full_indexer_layers)
    # for GLM-5.2 cross-layer reuse — "shared" layers reuse a "full" layer's cache and get no slot of their
    # own — falling back to num_layers when there is no indexer_types map. Dense variants use no index cache.
    tt_index_kv_cache = None
    if has_indexer:
        index_cache_layers = num_full_indexer_layers(config) or num_layers
        tt_index_kv_cache = init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=isl_total,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=index_cache_layers,
            num_users=1,
            dtype=ttnn.bfloat8_b,
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
                actual_isl=number_of_non_padded_tokens,
                return_intermediates=True,
                read_profiler=False,
                temperature=temperature,
                index_kv_cache=tt_index_kv_cache,
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
            actual_isl=number_of_non_padded_tokens,
            return_intermediates=pcc_validation,
            read_profiler=False,
            temperature=temperature,
            index_kv_cache=tt_index_kv_cache,
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
                tt_kvpe_cache.storage,
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

        # --- Logits PCC check (last-token logits vs trace reference) ---
        # Trace logits / next-token are products of the full traced model. They are
        # only meaningful when the TT model ran the same number of layers as the trace.
        # A sliced trace's stored logits/next-token belong to the full (longer) sequence,
        # so they are not a valid reference for the shorter prefill — skip those checks.
        trace_full_model = trace is not None and not trace_sliced and num_layers == trace.metadata.get("n_layers")
        if trace_full_model and trace.logits is not None and "logits" in tt_intermediates:
            try:
                _, logits_pcc = comp_pcc(trace.logits.float(), tt_intermediates["logits"].float())
                logger.info(f"{'logits':<20s}  PCC = {logits_pcc:.6f}")
                pcc_results.append(("logits", logits_pcc))
            except Exception as e:
                logger.error(f"{'logits':<20s}  PCC comparison failed: {e}")
                pcc_results.append(("logits", -1.0))
        elif trace is not None and not trace_full_model:
            reason = (
                "trace sliced to a shorter isl (full-sequence logits/next-token invalid)"
                if trace_sliced
                else f"num_layers={num_layers} != trace n_layers={trace.metadata.get('n_layers')}"
            )
            logger.info(f"Skipping trace logits/first-token checks: {reason}")

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
        # (skipped for a sliced trace: its next_token_id is the full sequence's, not the prefix's)
        if trace is not None and not trace_sliced and num_layers == trace.metadata.get("n_layers"):
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
        # PCC plots are opt-in (TT_PREFILL_PCC_PLOTS=1). generate_pcc_plots renders a PNG into trace_dir,
        # which for a pinned golden is a read-only shared mount (/mnt/models/...) -> PermissionError. Off by
        # default so trace-backed runs don't crash on artifact write; still skipped under GitHub Actions.
        if os.getenv("TT_PREFILL_PCC_PLOTS") == "1" and not os.getenv("GITHUB_ACTIONS") and trace_dir is not None:
            generate_pcc_plots(summary_result, output_dir=str(trace_dir))

    # Deferred PCC failure check (after timing report)
    if pcc_validation and has_pcc_failures:
        pytest.fail(f"PCC below {threshold} at: {pcc_failure_msg}")


@pytest.mark.skipif(not is_blackhole(), reason="Requires Blackhole.")
@pytest.mark.parametrize("tokenizer", ["right", "left"], indirect=True, ids=["right_pad", "left_pad"])
@pytest.mark.parametrize("temperature", [[0.5]], ids=["temp_sweep"])
@pytest.mark.parametrize("return_kv_cache", [True], ids=["kv_cache"])
@pytest.mark.parametrize(
    "input_source, pcc_validation, use_pretrained",
    [
        # The golden was captured from the pretrained model
        ("longbook_qa_eng", True, True),
        ("longbook_qa_eng", False, True),
        ("longbook_qa_eng", False, False),
        ("json_prompts", False, True),
        ("json_prompts", False, False),
        ("random", False, True),
        ("random", False, False),
    ],
    ids=[
        "pcc-longbook_qa_eng-pretrained",
        "smoke-longbook_qa_eng-pretrained",
        "smoke-longbook_qa_eng-random",
        "smoke-json_prompts-pretrained",
        "smoke-json_prompts-random",
        "smoke-random-pretrained",
        "smoke-random-random",
    ],
)
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
    "mesh_device, device_params, num_links",
    [
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
    topology = per_axis_topology(device_params["fabric_config"])
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


@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.parametrize("tokenizer", ["right", "left"], indirect=True, ids=["right_pad", "left_pad"])
@pytest.mark.parametrize("temperature", [[0.5]], ids=["temp_sweep"])
@pytest.mark.parametrize("return_kv_cache", [True], ids=["kv_cache"])
@pytest.mark.parametrize(
    "input_source, pcc_validation, use_pretrained",
    [
        (VARIANT_DEFAULT_TRACE, True, True),
        ("json_prompts", False, True),
        ("json_prompts", False, False),
        ("random", False, True),
        ("random", False, False),
    ],
    ids=[
        "pcc-variant_default-pretrained",
        "smoke-json_prompts-pretrained",
        "smoke-json_prompts-random",
        "smoke-random-pretrained",
        "smoke-random-random",
    ],
)
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
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
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
    topology = per_axis_topology(device_params["fabric_config"])
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


@pytest.mark.skipif(not is_blackhole(), reason="GLM-5.1 requires Blackhole")
@pytest.mark.parametrize("tokenizer", ["right", "left"], indirect=True, ids=["right_pad", "left_pad"])
@pytest.mark.parametrize("temperature", [[0.5]], ids=["temp_sweep"])
@pytest.mark.parametrize("return_kv_cache", [True], ids=["kv_cache"])
@pytest.mark.parametrize("use_pretrained", [True], ids=["pretrained"])
@pytest.mark.parametrize(
    "input_source, pcc_validation",
    [
        (VARIANT_DEFAULT_TRACE, True),
        ("json_prompts", False),
    ],
    ids=["pcc-variant_default", "smoke-json_prompts"],
)
@pytest.mark.parametrize("is_balanced", [False], ids=["non_balanced"])
@pytest.mark.parametrize(
    "isl_total, dispatch_buffer_capacity_factor",
    [(SEQ_LEN_5K, 8)],
    ids=["5k"],
)
@pytest.mark.parametrize(
    "num_layers",
    [
        5,
        12,
        pytest.param(78, marks=pytest.mark.skipif(not is_galaxy(), reason="Full 78-layer prefill only on Galaxy")),
    ],
    ids=["5_layers", "12_layers", "78_layers"],
)
@pytest.mark.parametrize(
    "n_routed_experts, gate_fallback_mode",
    [(256, GateComputeMode.DEVICE)],
    ids=["e256_device"],
)
@pytest.mark.parametrize("determinism_check", [False], ids=["no_determinism"])
@pytest.mark.parametrize("num_iterations", [1], ids=["iter1"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=GLM51Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_1", "glm_5_2"], indirect=True, ids=["glm51", "glm52"])
@pytest.mark.timeout(0)
def test_glm_prefill_transformer(
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
    topology = per_axis_topology(device_params["fabric_config"])
    # Full-transformer end-to-end validates against the GPU trace (variant.test_prefill_trace_default;
    # approach B) — MLA/DSA + MoE correctness live in their op-level tests.
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


# ---------------------------------------------------------------------------
# Mistral-Small-4-119B transformer test
# ---------------------------------------------------------------------------
# This variant does NOT go through run_model(), and the reason is structural rather than stylistic:
# run_model()'s two weight paths both dead-end for Mistral.
#   * random weights  -> create_hf_model() -> `variant.reference_model_cls`, which the adapter leaves
#     UNSET on purpose (the base property raises NotImplementedError). Every `reference_*_cls` hook is
#     unset there, each with the construction site that rejects it recorded in the adapter.
#   * pretrained      -> load_and_compute_layer_by_layer(), which needs the same
#     `reference_model_cls` to compute a reference, and whose cache build wants per-expert
#     gate_proj/up_proj tensors this checkpoint does not have: `packed_expert_checkpoint = True`
#     (mlp.experts.gate_up_proj is one [128, 4096, 4096] fp8 tensor), so the pretrained fixture loads
#     attention only (routed_expert_weights = None).
# GLM has no `reference_model_cls` either and dodges this by validating against a recorded GPU golden
# (`variant.test_prefill_trace_default`, input_source=variant_default). Mistral has no golden:
# `prefill_trace_default = None`, and run_model asserts a usable one for that input_source. So the
# only source of truth available is the one this repo owns — the composed CPU reference
# `reference.mistral_small_4.mistral4_decoder_layer_reference`, which is a SINGLE decoder layer. This
# test therefore CHAINS it (each layer's output is the next layer's input) and brackets the chain with
# the embedding gather, the final RMSNorm and the LM-head projection, producing exactly the snapshot
# list run_model's host-reference branch produces:
#
#     ["embed"] + [f"layer_{i}" for i in range(num_layers)] + ["norm", "lm_head"]
#
# which is also how the GLM golden's `trace.ref_snapshots` is keyed — so the comparison point is
# identical to test_glm_prefill_transformer's, and _compare_intermediate_pcc is reused verbatim
# (including its lm_head special case, which compares the single next-token position of the reference's
# full-sequence logits against TT's "logits" extract).
#
# Same device wiring as test_prefill_block.test_mistral4_prefill_block, which validates one block of
# this model on this mesh: (8, 4) = SP 8 x TP 4, FABRIC_1D with a router sized to this model's own
# FABRIC_PAYLOAD_SIZE (4096 == hidden), l1_small_size 768, num_links 2, Linear, is_balanced=False,
# routing_use_l1_small_for_semaphores=True, dense-MLA BFP8_TILE KVPE, dispatch capacity factor 8.
#
# seq_len 5120 for the same hard reason as the block test: 8192 is the ceiling for ANY comparison of
# this model. Above 8192 the real model scales queries by 1 + 0.1*log(1 + floor(pos/8192))
# (get_llama_4_attn_scale) and ttMLA has no equivalent, so a longer run measures that gap instead of
# the port. For the same reason `apply_llama4_attn_scale` is left at its default False and never passed
# to the reference: False is what the device computes, so False is what a PCC test must compare
# against. reference/mistral_small_4/block.py implements the flag so the gap can be demonstrated
# deliberately rather than hidden.
#
# num_layers is small on purpose. This is a wiring + PCC test over a 119B model whose random weights
# are materialized on host (~6.4 GB of bf16 routed experts per layer), and every layer is also replayed
# on CPU by the reference. 5 is the low rung the rest of the family uses ("5_layers"); 2 is the cheap
# default. There is deliberately no 36-layer row — full-depth coverage is a golden-trace job, and this
# model has no golden.
#
# NOT compared here: the per-layer KVPE cache. This model is rope_interleave=True, so the PE half needs
# the interleave-aware halves comparison (test_prefill_block_chunked's cache_half_pccs(pe_interleave=
# True)) rather than run_model's straight slice-and-compare, and KVPE fidelity is already covered by
# test_mla.test_mistral4_mla. So no return_kv_cache axis is carried rather than carrying a dead one —
# same for determinism_check / num_iterations, which measure device repeatability and are orthogonal to
# the reference this test exists to build.
#
# The threshold is PCC_THRESHOLD (0.99) — run_model's own rung for a random-weights host-reference run,
# used unchanged. The 0.985 rung next to it exists because DeepSeek SHRINKS its expert count for cheap
# runs (n_routed_experts < 256); Mistral runs its full 128, so that allowance does not transfer. See
# the EXPECTED-FAILURE note in the test body for why the MoE path cannot reach it today; the number it
# prints is the size of a real gap and lowering this constant would throw that measurement away.


def _mistral4_random_mla_weights(config, seed: int) -> dict:
    """Seed-parametric clone of the ``random_weights`` fixture's MLA dict (same shapes, same std).

    The fixture hands out ONE layer's weights off a fixed seed; a multi-layer transformer needs a
    distinct set per layer, because identical weights on every layer would let a layer-index / KV-slot
    mixup pass PCC silently — which is most of what a multi-layer wiring test exists to catch.
    """
    g = torch.Generator().manual_seed(seed)
    std = config.initializer_range

    def _w(*shape):
        return (torch.randn(*shape, generator=g) * std).to(torch.bfloat16)

    return {
        "q_a_proj.weight": _w(config.q_lora_rank, config.hidden_size),
        "q_a_layernorm.weight": torch.ones(config.q_lora_rank, dtype=torch.bfloat16),
        "q_b_proj.weight": _w(
            config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
            config.q_lora_rank,
        ),
        "kv_a_proj_with_mqa.weight": _w(config.kv_lora_rank + config.qk_rope_head_dim, config.hidden_size),
        "kv_a_layernorm.weight": torch.ones(config.kv_lora_rank, dtype=torch.bfloat16),
        "kv_b_proj.weight": _w(
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            config.kv_lora_rank,
        ),
        "o_proj.weight": _w(config.hidden_size, config.num_attention_heads * config.v_head_dim),
    }


def _mistral4_random_moe_weights(hidden: int, moe_intermediate: int, n_routed: int, seed: int):
    """Random gate + 128 routed experts + 1 shared expert, ``[out, in]`` throughout.

    ``e_score_correction_bias`` is emitted as ZEROS, not drawn: Mistral's router (HF
    ``Mistral4TopkRouter``) owns only ``weight`` and has no correction bias at all, but the device gate
    indexes that key unconditionally (tt/moe/tt_moe.py:100 and :275), so it is handed the mathematical
    identity. The composed reference ignores the key entirely (reference/mistral_small_4/moe.py). A
    non-zero bias here would also break the one thing that still agrees between the two routers — see
    the EXPECTED-FAILURE note in the test body.
    """
    g = torch.Generator().manual_seed(seed)
    hs, ds = hidden**-0.5, moe_intermediate**-0.5

    def _expert():
        return {
            "gate_proj": (torch.randn(moe_intermediate, hidden, generator=g) * hs).to(torch.bfloat16),
            "up_proj": (torch.randn(moe_intermediate, hidden, generator=g) * hs).to(torch.bfloat16),
            "down_proj": (torch.randn(hidden, moe_intermediate, generator=g) * ds).to(torch.bfloat16),
        }

    gate_weights = {
        "weight": (torch.randn(n_routed, hidden, generator=g) * hs).to(torch.bfloat16),
        "e_score_correction_bias": torch.zeros(n_routed, dtype=torch.float32),
    }
    return gate_weights, [_expert() for _ in range(n_routed)], _expert()


def _mistral4_random_transformer_weights(config, num_layers: int) -> dict:
    """Whole-model random weights in TtPrefillTransformer's state_dict format (extract_tt_state_dict's).

    Adds the one key extract_tt_state_dict does NOT produce: ``lm_head_weight``. The HF reference model
    it extracts from is a ``*Model`` (no LM head), so DeepSeek's random rows are all smoke-only and TT
    silently runs the head on ``torch.empty``. A PCC run has to compare against a real head, and
    TtLMHead asserts the weight is exactly ``(vocab_size, emb_dim)``.
    """
    hidden = config.hidden_size
    layers = []
    for i in range(num_layers):
        # Per-layer seed base, spaced so no two layers share a generator stream.
        base = 1000 * (i + 1)
        gate_weights, routed, shared = _mistral4_random_moe_weights(
            hidden, config.moe_intermediate_size, config.n_routed_experts, seed=base + 3
        )
        layers.append(
            {
                "attn_norm_weight": (
                    torch.randn(hidden, generator=torch.Generator().manual_seed(base + 1)) * 0.1 + 1.0
                ).to(torch.bfloat16),
                "mla_weights": _mistral4_random_mla_weights(config, seed=base),
                "ffn_norm_weight": (
                    torch.randn(hidden, generator=torch.Generator().manual_seed(base + 2)) * 0.1 + 1.0
                ).to(torch.bfloat16),
                "gate_weights": gate_weights,
                "routed_expert_weights": routed,
                "shared_expert_weights": shared,
                # No "ffn_weights": first_k_dense_replace = 0, so every one of this model's layers is
                # MoE and TtPrefillBlock's `is_moe = layer_idx >= model_cfg.NUM_DENSE_LAYERS` is always
                # True. (The block test's _Mistral4SyntheticDenseConfig shim exists only to build a
                # synthetic dense block in isolation; there is no dense layer type to parametrize here.)
            }
        )
    g = torch.Generator().manual_seed(7)
    return {
        # fp32, mirroring extract_tt_state_dict's `.float()` on the embedding; TtParallelEmbedding
        # uploads it as bf16, so the reference casts on gather (see _mistral4_reference_snapshots).
        "embed_weight": torch.randn(config.vocab_size, hidden, generator=g) * config.initializer_range,
        "norm_weight": (torch.randn(hidden, generator=g) * 0.1 + 1.0).to(torch.bfloat16),
        "lm_head_weight": (torch.randn(config.vocab_size, hidden, generator=g) * hidden**-0.5).to(torch.bfloat16),
        "layers": layers,
    }


def _mistral4_reference_snapshots(config, state_dict: dict, token_ids, num_layers: int, seq_len: int, release: bool):
    """Chain the single-layer CPU reference into run_model's host-reference snapshot list.

    Returns ``[embed, layer_0, ..., layer_{num_layers-1}, norm, lm_head]``, i.e. the same stages, in the
    same order, with the same dtypes as load_and_compute_layer_by_layer produces for the pretrained
    variants — so the labels below line up 1:1 with TT's intermediates dict.

    ``release=True`` clears each layer's host weights once its reference output exists. The device
    already holds its own ttnn copies, and at 128 routed experts a layer is ~6.4 GB of bf16.
    """
    hidden = config.hidden_size
    # Embedding: a plain row gather, cast to bf16 because that is what TtParallelEmbedding emits — the
    # cast belongs on the reference side at the very first comparison point, not later.
    h = state_dict["embed_weight"][token_ids[0]].unsqueeze(0).to(torch.bfloat16)  # [1, seq, hidden]
    snapshots = [h]
    # Passed explicitly: the vendored apply_rotary_pos_emb indexes cos[position_ids] and mis-indexes on
    # None. Sequential (non-balanced) layout, so plain arange.
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    for i in range(num_layers):
        layer = state_dict["layers"][i]
        logger.info(f"[mistral4 transformer] CPU reference layer {i}/{num_layers}")
        h, _ = mistral4_decoder_layer_reference(
            config,
            layer["mla_weights"],
            layer["attn_norm_weight"],
            layer["ffn_norm_weight"],
            h,
            seq_len,
            moe_weights={
                "gate_weights": layer["gate_weights"],
                "routed_expert_weights": layer["routed_expert_weights"],
                "shared_expert_weights": layer["shared_expert_weights"],
            },
            position_ids=position_ids,
            # apply_llama4_attn_scale stays at its default False — the device has no equivalent, and
            # the scale is exactly 1.0 below 8192 anyway. See the header.
        )
        snapshots.append(h)
        # The second return is the layer's KVPE cache in test_mla's layout; dropped on purpose (see
        # the header: rope_interleave=True needs the interleave-aware halves comparison).
        if release:
            layer.clear()
            gc.collect()
    # Final RMSNorm, then the LM head over the FULL sequence: _compare_intermediate_pcc narrows the
    # reference at actual_isl - 1, so a short reference would silently become an ERROR row rather than a
    # readable failure. Same two ops load_and_compute_layer_by_layer's tail runs.
    h = rms_norm(h, state_dict["norm_weight"], config.rms_norm_eps)
    snapshots.append(h)
    snapshots.append(torch.nn.functional.linear(h.to(torch.bfloat16), state_dict["lm_head_weight"]))
    assert snapshots[-1].shape[-1] == config.vocab_size and snapshots[-1].shape[-2] == seq_len
    assert h.shape == (1, seq_len, hidden)
    return snapshots


@pytest.mark.skipif(not is_blackhole(), reason="Mistral-Small-4 is validated on Blackhole only")
# Pretrained is not an option, not merely unselected: packed_expert_checkpoint = True means the
# pretrained fixture loads attention only, so an all-MoE model would run on no experts.
@pytest.mark.parametrize("use_pretrained", [False], ids=["random_weights"])
# input_source is never "variant_default": this variant's prefill_trace_default is None and run_model
# asserts a usable one for that source. "random" needs no golden — the reference is composed on host.
@pytest.mark.parametrize(
    "input_source, pcc_validation",
    [
        ("random", True),
        ("random", False),
    ],
    ids=["pcc-random", "smoke-random"],
)
@pytest.mark.parametrize("is_balanced", [False], ids=["non_balanced"])
@pytest.mark.parametrize("isl_total, dispatch_buffer_capacity_factor", [(SEQ_LEN_5K, 8)], ids=["5k"])
@pytest.mark.parametrize("num_layers", [2, 5], ids=["2_layers", "5_layers"])
# The model's real expert count, unreduced: n_routed_experts is read off the shared cached config by
# the device path, so shrinking it would mean mutating that object. DEVICE_FP32 is the adapter's
# default_gate_mode (n_group = 1 with a device gate).
@pytest.mark.parametrize(
    "n_routed_experts, gate_fallback_mode",
    [(128, GateComputeMode.DEVICE_FP32)],
    ids=["e128_device_fp32"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                # (sp, tp) = (8, 4): SP 8 on axis 0, TP 4 on axis 1 — the whole-galaxy shape the
                # mistral4 long-seq matmul configs were tuned on. FABRIC_1D + a router config sized to
                # this model's own FABRIC_PAYLOAD_SIZE (4096 == hidden), matching its dense-MLA + MoE
                # family (test_kimi_prefill_transformer) rather than GLM's DSA FABRIC_2D.
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=Mistral4Small119BConfig.FABRIC_PAYLOAD_SIZE
                ),
                # The adapter's l1_small_size, verbatim: routing_use_l1_small_for_semaphores=True
                # (below) puts the MoE routing all-gather's semaphores in L1_SMALL. Routing consumes
                # 512 B; the remaining 256 B is for MLA high-bandwidth-gather semaphores.
                "l1_small_size": 768,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["mistral_small_4_119b"], indirect=True, ids=["mistral4"])
@pytest.mark.timeout(0)
def test_mistral4_prefill_transformer(
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
    input_source,
    use_pretrained,
):
    """embed -> [Mistral-Small-4 block x num_layers] -> norm -> lm_head vs the chained CPU reference."""
    torch.manual_seed(42)
    assert not use_pretrained and input_source == "random", "see the decorator comments"
    profiler.clear()
    profiler.start("total_test_time")

    config = config_only
    # Device-path field only (rope-table sizing); the composed reference never reads it, and
    # mistral4_decoder_layer_reference does not mutate config either.
    config.max_seq_len = isl_total
    # Dense MLA: natural rope tables, BFP8_TILE KVPE, and no indexer key cache to thread through
    # forward(). Asserted rather than assumed, so a config that ever grew an indexer fails loudly here
    # instead of silently needing the indexed-rope + index_kv_cache wiring.
    assert not resolve_has_indexer(config), "mistral4 is dense MLA; the sparse/DSA wiring is not set up here"
    assert n_routed_experts == config.n_routed_experts, "the expert count is not reduced for this variant"
    # Sequential layout only. run_model() permutes the token IDs with create_balanced_chunk_order
    # before sharding when is_balanced; this body does not, and the reference chains with plain
    # arange() positions, which IS the sequential layout. So adding True to the axis above without
    # also wiring the reorder would produce a silently wrong reference (the transformer un-permutes its
    # intermediates on the way out, but the tokens would never have been permuted on the way in).
    assert not is_balanced, "the zigzag layout needs the token reorder run_model() does; wire it before enabling"

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp_factor = mesh_shape[sp_axis]
    isl_per_chip = isl_total // sp_factor
    # Right padding with actual_isl == isl_total: every token is real, so there is no padded tail to
    # slice and the padding-aware routing config comes out the identity. No tokenizer is involved —
    # input_source="random" draws token IDs directly — so padding_side is stated rather than derived.
    padding_side = "right"
    number_of_non_padded_tokens = isl_total
    logger.info(
        f"[mistral4 transformer] mesh={mesh_shape} isl_total={isl_total} isl_per_chip={isl_per_chip} "
        f"num_layers={num_layers} n_routed_experts={n_routed_experts} gate_fallback_mode={gate_fallback_mode} "
        f"pcc_validation={pcc_validation}"
    )

    # --- random weights, shared by device and reference ---
    profiler.start("weights_creation")
    state_dict = _mistral4_random_transformer_weights(config, num_layers)
    profiler.end("weights_creation")
    token_ids = torch.randint(0, config.vocab_size, (1, isl_total), dtype=torch.int64)

    # --- device transformer ---
    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=Mistral4Small119BConfig,
        state_dict=state_dict,
        num_layers=num_layers,
        seq_len=isl_total,
        is_balanced=is_balanced,
        padding_side=padding_side,
        # Worst case for the flat dispatch buffer: the whole SP group (8 chips x 640 tokens = 5120)
        # lands on one of this chip's 4 local experts, i.e. 4 * 5120 = 20480 raw tokens == factor 4.
        # 8 is 2x headroom on that bound (and the value the rest of this file runs with).
        dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=gate_fallback_mode,
        # Random weights only: nothing is loaded from, or written to, a ttnn cache. (The adapter's
        # ttnn_cache_default is "" — no prefill cache has been built for this variant.)
        weight_cache_path=None,
        lm_head_is_column_parallel=True,
        # n_group = 1 with a device gate, so the routing all-gather's semaphores go to L1_SMALL
        # (paired with l1_small_size=768 in device_params above).
        routing_use_l1_small_for_semaphores=True,
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_transformer_creation")
    if not pcc_validation:
        # Smoke row: nothing reads the host weights again.
        state_dict = None
        gc.collect()

    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,  # dense-MLA format, as in test_mla.test_mistral4_mla
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=isl_total,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
    )

    # --- SP-shard the token IDs: [1, isl_total] -> [sp_factor, 1, isl_per_chip] ---
    tt_tokens = ttnn.from_torch(
        token_ids.reshape(sp_factor, 1, isl_per_chip),
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
    )

    # --- forward ---
    profiler.start("tt_forward")
    logger.info("[mistral4 transformer] running TtPrefillTransformer forward")
    first_token_id, first_token_prob, tt_intermediates = transformer(
        tt_tokens,
        kvpe_cache,
        actual_isl=number_of_non_padded_tokens,
        return_intermediates=pcc_validation,
        read_profiler=False,
        # Argmax (temperature 0.0): this test asserts on PCC, not on a sampled token, so the sampler is
        # kept deterministic instead of carrying a temperature sweep axis.
        temperature=0.0,
        index_kv_cache=None,  # dense MLA
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")
    logger.info(f"[mistral4 transformer] first token: ID={first_token_id}, prob={first_token_prob:.4f}")

    if not pcc_validation:
        profiler.end("total_test_time")
        logger.success(f"TtPrefillTransformer mistral4 smoke test passed (num_layers={num_layers})")
        ttnn.synchronize_device(mesh_device)
        return

    # --- chained CPU reference (reference/mistral_small_4) ---
    assert tt_intermediates is not None, "Expected intermediates dict"
    profiler.start("reference_computation")
    logger.info(
        f"[mistral4 transformer] chaining {num_layers} x "
        f"reference.mistral_small_4.mistral4_decoder_layer_reference on host"
    )
    ref_snapshots = _mistral4_reference_snapshots(config, state_dict, token_ids, num_layers, isl_total, release=True)
    profiler.end("reference_computation")

    profiler.start("pcc_validation")
    # Exactly run_model's host-reference labels, in exactly its order.
    ref_labels = ["embed"] + [f"layer_{i}" for i in range(num_layers)] + ["norm", "lm_head"]
    pcc_results = _compare_intermediate_pcc(
        zip(ref_labels, ref_snapshots), tt_intermediates, number_of_non_padded_tokens, padding_side
    )
    profiler.end("pcc_validation")

    # run_model's own rung for a random-weights host-reference run, used unchanged — see the header for
    # why the neighbouring 0.985 allowance does not apply to this variant.
    threshold = PCC_THRESHOLD
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

    profiler.end("total_test_time")
    logger.info(f"\n{'='*60}")
    logger.info("Timing Report")
    logger.info(f"{'='*60}")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")

    ttnn.synchronize_device(mesh_device)
    # EXPECTED FAILURE from layer_0 onward — wired faithfully on purpose, NOT tuned to pass. The real
    # router is softmax (`router_logits.softmax(-1)`, transformers/models/mistral4/modeling_mistral4.py
    # :226, which the reference executes via HF's own Mistral4MoE.route_tokens_to_experts), while the
    # device gate is sigmoid-only: the ttnn op rejects anything else (moe_grouped_topk.cpp:23 TT_THROW),
    # the host golden agrees (tt/moe/validation_helpers.py:27), and mistral4 declares no SCORE_FUNC so
    # TtMoEGatePrefill takes the sigmoid default (tt_moe_gate_prefill.py:155). Both are monotone in the
    # logits, so with the zero correction bias above the top-4 SELECTION still matches (a non-zero bias
    # would break even that); the top-4 WEIGHTS never do, because softmax-then-renormalize !=
    # sigmoid-then-normalize over the same four logits and both sides normalize (norm_topk_prob=True,
    # route_scale=1.0). The gap then compounds layer over layer: "embed" is the only row upstream of a
    # router, so it passes, and every row from layer_0 on — including "norm" and "lm_head", which are
    # downstream of layer_{num_layers-1} — carries the gap. Expect those to land high (0.99x) rather
    # than obviously broken, because the top-4 selection agrees and only the four weights differ; a
    # near-threshold number here is the measurement, NOT an argument for bumping the threshold. The
    # threshold is NOT lowered, the reference is NOT switched to sigmoid, and this is NOT xfailed: what
    # would fix it is a softmax score_func on the device op, and any of those three would throw away
    # the measurement that sizes the work.
    if failures:
        pytest.fail(f"PCC below {threshold} at: " + "; ".join(f"{label}: {pcc:.6f}" for label, pcc in failures))
    logger.success(
        f"TtPrefillTransformer mistral4 PCC test passed (num_layers={num_layers}, "
        f"n_routed_experts={n_routed_experts}, gate_fallback_mode={gate_fallback_mode}, weights=random)"
    )

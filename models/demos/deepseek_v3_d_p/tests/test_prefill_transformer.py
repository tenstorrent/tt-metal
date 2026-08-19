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
from models.demos.deepseek_v3_d_p.reference.mistral_small4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tests.conftest import FABRIC_2D_PREFILL_BLOCK_MESH_PARAMS
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank, resolve_has_indexer
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
from models.demos.deepseek_v3_d_p.utils.test_utils import save_intermediate_output, token_normalized
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


def _compare_intermediate_pcc(
    reference_items, tt_intermediates, number_of_non_padded_tokens, padding_side, robust_out=None
):
    """Per-stage PCC of TT intermediates against the reference, as [(label, pcc), ...].

    `robust_out`, if given, is additionally filled with a per-token RMS-normalized PCC per label.
    That second number exists because raw PCC on late-layer hidden states measures almost nothing
    but outliers: this model develops massive activations (measured on the 1k reference — absmax
    429 vs rms 0.35 at layer 5, where the top 0.01% of elements hold 95% of the total squared
    energy, and max/rms is still ~120-165 through layers 30-35). A handful of channels therefore
    set the raw PCC, which is why the tail layers can read ~0.17 while `norm` — computed from
    layer 35's own output, but per-token rescaled by RMSNorm — reads 0.959 and the sampled first
    token matches the reference exactly. Normalizing the same way makes the per-layer curve
    comparable across depth. The pass/fail gate stays on the raw metric; this is an instrument.
    """
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
            ref_slice = slice_non_padded(ref_host, number_of_non_padded_tokens, padding_side).float()
            tt_slice = slice_non_padded(tt_host, number_of_non_padded_tokens, padding_side).float()
            _, pcc = comp_pcc(ref_slice, tt_slice)
            logger.debug(f"{label:<20s}  PCC = {pcc:.6f}")
            pcc_results.append((label, pcc))
            if robust_out is not None:
                try:
                    _, npcc = comp_pcc(token_normalized(ref_slice), token_normalized(tt_slice))
                    robust_out[label] = npcc
                except Exception as e:  # the raw number is the gate; this one is diagnostic
                    logger.debug(f"{label:<20s}  normalized PCC unavailable: {e}")
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
    # per-layer KVPE cache, the indexer stride is the COMPACTED full-indexer count over the layers this
    # instance builds — GLM-5.2 "shared" layers reuse a "full" layer's cache and get no slot of their own.
    # full_indexer_rank returns num_layers unchanged when there is no indexer_types map. Dense variants use
    # no index cache.
    tt_index_kv_cache = None
    if has_indexer:
        index_cache_layers = full_indexer_rank(config, num_layers)
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

        robust_pcc = {}
        pcc_results.extend(
            _compare_intermediate_pcc(
                reference_items,
                tt_intermediates,
                number_of_non_padded_tokens,
                padding_side,
                robust_out=robust_pcc,
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
            # A reference that stores anything other than the compressed [kv | pe] line cannot be
            # compared against the device's latent cache, and the slices below would not say so:
            # `[..., :kv_lora_rank]` on a narrower last dim just clamps. Name the mismatch instead.
            expected_last = kv_lora_rank + config.qk_rope_head_dim
            if ref_kvpe_list and ref_kvpe_list[0].shape[-1] != expected_last:
                raise AssertionError(
                    f"reference KVPE last dim {ref_kvpe_list[0].shape[-1]} != {expected_last} "
                    f"(kv_lora_rank {kv_lora_rank} + qk_rope_head_dim {config.qk_rope_head_dim}); "
                    f"shape {tuple(ref_kvpe_list[0].shape)} is not the compressed MLA line the "
                    "device caches -- see reference_kvpe_for_layer"
                )
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
        logger.info(f"\n{'='*64}")
        logger.info(f"{'Stage':<20s}  {'PCC':>10s}  {'nPCC':>10s}  {'Status':>8s}")
        logger.info(f"{'-'*64}")
        failures = []
        for label, pcc in pcc_results:
            status = "PASS" if pcc >= threshold else ("FAIL" if pcc >= 0 else "ERROR")
            # nPCC = per-token RMS-normalized PCC; see _compare_intermediate_pcc. Blank where it
            # does not apply (KVPE rows, logits).
            npcc = robust_pcc.get(label)
            npcc_s = f"{npcc:>10.6f}" if npcc is not None else f"{'-':>10s}"
            logger.info(f"{label:<20s}  {pcc:>10.6f}  {npcc_s}  {status:>8s}")
            if pcc < threshold:
                failures.append((label, pcc))
        logger.info(f"{'='*64}")

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


# Mistral Small 4 119B, full stack: embed -> [block x N] -> norm -> lm_head -> sample.
# Random weights only: the TT state_dict is derived from the SAME HF model instance the reference
# uses (`create_hf_model` -> `extract_tt_state_dict`), so no checkpoint is needed. n_routed_experts
# stays at the real 128 (4 experts/chip on a 32-device mesh) rather than a reduced count, so the
# routing distribution is the production one. GPT_DEVICE gate: Mistral's softmax top-4 rule, see the
# adapter docstring.
@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 bring-up targets Blackhole")
@pytest.mark.parametrize("tokenizer", ["right"], indirect=True, ids=["right_pad"])
@pytest.mark.parametrize("temperature", [[0.5]], ids=["temp_sweep"])
@pytest.mark.parametrize("return_kv_cache", [True], ids=["kv_cache"])
@pytest.mark.parametrize(
    "input_source, pcc_validation, use_pretrained",
    [
        ("random", False, False),
        ("json_prompts", False, False),
        # Real checkpoint: exercises the whole weight path -- per-tensor fp8 dequant, the
        # stacked+fused expert split, the zero router bias -- and writes the TTNN cache.
        # Loading is layer-by-layer, so host memory stays flat regardless of num_layers.
        ("random", False, True),
        ("json_prompts", False, True),
    ],
    ids=[
        "smoke-random-random",
        "smoke-json_prompts-random",
        "smoke-random-pretrained",
        "smoke-json_prompts-pretrained",
    ],
)
@pytest.mark.parametrize("is_balanced", [False], ids=["non_balanced"])
@pytest.mark.parametrize(
    "isl_total, dispatch_buffer_capacity_factor",
    [(SEQ_LEN_1K, 8), (SEQ_LEN_5K, 8)],
    ids=["1k", "5k"],
)
@pytest.mark.parametrize(
    "num_layers",
    [
        2,
        5,
        # 9 = 36/4, one PP=4 stage's depth. Pair with mesh-8x1 to run exactly what a stage would.
        9,
        pytest.param(36, marks=pytest.mark.skipif(not is_galaxy(), reason="Full 36-layer model only on Galaxy")),
    ],
    ids=["2_layers", "5_layers", "9_layers", "36_layers"],
)
@pytest.mark.parametrize(
    "n_routed_experts, gate_fallback_mode",
    [(MistralSmall4Config.NUM_ROUTED_EXPERTS, GateComputeMode.GPT_DEVICE)],
    ids=["e128_gpt_device"],
)
@pytest.mark.parametrize("determinism_check", [False], ids=["no_determinism"])
@pytest.mark.parametrize("num_iterations", [1], ids=["iter1"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
        # TP=1 probe for the PP=4 x (8,1) proposal. A PP stage is SP=8 x TP=1; (32,1) is the closest
        # shape reachable without an 8-chip TT_VISIBLE_DEVICES carve, and it is CI-listed for
        # BLACKHOLE_GALAXY with FABRIC_1D (conftest CI_ALLOWED_FABRICS). It keeps experts_per_chip at
        # 128/32 = 4, the same as mesh-8x4, so the MoE side is unchanged and TP=1 is the only variable.
        # What this actually tests: that the whole stack (embedding, MLA, MoE, norms, LM head) builds
        # and runs with no TP axis at all — every collective in the dense path is on the TP axis, so
        # TP=1 is the one shape where they all disappear.
        pytest.param(
            (32, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2,
            ttnn.Topology.Linear,
            id="mesh-32x1",
        ),
        # The ACTUAL PP=4 stage geometry: SP=8 x TP=1 on 8 chips. Needs an 8-chip carve
        # (TT_VISIBLE_DEVICES=0..7) since every BLACKHOLE_GALAXY CI_ALLOWED_FABRICS entry uses all 32.
        #
        # Why this and not mesh-32x1: (32,1) over-shards the sequence. SP=32 leaves isl/32 tokens per
        # chip — 32 at isl 1k — and masked_bincount (the MoE routing setup) TT_FATALs unless the
        # per-chip token count is a multiple of its 64-core grid. At SP=8 it is isl/8 = 128, which is
        # fine. That also constrains PP chunk sizes: chunk/8 must be a multiple of 64, i.e. chunk must
        # be a multiple of 512 (CHUNK=5120 -> 640/chip, OK).
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2,
            ttnn.Topology.Linear,
            id="mesh-8x1",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["mistral_small4"], indirect=True, ids=["mistral4"])
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

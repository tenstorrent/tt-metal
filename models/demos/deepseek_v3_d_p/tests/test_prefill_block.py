# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TtPrefillBlock — verifies composition of norm → MLA → residual → norm → FFN/MoE → residual.

Validates output shapes and PCC against torch reference.

Uses HF DeepseekV3Model layer as the reference: creates a model with random weights,
extracts those weights into our TT state_dict format, and compares forward passes.

Runs the block forward for `num_iterations` iterations, producing per-iteration PCC
numbers under two mutually exclusive validation modes:
  * pcc_validation: compare each iteration's output (and KVPE) against the torch reference.
  * determinism_check: compare iter N>=1 against the iter-0 baseline (near-bit-identical).
Per-iteration PCC is buffered and printed as a summary; the assertion is deferred until
after all iterations and the timing report.
"""

import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import DynamicCache

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3.demo.demo import load_prompts_from_json
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ABC_1K_PATH,
    create_hf_model,
    extract_layer_state_dict,
    get_4d_causal_mask,
    tokenize_prompt_to_isl,
)
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD_DENSE = 0.996
PCC_THRESHOLD_MOE_GATE_HOST = 0.996
PCC_THRESHOLD_MOE_GATE_DEVICE = 0.992
PCC_THRESHOLD_KVPE = 0.999
# Determinism: every iteration is expected to be (near-)bit-identical to iter 0.
DETERMINISM_PCC_THRESHOLD = 0.9999


def _threshold_for(label, determinism_check, output_threshold):
    """Resolve the PCC threshold for a given per-iteration result label."""
    if determinism_check:
        return DETERMINISM_PCC_THRESHOLD
    if label.endswith("_kv") or label.endswith("_pe"):
        return PCC_THRESHOLD_KVPE
    return output_threshold


@pytest.mark.parametrize(
    "input_source, pcc_validation, isl_total, dispatch_buffer_capacity_factor",
    [
        ("random", False, 1024, 8),
        ("abc_1k", False, 25 * 1024, 8),
        ("abc_1k", True, 1024, 8),
    ],
    ids=["smoke-random", "perf-abc_25k", "pcc-abc_1k"],
)
@pytest.mark.parametrize(
    "layer_type, gate_fallback_mode",
    [
        ("dense", None),
        ("moe", GateComputeMode.DEVICE),
    ],
    ids=["dense", "moe-gate_device"],
)
@pytest.mark.parametrize("is_balanced", [True, False], ids=["balanced", "non_balanced"])
@pytest.mark.parametrize("determinism_check", [False, True], ids=["no_determinism", "with_determinism"])
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
@pytest.mark.timeout(600)
def test_prefill_block(
    config_only,
    mesh_device,
    device_params,
    is_balanced,
    isl_total,
    dispatch_buffer_capacity_factor,
    layer_type,
    gate_fallback_mode,
    num_links,
    topology,
    pcc_validation,
    determinism_check,
    num_iterations,
    input_source,
    tokenizer,
    is_ci_env,
    is_ci_v2_env,
):
    if is_ci_env or is_ci_v2_env and pcc_validation == False:
        pytest.skip("Skip non-PCC test in CI to save time")
    if (is_ci_env or is_ci_v2_env) and not is_balanced:
        pytest.skip("Skip non_balanced variant in CI — runnable locally for non_balanced-mode validation")

    # determinism_check and pcc_validation are mutually exclusive validation modes:
    # determinism compares iter N against iter 0; pcc_validation compares against torch reference.
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

    # layer_idx=0 for dense (< NUM_DENSE_LAYERS=3), layer_idx=3 for MoE (>= 3)
    layer_idx = 0 if layer_type == "dense" else DeepSeekV3Config.NUM_DENSE_LAYERS

    logger.info(f"mesh_shape={mesh_shape}, sp_factor={sp_factor}, tp_factor={tp_factor}")
    logger.info(
        f"isl_total={isl_total}, isl_per_chip={isl_per_chip}, "
        f"layer_type={layer_type}, layer_idx={layer_idx}, gate_fallback_mode={gate_fallback_mode}, "
        f"input_source={input_source}"
    )

    # --- Cache setup ---
    is_dense = layer_idx < DeepSeekV3Config.NUM_DENSE_LAYERS
    cache_dir = Path(f"/tmp/deepseek_v3_prefill_block/{layer_type}_{sp_factor}x{tp_factor}mesh_{isl_total}isl")
    cache_dir.mkdir(parents=True, exist_ok=True)

    init_checker(cache_dir)
    ttnn_cache_complete = TtPrefillBlock.check_cache_complete(cache_dir, layer_idx, is_dense)
    torch_ref_cache = cache_dir / f"torch_reference_{input_source}.pt"

    ref_cache_loadable = torch_ref_cache.exists() and (pcc_validation or input_source == "abc_1k")
    need_hf_model = not ttnn_cache_complete or ((pcc_validation or input_source == "abc_1k") and not ref_cache_loadable)
    logger.info(
        f"Cache status: TTNN={ttnn_cache_complete}, ref_cache={torch_ref_cache.exists()}, "
        f"need_hf_model={need_hf_model}"
    )

    # --- Build HF reference model and extract weights ---
    num_layers = layer_idx + 1
    hf_model = None
    if need_hf_model:
        profiler.start("weights_creation")
        torch.manual_seed(42)
        hf_model = create_hf_model(config, num_layers)
        hf_sd = hf_model.state_dict()
        state_dict = extract_layer_state_dict(hf_sd, layer_idx, hf_model.layers[layer_idx])
        profiler.end("weights_creation")
    else:
        logger.info("TTNN cache complete, skipping torch weight creation")
        state_dict = {}

    # --- Resolve torch_input and torch reference (single decision point for ref_cache) ---
    torch_output = None
    ref_kvpe = None
    if ref_cache_loadable:
        logger.info(f"Loading cached reference from {torch_ref_cache}")
        profiler.start("reference_loading")
        ref_cached = torch.load(torch_ref_cache, weights_only=True)
        torch_input = ref_cached["torch_input"]
        if pcc_validation:
            torch_output = ref_cached["torch_output"]
            ref_kvpe = ref_cached["ref_kvpe"]
        profiler.end("reference_loading")
    elif input_source == "abc_1k":
        profiler.start("tokenization")
        prompts = load_prompts_from_json(str(ABC_1K_PATH))
        prompt_text = prompts[0] if isinstance(prompts, list) else prompts
        token_ids, attention_mask, tokens = tokenize_prompt_to_isl(
            tokenizer, max_isl=isl_total, prompt_text=prompt_text
        )
        attention_mask = get_4d_causal_mask(attention_mask, causal_only=True)
        profiler.end("tokenization")
        logger.info(f"Tokenized ABC_1k input shape: {token_ids.shape}, first 10 tokens: {token_ids[0, :10].tolist()}")
        with torch.no_grad():
            torch_input = hf_model.embed_tokens(token_ids).to(torch.bfloat16)
        logger.info(f"Embedded input shape: {torch_input.shape}")
    else:
        torch.manual_seed(123)
        torch_input = torch.randn(1, isl_total, emb_dim, dtype=torch.bfloat16)

    if pcc_validation and torch_output is None:
        profiler.start("torch_reference")
        logger.info("Running torch reference forward...")
        position_ids = torch.arange(isl_total, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.zeros(1, 1, isl_total, isl_total, dtype=torch.bfloat16)
        ref_cache = DynamicCache()
        with torch.no_grad():
            layer_out = hf_model.layers[layer_idx](
                torch_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=ref_cache,
                use_cache=True,
            )
            torch_output = layer_out[0]
        logger.info(f"Torch reference output shape: {torch_output.shape}")
        if ref_cache is not None:
            ref_kvpe = ref_cache.key_cache[layer_idx]
            logger.info(f"Reference KVPE shape: {ref_kvpe.shape}")
        profiler.end("torch_reference")

        logger.info(f"Saving reference to {torch_ref_cache}")
        torch.save(
            {"torch_input": torch_input, "torch_output": torch_output, "ref_kvpe": ref_kvpe},
            torch_ref_cache,
        )

    # Free HF model early
    if hf_model is not None:
        del hf_model

    # --- Build TTNN cache if needed ---
    if not ttnn_cache_complete:
        logger.info("Building TTNN cache...")
        profiler.start("ttnn_cache_build")
        TtPrefillBlock.build_ttnn_cache(
            state_dict=state_dict,
            layer_idx=layer_idx,
            cache_path=cache_dir,
            mesh_device=mesh_device,
            config=config,
            seq_len=isl_total,
            num_links=num_links,
            topology=topology,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
        )
        profiler.end("ttnn_cache_build")

    # --- TT block ---
    profiler.start("tt_block_creation")
    block_kwargs = dict(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        layer_idx=layer_idx,
        seq_len=isl_total,
        dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        weight_cache_path=cache_dir,
        capacity_factor=32,
        is_balanced=is_balanced,
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

    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)
    rope_tensors = rope_setup.get_rope_tensors(isl_total)

    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
        mesh_device=mesh_device,
        seq_len=isl_total,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    # --- Output PCC threshold (used in pcc_validation mode) ---
    if layer_type == "dense":
        output_threshold = PCC_THRESHOLD_DENSE
    else:
        if gate_fallback_mode == GateComputeMode.DEVICE:
            output_threshold = PCC_THRESHOLD_MOE_GATE_DEVICE
        else:
            output_threshold = PCC_THRESHOLD_MOE_GATE_HOST

    # --- Forward (with per-iteration validation, buffered) ---
    # Two validation modes are supported, both producing per-iteration PCC numbers:
    #   * pcc_validation: compare each iter's output (and KVPE) against the torch reference.
    #   * determinism_check: compare iter N>=1 against the iter-0 baseline.
    need_validation = pcc_validation or determinism_check
    do_return_kv = need_validation  # block returns its KVPE slice when return_kv_cache=True
    kv_lora_rank = config.kv_lora_rank

    # Determinism-mode baselines captured from iter 0 (None until then).
    baseline_output = None
    baseline_kvpe = None

    profiler.start("tt_forward")
    logger.info("Running TtPrefillBlock forward...")
    # Buffer of per-iteration PCC results: list of (iter_idx, list[(label, pcc)]).
    # PCC is computed every iteration but assertions are deferred until after
    # all iterations complete and the full summary has been printed.
    per_iter_pcc_buffer = []
    shape_validated = False
    for i in range(num_iterations):
        logger.info(f"Starting iteration: {i}")
        start_time = time.time()
        tt_output, tt_kvpe = block(tt_input, rope_tensors, tt_kvpe_cache, return_kv_cache=do_return_kv)
        ttnn.synchronize_device(mesh_device)
        end_time = time.time()
        logger.info(f"Iteration {i}: {end_time - start_time} seconds")

        # --- Validate output shape (once) ---
        if not shape_validated:
            expected_per_device_shape = [1, 1, isl_per_chip, emb_dim // tp_factor]
            output_shape = list(tt_output.shape)
            assert (
                output_shape == expected_per_device_shape
            ), f"Output shape mismatch: got {output_shape}, expected {expected_per_device_shape}"
            logger.info(f"Output shape: {output_shape} (matches expected)")
            shape_validated = True

        if not need_validation:
            continue

        profiler.start("pcc_validation")
        tt_output_host = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)
        # Remove leading batch dim: [1, 1, isl_total, emb_dim] → [1, isl_total, emb_dim]
        tt_output_host = tt_output_host.squeeze(0)

        iter_pcc = []

        # Determinism iter 0: snapshot output + KVPE baseline for later iterations to
        # compare against. No PCC this iteration.
        if determinism_check and i == 0:
            baseline_output = tt_output_host.clone()
            if tt_kvpe is not None:
                baseline_kvpe = tt_kvpe.clone()
            per_iter_pcc_buffer.append((i, iter_pcc))
            logger.info(
                f"Iteration {i}: captured baseline for determinism check "
                f"(output{', KVPE' if baseline_kvpe is not None else ''})"
            )
            profiler.end("pcc_validation")
            continue

        # Resolve reference: iter-0 baseline (determinism) or torch reference (pcc_validation).
        if determinism_check:
            ref_output = baseline_output
            ref_kvpe_cmp = baseline_kvpe
        else:
            ref_output = torch_output
            ref_kvpe_cmp = ref_kvpe

        if ref_output is not None:
            _, pcc = comp_pcc(ref_output.float(), tt_output_host.float())
            iter_pcc.append(("output", pcc))

        # --- KVPE cache validation ---
        if ref_kvpe_cmp is not None and tt_kvpe is not None:
            _, kv_pcc = comp_pcc(ref_kvpe_cmp[:, :, :, :kv_lora_rank].float(), tt_kvpe[:, :, :, :kv_lora_rank].float())
            _, pe_pcc = comp_pcc(ref_kvpe_cmp[:, :, :, kv_lora_rank:].float(), tt_kvpe[:, :, :, kv_lora_rank:].float())
            iter_pcc.append(("kvpe_kv", kv_pcc))
            iter_pcc.append(("kvpe_pe", pe_pcc))

        per_iter_pcc_buffer.append((i, iter_pcc))
        profiler.end("pcc_validation")

    profiler.end("tt_forward")
    logger.info("Forward pass completed successfully")

    # --- PCC summary (print buffered per-iteration results, then defer assertion) ---
    if need_validation:
        ref_source = "iter0_baseline" if determinism_check else "torch"
        failures = []  # list of (iter_idx, label, pcc)
        for iter_idx, iter_pcc in per_iter_pcc_buffer:
            logger.info(f"\n{'='*60}")
            if determinism_check and iter_idx == 0:
                logger.info(f"Iteration {iter_idx} (baseline — no comparison)")
                logger.info(f"{'='*60}")
                continue
            logger.info(f"Iteration {iter_idx} PCC results (ref={ref_source})")
            logger.info(f"{'Stage':<20s}  {'PCC':>10s}  {'Threshold':>10s}  {'Status':>8s}")
            logger.info(f"{'-'*60}")
            for label, pcc in iter_pcc:
                label_threshold = _threshold_for(label, determinism_check, output_threshold)
                status = "PASS" if pcc > label_threshold else ("FAIL" if pcc >= 0 else "ERROR")
                logger.info(f"{label:<20s}  {pcc:>10.6f}  {label_threshold:>10.4f}  {status:>8s}")
                if pcc <= label_threshold:
                    failures.append((iter_idx, label, pcc))
            logger.info(f"{'='*60}")

        has_pcc_failures = len(failures) > 0
        mode_label = "determinism" if determinism_check else "PCC"
        if not has_pcc_failures:
            logger.success(
                f"TtPrefillBlock {mode_label} test passed across {num_iterations} iteration(s) "
                f"(layer_type={layer_type}, gate_fallback_mode={gate_fallback_mode}, ref_source={ref_source})"
            )
        else:
            pcc_failure_msg = "; ".join(f"iter {it} {label}: {pcc:.6f}" for it, label, pcc in failures)
            logger.error(
                f"TtPrefillBlock {mode_label} test has failures "
                f"(layer_type={layer_type}, failures={len(failures)} across {num_iterations} iteration(s))"
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

    # Deferred PCC failure check (after timing report)
    if need_validation and has_pcc_failures:
        pytest.fail(f"PCC below threshold at: {pcc_failure_msg}")

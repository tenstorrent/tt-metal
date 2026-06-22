# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TtPrefillBlock — verifies composition of norm → MLA → residual → norm → FFN/MoE → residual.

Validates output shapes and PCC against torch reference.

Uses HF DeepseekV3Model layer as the reference: creates a model with random weights,
extracts those weights into our TT state_dict format, and compares forward passes.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import DynamicCache

import ttnn
from models.common.utility_functions import hf_cache_layer_kv, is_blackhole, profiler
from models.demos.deepseek_v3.demo.demo import load_prompts_from_json
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    create_balanced_chunk_order,
    reorder_tensor_chunks,
    reverse_reorder_tensor_chunks,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ABC_1K_PATH,
    PROMPT_5K_PATH,
    PROMPT_25K_PATH,
    create_hf_model,
    extract_layer_state_dict,
    get_4d_causal_mask,
    tokenize_prompt_to_isl,
)

_PROMPT_PATHS = {"abc_1k": ABC_1K_PATH, "prompt_5k": PROMPT_5K_PATH, "prompt_25k": PROMPT_25K_PATH}
from tests.ttnn.utils_for_testing import comp_pcc


@dataclass(frozen=True)
class PrefillBlockThresholds:
    dense: float = 0.996
    moe_gate_host: float = 0.996
    moe_gate_device: float = 0.992
    kvpe_kv: float = 0.999
    kvpe_pe: float = 0.999


DSV3_THRESHOLDS = PrefillBlockThresholds()
KIMI_THRESHOLDS = PrefillBlockThresholds(moe_gate_host=0.950)

# Determinism: every iteration must be bit-identical to the iter-0 baseline (strict).
DETERMINISM_PCC_THRESHOLD = 1.0


def run_model(
    variant,
    config,
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
    input_source,
    tokenizer,
    is_ci_env,
    is_ci_v2_env,
    thresholds: PrefillBlockThresholds,
    determinism_check: bool = False,
    num_iterations: int = 1,
):
    if (is_ci_env or is_ci_v2_env) and pcc_validation == False and not determinism_check:
        pytest.skip("Skip non-PCC test in CI to save time")
    # Kimi's parametrize has no `balanced` entry today (only non_balanced).
    # Applying this skip would zero out Kimi's CI coverage for this test.
    # Remove this exception once there's need to test both balanced and non_balanced for Kimi.
    if (is_ci_env or is_ci_v2_env) and not is_balanced and variant.name != "kimi_k2_6":
        pytest.skip("Skip non_balanced variant in CI — runnable locally for non_balanced-mode validation")

    # The 25k-ISL cases only fit L1 on the full 8x4 mesh. There sp_factor=8 keeps the per-chip
    # sequence at 3200 tokens, so the shared-expert down-projection matmul runs with per_core_M=2.
    # On the smaller 2x4 meshes the per-chip sequence is 12800 tokens, pushing per_core_M to 5 and
    # growing the down-matmul output circular buffer to ~2.9 MB — beyond the 1.5 MB L1 (OOM).
    if isl_total == 25 * 1024 and tuple(mesh_device.shape) != (8, 4):
        pytest.skip("25k ISL only fits L1 on the full 8x4 mesh; skipping on smaller meshes")

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

    # layer_idx=0 for dense (< NUM_DENSE_LAYERS=3), layer_idx=3 for MoE (>= 3)
    layer_idx = 0 if layer_type == "dense" else config.first_k_dense_replace

    logger.info(f"mesh_shape={mesh_shape}, sp_factor={sp_factor}, tp_factor={tp_factor}")
    logger.info(
        f"isl_total={isl_total}, isl_per_chip={isl_per_chip}, "
        f"layer_type={layer_type}, layer_idx={layer_idx}, gate_fallback_mode={gate_fallback_mode}, "
        f"input_source={input_source}"
    )

    # --- Cache setup ---
    is_dense = layer_idx < config.first_k_dense_replace
    cache_root = os.environ.get("TT_DS_PREFILL_HOST_REF_CACHE", "/tmp")
    balanced_tag = "balanced" if is_balanced else "non_balanced"
    gate_tag = gate_fallback_mode.value if gate_fallback_mode else "no_gate_fallback"
    cache_dir = Path(
        f"{cache_root}/{variant.name}_prefill_block/"
        f"{layer_type}_{sp_factor}x{tp_factor}mesh_{isl_total}isl_{balanced_tag}_{gate_tag}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    init_checker(cache_dir)
    ttnn_cache_complete = TtPrefillBlock.check_cache_complete(cache_dir, layer_idx, is_dense)
    torch_ref_cache = cache_dir / f"torch_reference_{input_source}.pt"

    ref_cache_loadable = torch_ref_cache.exists() and (pcc_validation or input_source in _PROMPT_PATHS)
    need_hf_model = not ttnn_cache_complete or (
        (pcc_validation or input_source in _PROMPT_PATHS) and not ref_cache_loadable
    )
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
        hf_model = create_hf_model(variant, config, num_layers)
        hf_sd = hf_model.state_dict()
        state_dict = extract_layer_state_dict(variant, hf_sd, layer_idx, hf_model.layers[layer_idx])
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
    elif input_source in _PROMPT_PATHS:
        profiler.start("tokenization")
        prompt_path = _PROMPT_PATHS[input_source]
        prompts = load_prompts_from_json(str(prompt_path))
        prompt_text = prompts[0] if isinstance(prompts, list) else prompts
        token_ids, attention_mask, tokens = tokenize_prompt_to_isl(
            tokenizer, max_isl=isl_total, prompt_text=prompt_text
        )
        attention_mask = get_4d_causal_mask(attention_mask, causal_only=True)
        profiler.end("tokenization")
        logger.info(
            f"Tokenized {input_source} input shape: {token_ids.shape}, first 10 tokens: {token_ids[0, :10].tolist()}"
        )
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
        attention_mask = get_4d_causal_mask(torch.ones(1, isl_total), causal_only=True).to(torch.bfloat16)
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
            ref_kvpe = hf_cache_layer_kv(ref_cache, layer_idx)[0]
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
            model_cfg=variant.model_config,
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
        model_cfg=variant.model_config,
        state_dict=state_dict,
        layer_idx=layer_idx,
        seq_len=isl_total,
        dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        weight_cache_path=cache_dir,
        is_balanced=is_balanced,
    )
    if gate_fallback_mode is not None:
        block_kwargs["gate_fallback_mode"] = gate_fallback_mode

    block = TtPrefillBlock(**block_kwargs)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_block_creation")

    # Shard input to device: [1, 1, isl_total, emb_dim] → [1, 1, isl_per_chip, emb_dim/tp]
    tt_input_4d = torch_input.unsqueeze(0)  # [1, 1, isl_total, emb_dim]
    if is_balanced == True:
        chunk_order = create_balanced_chunk_order(sp_factor)
        tt_input_4d = reorder_tensor_chunks(tt_input_4d, chunk_order, seq_dim=2)

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

    # --- Determinism check (isolated from the pcc path below) ---
    # Run num_iterations forwards on identical input; every iteration's output must be
    # bit-identical (PCC == 1.0) to the iter-0 baseline.
    if determinism_check:
        if pcc_validation:
            pytest.skip("determinism_check and pcc_validation are mutually exclusive — pick one")
        if num_iterations < 2:
            pytest.skip("determinism_check requires num_iterations >= 2 (iter 0 is the baseline)")
        threshold = DETERMINISM_PCC_THRESHOLD
        logger.info(f"Determinism check (threshold={threshold}, baseline=iter0)")
        profiler.start("tt_forward")
        baseline = None
        det_failures = []
        for i in range(num_iterations):
            tt_output, _ = block(tt_input, rope_tensors, tt_kvpe_cache, return_kv_cache=False)
            ttnn.synchronize_device(mesh_device)
            out_host = ttnn.to_torch(
                tt_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape),
            ).to(torch.bfloat16)
            if i == 0:
                baseline = out_host.clone()
                logger.info("Determinism: captured iter0 baseline")
                continue
            try:
                _, pcc = comp_pcc(baseline.float(), out_host.float())
            except Exception as e:
                logger.error(f"output PCC comparison failed: {e}")
                pcc = -1.0
            status = "PASS" if pcc >= threshold else ("FAIL" if pcc >= 0 else "ERROR")
            logger.info(f"iter {i} output  PCC = {pcc:.6f}  {status}")
            if pcc < threshold:
                det_failures.append((i, pcc))
        profiler.end("tt_forward")
        profiler.end("total_test_time")
        if det_failures:
            msg = "; ".join(f"iter {it}: {pcc:.6f}" for it, pcc in det_failures)
            pytest.fail(f"Determinism PCC below {threshold}: {msg}")
        logger.success(
            f"TtPrefillBlock determinism test passed across {num_iterations} iteration(s) "
            f"(layer_type={layer_type}, gate_fallback_mode={gate_fallback_mode})"
        )
        return

    profiler.start("tt_forward")
    logger.info("Running TtPrefillBlock forward...")
    tt_output, tt_kvpe = block(tt_input, rope_tensors, tt_kvpe_cache, return_kv_cache=pcc_validation)
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
        if is_balanced:
            tt_output_host = reverse_reorder_tensor_chunks(tt_output_host, chunk_order, seq_dim=-2)
        tt_output_host = tt_output_host.squeeze(0)

        if layer_type == "dense":
            pcc_threshold = thresholds.dense
        else:
            if gate_fallback_mode == GateComputeMode.DEVICE:
                pcc_threshold = thresholds.moe_gate_device
            else:
                pcc_threshold = thresholds.moe_gate_host

        _, pcc = comp_pcc(torch_output.float(), tt_output_host.float())
        profiler.end("pcc_validation")
        logger.info(f"PCC: {pcc:.6f} (threshold: {pcc_threshold})")
        assert pcc > pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"

        # --- KVPE cache validation ---
        if ref_kvpe is not None and tt_kvpe is not None:
            kv_lora_rank = config.kv_lora_rank
            if is_balanced:
                tt_kvpe = reverse_reorder_tensor_chunks(tt_kvpe, chunk_order, seq_dim=2)
            _, kv_pcc = comp_pcc(ref_kvpe[:, :, :, :kv_lora_rank].float(), tt_kvpe[:, :, :, :kv_lora_rank].float())
            _, pe_pcc = comp_pcc(ref_kvpe[:, :, :, kv_lora_rank:].float(), tt_kvpe[:, :, :, kv_lora_rank:].float())
            logger.info(f"KVPE cache KV part PCC: {kv_pcc:.6f} (threshold: {thresholds.kvpe_kv})")
            logger.info(f"KVPE cache PE part PCC: {pe_pcc:.6f} (threshold: {thresholds.kvpe_pe})")
            assert kv_pcc > thresholds.kvpe_kv, f"KVPE KV PCC {kv_pcc:.6f} below threshold {thresholds.kvpe_kv}"
            assert pe_pcc > thresholds.kvpe_pe, f"KVPE PE PCC {pe_pcc:.6f} below threshold {thresholds.kvpe_pe}"

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


@pytest.mark.parametrize(
    "input_source, pcc_validation, isl_total, dispatch_buffer_capacity_factor",
    [
        ("random", False, 1024, 8),
        ("prompt_25k", False, 25 * 1024, 8),
        ("abc_1k", True, 1024, 8),
    ],
    ids=["smoke-random", "perf-prompt_25k", "pcc-abc_1k"],
)
@pytest.mark.parametrize(
    "layer_type, gate_fallback_mode",
    [("dense", None), ("moe", GateComputeMode.DEVICE)],
    ids=["dense", "moe-gate_device"],
)
@pytest.mark.parametrize("is_balanced", [True, False], ids=["balanced", "non_balanced"])
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
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        ),
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.parametrize("determinism_check", [False, True], ids=["no_determinism", "with_determinism"])
@pytest.mark.parametrize("num_iterations", [1, 2, 5, 25, 2000], ids=["iter1", "iter2", "iter5", "iter25", "iter2000"])
@pytest.mark.timeout(600)
def test_ds_prefill_block(
    variant,
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
    input_source,
    tokenizer,
    is_ci_env,
    is_ci_v2_env,
    determinism_check,
    num_iterations,
):
    run_model(
        variant,
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
        input_source,
        tokenizer,
        is_ci_env,
        is_ci_v2_env,
        determinism_check=determinism_check,
        num_iterations=num_iterations,
        thresholds=DSV3_THRESHOLDS,
    )


@pytest.mark.parametrize(
    "input_source, pcc_validation, isl_total, dispatch_buffer_capacity_factor",
    [
        ("random", False, 1024, 8),
        ("random", False, 5 * 1024, 8),
        ("random", False, 25 * 1024, 8),
        ("abc_1k", True, 1024, 8),
        ("prompt_5k", True, 5 * 1024, 8),
        ("prompt_25k", True, 25 * 1024, 8),
    ],
    ids=["smoke-random", "perf-random-5k", "perf-random-25k", "pcc-abc_1k", "pcc-prompt_5k", "pcc-prompt_25k"],
)
@pytest.mark.parametrize(
    "layer_type, gate_fallback_mode",
    [("dense", None), ("moe", GateComputeMode.DEVICE_FP32)],
    ids=["dense", "moe_gate_device"],
)
@pytest.mark.parametrize("is_balanced", [False], ids=["non_balanced"])
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
@pytest.mark.parametrize("determinism_check", [False, True], ids=["no_determinism", "with_determinism"])
@pytest.mark.parametrize("num_iterations", [1, 2, 5, 25, 2000], ids=["iter1", "iter2", "iter5", "iter25", "iter2000"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.timeout(900)
def test_kimi_prefill_block(
    variant,
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
    input_source,
    tokenizer,
    is_ci_env,
    is_ci_v2_env,
    determinism_check,
    num_iterations,
):
    run_model(
        variant,
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
        input_source,
        tokenizer,
        is_ci_env,
        is_ci_v2_env,
        determinism_check=determinism_check,
        num_iterations=num_iterations,
        thresholds=KIMI_THRESHOLDS,
    )

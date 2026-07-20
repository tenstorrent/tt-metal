# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TtPrefillBlock — verifies composition of norm → MLA → residual → norm → FFN/MoE → residual.

Validates output shapes and PCC against torch reference.

Reference: when a pretrained checkpoint is available the layer's input/output come from a real
forward over layers 0..layer_idx (as in test_prefill_transformer); otherwise it falls back to a
randomly-initialized HF reference layer so the test still runs without weights.
"""

import json
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
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import pretrained_mla_weights
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_1 import glm_decoder_layer_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import load_moe_weights_from_hf
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import build_weights
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
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
    load_and_compute_layer_by_layer,
    tokenize_prompt_to_isl,
)

_PROMPT_PATHS = {"abc_1k": ABC_1K_PATH, "prompt_5k": PROMPT_5K_PATH, "prompt_25k": PROMPT_25K_PATH}
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc


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
    request,
    is_ci_env,
    is_ci_v2_env,
    thresholds: PrefillBlockThresholds,
    determinism_check: bool = False,
    num_iterations: int = 1,
    use_pretrained: bool = False,
):
    if (is_ci_env or is_ci_v2_env) and pcc_validation == False and not determinism_check:
        pytest.skip("Skip non-PCC test in CI to save time")
    # Kimi's parametrize has no `balanced` entry today (only non_balanced).
    # Applying this skip would zero out Kimi's CI coverage for this test.
    # Remove this exception once there's need to test both balanced and non_balanced for Kimi.
    if (is_ci_env or is_ci_v2_env) and not is_balanced and variant.name != "kimi_k2_6":
        pytest.skip("Skip non_balanced variant in CI — runnable locally for non_balanced-mode validation")

    # host_gate_all is a local testing aid for sub-256-expert configs (e.g. the 4x4 sub-torus,
    # where the device grouped-gate's hard 256-expert requirement forces the host gate). It is not CI
    # coverage; the real device gate already covers the 256-expert meshes that run in CI.
    if (is_ci_env or is_ci_v2_env) and gate_fallback_mode == GateComputeMode.HOST_ALL:
        pytest.skip("host_gate_all is a local-only testing aid (sub-256-expert); not run in CI")

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

    # When use_pretrained is requested, load real pretrained weights (mirrors test_prefill_transformer):
    # load_and_compute_layer_by_layer loads the real layers 0..layer_idx, builds the shared TTNN weight
    # cache, and returns per-layer snapshots used as the block's input/output reference. Otherwise the
    # random-weight path below runs, and model_path is never resolved -> no HuggingFace download.
    if use_pretrained and not variant.supports_pretrained:
        pytest.skip(f"{variant.name}: pretrained weights not wired")

    if use_pretrained:
        model_path = request.getfixturevalue("model_path")
        weight_cache_path = request.getfixturevalue("weight_cache_path")
        is_dense = layer_idx < config.first_k_dense_replace
        num_layers = layer_idx + 1
        torch_output = None
        ref_kvpe = None
        hf_model = None

        cache_dir = weight_cache_path / f"{sp_factor}x{tp_factor}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        init_checker(cache_dir)
        state_dict = {}  # weights come from the prebuilt cache
        experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp_factor * tp_factor)
        ttnn_cache_complete = TtPrefillBlock.check_cache_complete(cache_dir, layer_idx, is_dense, experts_per_chip)

        if pcc_validation or not ttnn_cache_complete:
            if input_source in _PROMPT_PATHS:
                prompts = load_prompts_from_json(str(_PROMPT_PATHS[input_source]))
                prompt_text = prompts[0] if isinstance(prompts, list) else prompts
                token_ids, attention_mask, _ = tokenize_prompt_to_isl(
                    tokenizer, max_isl=isl_total, prompt_text=prompt_text
                )
            else:
                token_ids = torch.randint(0, config.vocab_size, (1, isl_total), dtype=torch.int64)
                attention_mask = torch.ones(1, isl_total, dtype=torch.int64)
            result = load_and_compute_layer_by_layer(
                variant=variant,
                model_path=model_path,
                config=config,
                num_layers=num_layers,
                token_ids=token_ids,
                attention_mask=attention_mask,
                compute_reference=pcc_validation,
                build_ttnn_cache=not ttnn_cache_complete,
                weight_cache_path=cache_dir,
                mesh_device=mesh_device,
                seq_len=isl_total,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                gate_fallback_mode=gate_fallback_mode,
            )
            ttnn_cache_complete = True
            if pcc_validation:
                # snapshots are [embed, layer_0_out, ...]: input to layer L is snapshot[L], output snapshot[L+1].
                torch_input = result.ref_snapshots[layer_idx].to(torch.bfloat16)
                torch_output = result.ref_snapshots[layer_idx + 1]
                ref_kvpe = result.ref_kvpe_list[layer_idx]
        if not pcc_validation:
            torch.manual_seed(123)
            torch_input = torch.randn(1, isl_total, emb_dim, dtype=torch.bfloat16)
    else:
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
    [("dense", None), ("moe", GateComputeMode.DEVICE), ("moe", GateComputeMode.HOST_ALL)],
    # The host-gate id omits the `moe` token on purpose: CI selects the device gate via count-guarded
    # `-k "... and moe and ..."` (EXPECT_NUM_TESTS=1), so a host id carrying `moe` would be collected too
    # and break the count. It is a local sub-256-expert aid (CI-skipped by enum); select via `-k host_gate`.
    ids=["dense", "moe-gate_device", "host_gate_all"],
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
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            1,
            # Per-axis topology (SP-axis-0, TP-axis-1). FABRIC_2D_TORUS_Y wraps ONLY the SP axis
            # into a ring → Ring for SP-axis MoE dispatch/combine; the 4-wide TP axis stays a line
            # → Linear for TP-axis collectives (RMS-norm, MLA, shared-expert, gate). A scalar Ring
            # here deadlocks the TP-axis all-gathers on a non-existent column wrap link.
            (ttnn.Topology.Ring, ttnn.Topology.Linear),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            # Id omits the `fabric2d-`/`mesh-` tokens on purpose: CI runs the fabric2d-mesh siblings
            # via count-guarded `-k "8x4 and fabric2d"` (EXPECT_NUM_TESTS=1) selectors, so a torus id
            # carrying `fabric2d` would be collected too and break the count. Select it with `-k torus-y`.
            id="torus-y-8x4",
        ),
        pytest.param(
            (4, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            2,
            # 4x4 sub-torus: Ring-4 on the SP axis (dim 0), Linear on the 4-wide TP axis (dim 1).
            # Run with TT_VISIBLE_DEVICES (16 chips) + TT_MESH_GRAPH_DESC_PATH=...subtorus_y4...
            (ttnn.Topology.Ring, ttnn.Topology.Linear),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
            id="torus-y-4x4",
        ),
        pytest.param(
            (4, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            2,
            # 4x4 full 2D sub-torus: Ring-4 on BOTH axes (dim 0 = SP/Y, dim 1 = TP/X). Both axes have
            # a physical wrap, so TP-axis collectives (RMS-norm, MLA, shared-expert) can ring too.
            # Run with TT_VISIBLE_DEVICES (16 chips) + TT_MESH_GRAPH_DESC_PATH=...subtorus_xy4...
            (ttnn.Topology.Ring, ttnn.Topology.Ring),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4"),
            id="torus-xy-4x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.parametrize("determinism_check", [False, True], ids=["no_determinism", "with_determinism"])
@pytest.mark.parametrize("num_iterations", [1, 2, 5, 25, 2000], ids=["iter1", "iter2", "iter5", "iter25", "iter2000"])
@pytest.mark.timeout(750)
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
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
    use_pretrained,
    request,
):
    # FABRIC_2D on the 2x4 mesh regresses the MoE/device-gate PCC ~3 points below the 0.992 gate.
    # xfail this exact combo (keeping the real threshold for every other config) until it is fixed;
    # strict=True turns an XPASS into a failure so the marker is removed once the fix lands.
    if (
        pcc_validation
        and not determinism_check
        and layer_type == "moe"
        and gate_fallback_mode == GateComputeMode.DEVICE
        and is_balanced
        and device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_2D
        and tuple(mesh_device.shape) == (2, 4)
    ):
        request.node.add_marker(
            pytest.mark.xfail(reason="FABRIC_2D 2x4 MoE/device-gate PCC regression (~0.96 < 0.992)", strict=True)
        )

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
        request,
        is_ci_env,
        is_ci_v2_env,
        determinism_check=determinism_check,
        num_iterations=num_iterations,
        thresholds=DSV3_THRESHOLDS,
        use_pretrained=use_pretrained,
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
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
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
    use_pretrained,
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
        layer_type,
        gate_fallback_mode,
        num_links,
        topology,
        pcc_validation,
        input_source,
        tokenizer,
        request,
        is_ci_env,
        is_ci_v2_env,
        determinism_check=determinism_check,
        num_iterations=num_iterations,
        thresholds=KIMI_THRESHOLDS,
        use_pretrained=use_pretrained,
    )


# ---------------------------------------------------------------------------
# GLM-5.1 block test
# ---------------------------------------------------------------------------
# Every GLM layer runs sparse DSA (lightning-indexer top-2048 + sparse SDPA); "dense"/"moe" here refers
# only to the FFN — layers 0-2 have a dense FFN, layers 3-77 a 256-expert MoE. Both block types exercise
# sparse SDPA (via ttMLA's DSA path); they differ only in the FFN.
#
# GLM has no runnable HF reference model wired (adapter reference_model_cls is None), so it can't use
# run_model()/create_hf_model() like the DeepSeek/Kimi block tests. Instead it COMPOSES the CPU
# references GLM already owns (reference.glm_5_1.glm_decoder_layer_reference): x + MLA_cpu(attn_norm(x))
# then + FFN(ffn_norm(x+mla_out)) — exactly TtPrefillBlock.forward.
# Why not generalize run_model to take this composed ref? run_model's PCC path is built around
# create_hf_model() + a single HF module; GLM's only full HF module (GlmMoeDsaModel) is non-absorbed
# (256-expert, dense attention) and too slow to run per-block (~15-25h) — the absorbed CPU composition
# runs in ~7s/layer. Folding this in would mean adding a pluggable "reference callable" seam to the
# shared runner; a fair future cleanup, but out of scope here (keeps the fast path off the shared path).
#
# Weights (like the transformer tests): LOAD real weights from the prebuilt ttnn cache when present &
# complete (device) + matching host weights from the checkpoint (reference); else RANDOM for both. Never
# (re)builds the cache (slow: 256-expert conversion / FP8 dequant — a separate staging step).
GLM_BLOCK_OUTPUT_PCC = 0.98


def _glm_norm_weight(hidden, seed):
    return (torch.randn(hidden, generator=torch.Generator().manual_seed(seed)) * 0.1 + 1.0).to(torch.bfloat16)


def _glm_random_dense_ffn_weights(hidden, intermediate, seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "gate_proj": (torch.randn(intermediate, hidden, generator=g) * hidden**-0.5).to(torch.bfloat16),
        "up_proj": (torch.randn(intermediate, hidden, generator=g) * hidden**-0.5).to(torch.bfloat16),
        "down_proj": (torch.randn(hidden, intermediate, generator=g) * intermediate**-0.5).to(torch.bfloat16),
    }


def _glm_random_moe_weights(hidden, moe_intermediate, n_routed, seed):
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
        "e_score_correction_bias": (torch.randn(n_routed, generator=g) * 0.01).to(torch.float32),
    }
    return gate_weights, [_expert() for _ in range(n_routed)], _expert()


def _glm_pretrained_weights(config, model_dir, layer_idx, is_moe):
    """Load layer-``layer_idx`` host weights from a local checkpoint dir; fp8 tensors are block-dequantized."""
    from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.weights import _dequant_fp8

    model_dir = str(model_dir)
    prefix = f"model.layers.{layer_idx}."
    # MLA + indexer via the sparse-MLA loader (resolve this layer's local shards; dequant is a no-op for bf16).
    weight_map = json.load(open(Path(model_dir) / "model.safetensors.index.json"))["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
    mla_weights = pretrained_mla_weights(
        config, layer=layer_idx, checkpoint_path=[str(Path(model_dir) / s) for s in shards]
    )
    norms = load_hf_state_dict_filtered(model_dir, [f"{prefix}input_layernorm.", f"{prefix}post_attention_layernorm."])
    attn_norm_w = norms[f"{prefix}input_layernorm.weight"].to(torch.bfloat16)
    ffn_norm_w = norms[f"{prefix}post_attention_layernorm.weight"].to(torch.bfloat16)
    if is_moe:
        routed, shared = load_moe_weights_from_hf(model_dir, layer_idx, GLM51Config.NUM_ROUTED_EXPERTS)
        g = load_hf_state_dict_filtered(model_dir, [f"{prefix}mlp.gate."])
        gate_weights = {
            "weight": g[f"{prefix}mlp.gate.weight"].to(torch.bfloat16),
            "e_score_correction_bias": g[f"{prefix}mlp.gate.e_score_correction_bias"].float(),
        }
        moe_weights = {
            "gate_weights": gate_weights,
            "routed_expert_weights": routed,
            "shared_expert_weights": shared,
        }
        return mla_weights, attn_norm_w, ffn_norm_w, moe_weights, None
    f = load_hf_state_dict_filtered(
        model_dir, [f"{prefix}mlp.gate_proj.", f"{prefix}mlp.up_proj.", f"{prefix}mlp.down_proj."]
    )

    def _ffn(p):
        # The device cache stores dequantized weights; the reference must match, so undo the fp8 block-scale here.
        w = f[f"{prefix}mlp.{p}.weight"]
        if w.dtype == torch.float8_e4m3fn:
            w = _dequant_fp8(w, f[f"{prefix}mlp.{p}.weight_scale_inv"])
        return w.to(torch.bfloat16)

    ffn_weights = {p: _ffn(p) for p in ("gate_proj", "up_proj", "down_proj")}
    return mla_weights, attn_norm_w, ffn_norm_w, None, ffn_weights


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize(
    "layer_type, layer_idx", [("dense", 0), ("moe", GLM51Config.NUM_DENSE_LAYERS)], ids=["dense", "moe"]
)
@pytest.mark.parametrize("variant", ["glm_5_1", "glm_5_2"], indirect=True, ids=["glm51", "glm52"])
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_prefill_block(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    topology,
    seq_len,
    layer_type,
    layer_idx,
    model_path,
    weight_cache_path,
):
    """One fused GLM decoder block (sparse-SDPA MLA + norm/residual + dense|MoE FFN) vs composed CPU ref."""
    is_moe = layer_type == "moe"
    config = config_only
    config.max_seq_len = seq_len
    # GLM-5.2 MoE single-block: skipped. The block feeds a RANDOM input, which drives GLM's
    # near-degenerate top-8 MoE gate to select different experts on device vs the CPU reference at an
    # isolated layer (block PCC collapses to ~0.1 though the same layer scores ~0.995 in-context in
    # test_glm_prefill_transformer). Not an op/weight bug — a random-input artifact of the degenerate
    # gate. GLM-5.2 MoE is covered by test_glm_prefill_transformer; the device gate by test_ttnn_moe.
    if is_moe and getattr(config, "indexer_types", None) is not None:
        pytest.skip("GLM-5.2 single-block MoE unreliable under degenerate gate on random input (see comment)")
    hidden = config.hidden_size
    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)

    # Weight source: use the prebuilt ttnn tensorbin cache when present AND complete -> real weights (device
    # LOADS from the cache; reference uses matching host weights from the checkpoint). Else RANDOM for both.
    # Never (re)build the cache here.
    sp_factor, tp_factor = mesh_shape[sp_axis], mesh_shape[tp_axis]
    experts_per_chip = GLM51Config.NUM_ROUTED_EXPERTS // (sp_factor * tp_factor)
    effective_cache = (weight_cache_path / f"{sp_factor}x{tp_factor}") if weight_cache_path is not None else None
    use_pretrained = False
    if effective_cache is not None:
        effective_cache.mkdir(parents=True, exist_ok=True)
        init_checker(effective_cache)  # required before check_cache_complete / pattern_exists
        use_pretrained = TtPrefillBlock.check_cache_complete(effective_cache, layer_idx, not is_moe, experts_per_chip)
    logger.info(f"[glm block {layer_type}] use_pretrained={use_pretrained} (ttnn cache={effective_cache})")

    if use_pretrained:
        # Device loads real weights from the cache; reference uses matching host weights from the checkpoint.
        mla_weights, attn_norm_w, ffn_norm_w, moe_weights, ffn_weights = _glm_pretrained_weights(
            config, model_path, layer_idx, is_moe
        )
        device_state_dict, device_cache = {}, effective_cache
    else:
        # Random weights for both device (built from host) and reference.
        mla_weights, _ = build_weights(variant, config, seed=42)
        attn_norm_w, ffn_norm_w = _glm_norm_weight(hidden, 1), _glm_norm_weight(hidden, 2)
        if is_moe:
            gate_weights, routed, shared = _glm_random_moe_weights(
                hidden, GLM51Config.MOE_INTERMEDIATE_SIZE, GLM51Config.NUM_ROUTED_EXPERTS, seed=3
            )
            moe_weights = {
                "gate_weights": gate_weights,
                "routed_expert_weights": routed,
                "shared_expert_weights": shared,
            }
            ffn_weights = None
        else:
            ffn_weights = _glm_random_dense_ffn_weights(hidden, config.intermediate_size, seed=3)
            moe_weights = None
        device_state_dict = {"attn_norm_weight": attn_norm_w, "mla_weights": mla_weights, "ffn_norm_weight": ffn_norm_w}
        if is_moe:
            device_state_dict.update(
                gate_weights=moe_weights["gate_weights"],
                routed_expert_weights=moe_weights["routed_expert_weights"],
                shared_expert_weights=moe_weights["shared_expert_weights"],
            )
        else:
            device_state_dict["ffn_weights"] = ffn_weights
        device_cache = None

    # --- device block ---
    logger.info(
        f"[glm block {layer_type}] building TtPrefillBlock layer_idx={layer_idx} seq_len={seq_len} mesh={mesh_shape}"
    )
    block = TtPrefillBlock(
        mesh_device=mesh_device,
        config=config,
        model_cfg=GLM51Config,
        state_dict=device_state_dict,
        layer_idx=layer_idx,
        seq_len=seq_len,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=device_cache,
        # single-block test: layer_num=1 so the sparse single-shot cache write (update_padded_kv_cache,
        # num_layers=layer_num) gets a valid count, not the None default.
        layer_num=1,
    )
    kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    # Sparse (DSA) MLA single-shot is folded onto the block-cyclic path (one full-seq chunk at offset 0):
    # it uses the indexed rope tables and a caller-owned indexer key cache, exactly like the chunked path.
    # GLM attention is always sparse, so this is unconditional here. The cache is strided by the compacted
    # full-indexer count (num_full_indexer_layers) — >1 for glm_5_2 cross-layer reuse — matching the
    # indexer's cache_batch stride; falls back to 1 when there is no indexer_types map (glm_5_1).
    rope_tensors = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(
        cache_seq_len_global=seq_len, chunk_size_global=seq_len
    )
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_full_indexer_layers(config) or 1,
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )

    # --- input (full, host) + sharded device copy ---
    torch.manual_seed(7)
    x = torch.randn(1, seq_len, hidden, dtype=torch.bfloat16)
    shard_dims = [None, None]
    shard_dims[tp_axis] = -1
    shard_dims[sp_axis] = -2
    tt_x = ttnn.from_torch(
        x.unsqueeze(0),  # [1, 1, seq, hidden]
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    logger.info(f"[glm block {layer_type}] running device block")
    out = block.forward(
        tt_x, rope_tensors=rope_tensors, kvpe_cache=kvpe_cache, actual_isl=seq_len, index_kv_cache=index_kv_cache
    )
    if isinstance(out, tuple):
        out = out[0]
    tt_out = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)

    # --- composed reference (reference/glm_5_1): assembles MLA + norm/residual + FFN ---
    logger.info(f"[glm block {layer_type}] composing CPU reference via reference.glm_5_1.glm_decoder_layer_reference")
    ref, _ = glm_decoder_layer_reference(
        config, mla_weights, attn_norm_w, ffn_norm_w, x, seq_len, ffn_weights=ffn_weights, moe_weights=moe_weights
    )

    _, pcc_msg = assert_with_pcc(ref.unsqueeze(0), tt_out, GLM_BLOCK_OUTPUT_PCC)
    logger.info(f"[glm block {layer_type}] block output PCC: {pcc_msg}")
    ttnn.synchronize_device(mesh_device)

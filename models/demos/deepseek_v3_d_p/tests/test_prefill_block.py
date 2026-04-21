# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TtPrefillBlock — verifies composition of norm → MLA → residual → norm → FFN/MoE → residual.

Validates output shapes and PCC against torch reference.

Uses HF DeepseekV3Model layer as the reference: creates a model with random weights,
extracts those weights into our TT state_dict format, and compares forward passes.
"""

import json

import pytest
import torch
from loguru import logger
from transformers import DynamicCache

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3.demo.demo import load_prompts_from_json
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    create_balanced_chunk_order,
    reorder_tensor_chunks,
    reverse_reorder_tensor_chunks,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ABC_1K_PATH,
    create_hf_model,
    download_infinitebench_subset,
    extract_layer_state_dict,
    get_4d_causal_mask,
    tokenize_prompt_to_isl,
)
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD_DENSE = 0.996
PCC_THRESHOLD_MOE_GATE_HOST = 0.996
PCC_THRESHOLD_MOE_GATE_DEVICE = 0.992
PCC_THRESHOLD_KVPE = 0.999


@pytest.mark.parametrize(
    "is_balanced",
    [False, True],
    ids=["unbalanced", "balanced"],
)
@pytest.mark.parametrize(
    "input_source, pcc_validation, isl_total",
    [
        ("random", False, 1024),
        ("abc_1k", True, 1024),
        ("random", False, 25600),
        ("infinitebench_longbook", True, 25600),
    ],
    ids=["smoke-random", "pcc-abc_1k", "smoke-random-25k", "pcc-infinitebench-25k"],
)
@pytest.mark.parametrize(
    "layer_type, gate_fallback_mode",
    [
        ("dense", None),
        ("moe", GateComputeMode.DEVICE),
    ],
    ids=["dense", "moe-gate_device"],
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
@pytest.mark.timeout(600)
def test_prefill_block(
    config_only,
    mesh_device,
    device_params,
    isl_total,
    layer_type,
    gate_fallback_mode,
    num_links,
    topology,
    pcc_validation,
    input_source,
    is_balanced,
    request,
):
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
        f"input_source={input_source}, is_balanced={is_balanced}"
    )

    # --- Build HF reference model and extract weights ---
    profiler.start("weights_creation")
    torch.manual_seed(42)
    num_layers = layer_idx + 1
    hf_model = create_hf_model(config, num_layers)
    hf_sd = hf_model.state_dict()
    state_dict = extract_layer_state_dict(hf_sd, layer_idx, hf_model.layers[layer_idx])
    profiler.end("weights_creation")

    # --- Create input ---
    if input_source in ("abc_1k", "infinitebench_longbook"):
        profiler.start("tokenization")
        tok = request.getfixturevalue("tokenizer")
        if input_source == "abc_1k":
            prompts = load_prompts_from_json(str(ABC_1K_PATH))
            prompt_text = prompts[0] if isinstance(prompts, list) else prompts
        else:  # infinitebench_longbook
            cached_path = download_infinitebench_subset("longbook_qa_eng")
            with open(cached_path) as f:
                prompt_text = json.load(f)["prompt"]
        token_ids, attention_mask, tokens = tokenize_prompt_to_isl(tok, max_isl=isl_total, prompt_text=prompt_text)
        attention_mask = get_4d_causal_mask(attention_mask, ignore_padding=True)
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

    # --- Torch reference (only when pcc_validation is enabled) ---
    torch_output = None
    ref_kvpe = None
    if pcc_validation:
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

    # --- TT block ---
    profiler.start("tt_block_creation")
    block_kwargs = dict(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        layer_idx=layer_idx,
        seq_len=isl_total,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=is_balanced,
    )
    if gate_fallback_mode is not None:
        block_kwargs["gate_fallback_mode"] = gate_fallback_mode

    block = TtPrefillBlock(**block_kwargs)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_block_creation")

    # Shard input to device: [1, 1, isl_total, emb_dim] → [1, 1, isl_per_chip, emb_dim/tp]
    tt_input_4d = torch_input.unsqueeze(0)  # [1, 1, isl_total, emb_dim]
    if is_balanced:
        # Reorder into zigzag layout so each SP chip receives its balanced chunk pair.
        # create_balanced_chunk_order produces e.g. [0,7,1,6,2,5,3,4] for sp_factor=4.
        # After this reorder, a plain SP split along seq gives chip k chunks [k, 2*sp-1-k].
        balanced_chunk_order = create_balanced_chunk_order(sp_factor)
        tt_input_4d = reorder_tensor_chunks(tt_input_4d, balanced_chunk_order, seq_dim=2)
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
        # Balanced output is in zigzag order; reverse to natural token order before comparing.
        if is_balanced:
            tt_output_host = reverse_reorder_tensor_chunks(tt_output_host, balanced_chunk_order, seq_dim=2)
        # Remove leading batch dim: [1, 1, isl_total, emb_dim] → [1, isl_total, emb_dim]
        tt_output_host = tt_output_host.squeeze(0)

        if layer_type == "dense":
            pcc_threshold = PCC_THRESHOLD_DENSE
        else:
            if gate_fallback_mode == GateComputeMode.DEVICE:
                pcc_threshold = PCC_THRESHOLD_MOE_GATE_DEVICE
            else:
                pcc_threshold = PCC_THRESHOLD_MOE_GATE_HOST

        _, pcc = comp_pcc(torch_output.float(), tt_output_host.float())
        profiler.end("pcc_validation")
        logger.info(f"PCC: {pcc:.6f} (threshold: {pcc_threshold})")
        assert pcc > pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"

        # --- KVPE cache validation ---
        if ref_kvpe is not None and tt_kvpe is not None:
            kv_lora_rank = config.kv_lora_rank
            # KVPE cache is written in balanced order; reverse to natural order before comparing.
            tt_kvpe_ordered = (
                reverse_reorder_tensor_chunks(tt_kvpe, balanced_chunk_order, seq_dim=2) if is_balanced else tt_kvpe
            )
            _, kv_pcc = comp_pcc(
                ref_kvpe[:, :, :, :kv_lora_rank].float(), tt_kvpe_ordered[:, :, :, :kv_lora_rank].float()
            )
            _, pe_pcc = comp_pcc(
                ref_kvpe[:, :, :, kv_lora_rank:].float(), tt_kvpe_ordered[:, :, :, kv_lora_rank:].float()
            )
            logger.info(f"KVPE cache KV part PCC: {kv_pcc:.6f} (threshold: {PCC_THRESHOLD_KVPE})")
            logger.info(f"KVPE cache PE part PCC: {pe_pcc:.6f} (threshold: {PCC_THRESHOLD_KVPE})")
            assert kv_pcc > PCC_THRESHOLD_KVPE, f"KVPE KV PCC {kv_pcc:.6f} below threshold {PCC_THRESHOLD_KVPE}"
            assert pe_pcc > PCC_THRESHOLD_KVPE, f"KVPE PE PCC {pe_pcc:.6f} below threshold {PCC_THRESHOLD_KVPE}"

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

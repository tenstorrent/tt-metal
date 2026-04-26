# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo Prefill PCC test with ring_distributed_sdpa enabled.

Tests that enabling ring SDPA (like Llama 70B) produces correct PCC
at ISL=8192 compared to HuggingFace reference.

Run:
    export HF_MODEL=~/models/OLMo-3.1-32B-Think
    pytest models/demos/olmo_galaxy/tests/test_olmo_ring_sdpa_prefill_pcc.py -xvs --timeout=300
"""

import math
import os

import pytest
import torch
from loguru import logger
from transformers import GPT2Tokenizer

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.olmo_galaxy.tt.llama_common import PagedAttentionConfig, gather_cos_sin, precompute_freqs_yarn
from models.demos.olmo_galaxy.tt.llama_model import TtTransformer
from models.demos.olmo_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.tt_transformers.tt.common import copy_host_to_device


def load_hf_olmo3_1layer(hf_model_path):
    """Load HF OLMo3 model with only 1 layer for reference."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    config.num_hidden_layers = 1
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    return hf_model


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 165136000,
            "fabric_config": True,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("use_ring_sdpa", [False, True], ids=["standard_sdpa", "ring_sdpa"])
@pytest.mark.parametrize("seq_len", [128, 8192])
def test_olmo_prefill_ring_sdpa_pcc(mesh_device, reset_seeds, ensure_gc, use_ring_sdpa, seq_len):
    """Compare TTNN prefill with ring SDPA vs standard SDPA against HuggingFace reference."""
    hf_model_path = os.environ.get("HF_MODEL")
    if not hf_model_path:
        pytest.skip("HF_MODEL not set")

    if use_ring_sdpa and seq_len <= 4096:
        pytest.skip("Ring SDPA only for ISL > 4096")

    n_layers = 1
    dtype = ttnn.bfloat8_b

    # Load HF reference
    logger.info("Loading HF OLMo3 (1 layer)...")
    hf_model = load_hf_olmo3_1layer(hf_model_path)

    # Load TTNN model
    logger.info("Loading TTNN model (1 layer)...")
    model_args = TtOlmoModelArgs(mesh_device, max_batch_size=32, max_seq_len=128 * 1024)
    model_args.n_layers = n_layers
    state_dict = model_args.load_state_dict()

    paged_attention_config = PagedAttentionConfig(block_size=64, max_num_blocks=4096)
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        model_args.batch_size_per_device_group,
        paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
    )

    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
        decode_mode_only=False,
    )

    # Patch ring SDPA on attention layer if requested
    if use_ring_sdpa:
        logger.info("*** ENABLING ring_distributed_sdpa for OLMo ***")
        # The attention layer checks ring_distributed_sdpa at runtime based on seq_len
        # We need to remove the is_olmo override that forces it to False
        # Monkey-patch the forward_prefill to set ring_distributed_sdpa = True
        for layer in tt_model.layers:
            layer.attention._force_ring_sdpa = True

    # Prepare input
    tokenizer = model_args.tokenizer or GPT2Tokenizer.from_pretrained(model_args.TOKENIZER_PATH)

    # Generate enough tokens to fill seq_len
    prompt = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10 + 1)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)[:seq_len]
    actual_seq_len = len(input_ids)

    # Pad to nearest power of 2 or supported seqlen
    from models.demos.olmo_galaxy.demo.demo_olmo_decode import get_padded_prefill_len

    padded_len = get_padded_prefill_len(actual_seq_len)
    input_ids_padded = input_ids + [tokenizer.eos_token_id or 50256] * (padded_len - actual_seq_len)
    tokens_pt = torch.tensor([input_ids_padded], dtype=torch.long)
    logger.info(f"Input: {actual_seq_len} tokens, padded to {padded_len}")

    # HF reference forward
    logger.info("Running HF forward...")
    hf_pre_final_norm = {}

    def _hook(module, inp, out):
        hf_pre_final_norm["h"] = inp[0].detach().float()

    hook_handle = hf_model.model.norm.register_forward_hook(_hook)
    hf_model(tokens_pt[:, :padded_len], output_hidden_states=True)
    hook_handle.remove()
    hf_decoder_out = hf_pre_final_norm["h"]  # [1, padded_len, 5120]
    logger.info(f"HF decoder output: shape={hf_decoder_out.shape}, std={hf_decoder_out.std():.6f}")

    # TTNN prefill forward
    logger.info("Running TTNN forward...")
    kv_cache = [layer.attention.layer_past for layer in tt_model.layers]

    ttnn_cos, ttnn_sin, _ = precompute_freqs_yarn(
        dim=model_args.head_dim,
        end=model_args.max_seq_len * 2,
        theta=model_args.rope_theta,
        scaling_factor=model_args.rope_scaling_factor,
        original_max_position_embeddings=model_args.original_max_position_embeddings,
        beta_fast=model_args.yarn_beta_fast,
        beta_slow=model_args.yarn_beta_slow,
        attention_factor=model_args.yarn_attention_factor,
    )
    cos_gathered, sin_gathered = gather_cos_sin(torch.arange(padded_len), ttnn_cos, ttnn_sin)
    rot_mats_prefill = [
        ttnn.from_torch(
            cos_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        ttnn.from_torch(
            sin_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
    ]
    tt_model.tt_rot_mats_prefill = rot_mats_prefill

    block_size = paged_attention_config.block_size
    num_prefill_blocks = math.ceil(padded_len / block_size)
    prefill_page_table = torch.ones(32, num_prefill_blocks, dtype=torch.int32) * -1
    prefill_page_table[0, :] = page_table[0, :num_prefill_blocks]

    host_inputs = tt_model.prepare_prefill_inputs_host(tokens_pt, user_id=0, page_table=prefill_page_table)
    device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
    transformed_inputs = tt_model.transform_prefill_inputs_device(*device_inputs)

    tt_hidden = tt_model.ttnn_prefill_forward(*transformed_inputs, kv_cache=kv_cache, batch_size=1)
    ttnn.synchronize_device(mesh_device)

    # Read back
    tt_hidden_torch = ttnn.to_torch(
        tt_hidden,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_hidden_trimmed = tt_hidden_torch[:, :1, :padded_len, : model_args.dim]

    # Compare PCC
    tt_hidden_2d = tt_hidden_trimmed[:, 0, :, :]
    passing, pcc = comp_pcc(hf_decoder_out, tt_hidden_2d.float(), 0.90)

    # Also compare at last token
    hf_last = hf_decoder_out[:, actual_seq_len - 1, :]
    tt_last = tt_hidden_trimmed[:, 0, actual_seq_len - 1, :]
    passing_last, pcc_last = comp_pcc(hf_last, tt_last.float(), 0.90)

    sdpa_mode = "ring_sdpa" if use_ring_sdpa else "standard_sdpa"
    logger.info(f"[{sdpa_mode}] seq_len={seq_len}: PCC(all)={pcc}, PCC(last_tok)={pcc_last}")
    logger.info(f"  HF std={hf_decoder_out.std():.6f}, TT std={tt_hidden_2d.std():.6f}")

    assert float(pcc) > 0.90, f"PCC {pcc} < 0.90 for {sdpa_mode} at seq_len={seq_len}"
    logger.info(f"PASSED: {sdpa_mode} seq_len={seq_len}, PCC={pcc}")

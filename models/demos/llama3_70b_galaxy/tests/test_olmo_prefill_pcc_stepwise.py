# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Prefill PCC Step-by-Step Comparison.

Compares TTNN prefill output vs HuggingFace reference at each stage:
1. Embedding output
2. After 1 decoder layer (hidden states)
3. After final norm
4. After lm_head (logits)

Run with:
    export HF_MODEL=~/.cache/huggingface/hub/models--allenai--Olmo-3.1-32B-Think/snapshots/<hash>
    export LINE_RS=1
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_prefill_pcc_stepwise.py -xvs
"""

import os
import math
import torch
import pytest
from loguru import logger
import ttnn

from models.demos.llama3_70b_galaxy.tt.llama_common import (
    PagedAttentionConfig,
    precompute_freqs_yarn,
    gather_cos_sin,
)
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.tt_transformers.tt.common import copy_host_to_device
from models.common.utility_functions import comp_pcc
from transformers import GPT2Tokenizer


def resolve_snapshot_path(hf_model_path):
    """Resolve HF cache dir to snapshot path if needed."""
    import glob

    base_path = os.path.expanduser(hf_model_path)
    if os.path.exists(os.path.join(base_path, "snapshots")):
        snapshot_dirs = glob.glob(os.path.join(base_path, "snapshots", "*"))
        if snapshot_dirs:
            return snapshot_dirs[0]
    return base_path


def load_hf_olmo3_1layer(hf_model_path):
    """Load HF OLMo3 model with only 1 layer for reference."""
    from transformers import AutoModelForCausalLM, AutoConfig

    snapshot_path = resolve_snapshot_path(hf_model_path)
    config = AutoConfig.from_pretrained(snapshot_path, trust_remote_code=True)
    config.num_hidden_layers = 1

    hf_model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
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
def test_olmo_prefill_pcc_stepwise(mesh_device, reset_seeds, ensure_gc):
    """Step-by-step PCC comparison of TTNN prefill vs HuggingFace OLMo3."""
    hf_model_path = os.environ.get("HF_MODEL")
    if not hf_model_path:
        pytest.skip("HF_MODEL not set")

    n_layers = 1
    max_seq_len = 256
    dtype = ttnn.bfloat8_b

    # ===== Load HF Reference (1 layer) =====
    logger.info("Loading HF OLMo3 model (1 layer)...")
    hf_model = load_hf_olmo3_1layer(hf_model_path)
    logger.info(f"HF model loaded: {type(hf_model).__name__}")

    # ===== Load TTNN Model (1 layer) =====
    logger.info("Loading TTNN model (1 layer)...")
    model_args = TtOlmoModelArgs(mesh_device, max_batch_size=32, max_seq_len=max_seq_len)
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

    # ===== Prepare Input =====
    tokenizer = model_args.tokenizer
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained(model_args.TOKENIZER_PATH)

    prompt = "What is your favorite condiment?"
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    seq_len = len(input_ids)
    padded_len = 128
    input_ids_padded = input_ids + [tokenizer.eos_token_id or 50256] * (padded_len - seq_len)
    tokens_pt = torch.tensor([input_ids_padded], dtype=torch.long)
    logger.info(f"Input: '{prompt}' -> {seq_len} tokens, padded to {padded_len}")

    # ===== Step 1: HF Embedding =====
    logger.info("=" * 60)
    logger.info("STEP 1: Embedding")
    logger.info("=" * 60)
    hf_embed = hf_model.model.embed_tokens(tokens_pt[:, :padded_len]).float()
    logger.info(f"HF embed shape: {hf_embed.shape}, mean={hf_embed.mean():.6f}, std={hf_embed.std():.6f}")

    # ===== Step 2: HF Full Forward (1 layer) =====
    logger.info("=" * 60)
    logger.info("STEP 2: HF Full Forward (1 layer)")
    logger.info("=" * 60)
    # NOTE: hf_outputs.hidden_states[1] is AFTER the model's final norm (hf_model.model.norm),
    # not the raw decoder output. We use a hook to capture the pre-final-norm hidden state.
    hf_pre_final_norm = {}

    def _hook_pre_final_norm(module, inp, out):
        hf_pre_final_norm["h2"] = inp[0].detach().float()  # decoder output before final norm

    hook_handle = hf_model.model.norm.register_forward_hook(_hook_pre_final_norm)

    hf_outputs = hf_model(tokens_pt[:, :padded_len], output_hidden_states=True)
    hook_handle.remove()

    hf_logits = hf_outputs.logits.float()
    hf_hidden_states = [h.float() for h in hf_outputs.hidden_states]

    # hf_decoder_out = pre-final-norm output (what TTNN ttnn_prefill_forward returns)
    hf_decoder_out = hf_pre_final_norm["h2"]  # [1, padded_len, 5120], std~0.14
    # hf_hidden_states[1] = post-final-norm output (std~0.96)
    logger.info(f"HF hidden states count: {len(hf_hidden_states)}")
    logger.info(
        f"  HF hidden[0] (embed):            mean={hf_hidden_states[0].mean():.6f}, std={hf_hidden_states[0].std():.6f}"
    )
    logger.info(
        f"  HF hidden[1] (post-final-norm):  mean={hf_hidden_states[1].mean():.6f}, std={hf_hidden_states[1].std():.6f}"
    )
    logger.info(f"  HF decoder_out (pre-final-norm): mean={hf_decoder_out.mean():.6f}, std={hf_decoder_out.std():.6f}")
    logger.info(f"HF logits shape: {hf_logits.shape}, mean={hf_logits.mean():.6f}, std={hf_logits.std():.6f}")

    hf_last_token_logits = hf_logits[:, seq_len - 1, :]
    hf_token = hf_last_token_logits.argmax(dim=-1).item()
    logger.info(f"HF predicted token: {hf_token} ({tokenizer.decode([hf_token])})")

    # ===== Step 3: TTNN Prefill Forward =====
    logger.info("=" * 60)
    logger.info("STEP 3: TTNN Prefill Forward")
    logger.info("=" * 60)

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
    position_ids = torch.arange(padded_len)
    cos_gathered, sin_gathered = gather_cos_sin(position_ids, ttnn_cos, ttnn_sin)
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

    # The forward returns hidden states after all layers (before norm/lm_head)
    tt_hidden = tt_model.ttnn_prefill_forward(
        *transformed_inputs,
        kv_cache=kv_cache,
        batch_size=1,
    )
    ttnn.synchronize_device(mesh_device)

    # Convert TTNN hidden states to torch
    tt_hidden_torch = ttnn.to_torch(
        tt_hidden,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    logger.info(f"TTNN raw tensor shape from mesh: {tt_hidden_torch.shape}")
    # Shape: [1, 8, seq_per_device, 5120] or [1, 8, padded_len, 5120]
    # For each slice along dim=1 (row device), check the stats to understand layout
    for i in range(min(4, tt_hidden_torch.shape[1])):
        sl = tt_hidden_torch[:, i, :, : model_args.dim]
        logger.info(
            f"  Row-device {i}: shape={sl.shape}, mean={sl.mean():.6f}, std={sl.std():.6f}, min={sl.min():.4f}, max={sl.max():.4f}"
        )
    # Trim to get hidden states
    tt_hidden_trimmed = tt_hidden_torch[:, :1, :padded_len, : model_args.dim]
    logger.info(
        f"TTNN hidden (after layers): shape={tt_hidden_trimmed.shape}, "
        f"mean={tt_hidden_trimmed.mean():.6f}, std={tt_hidden_trimmed.std():.6f}"
    )

    # ===== Step 4: Compare Hidden States (pre-final-norm) =====
    logger.info("=" * 60)
    logger.info("STEP 4: Compare Hidden States After 1 Layer (pre-final-norm)")
    logger.info("=" * 60)
    # CORRECT: compare TTNN pre-final-norm output with HF pre-final-norm decoder output
    # hf_decoder_out: captured via hook BEFORE hf.model.norm, std~0.14
    # tt_hidden_trimmed: raw decoder output from TTNN, std~0.14 (before process_output_prefill)
    logger.info(
        f"HF decoder_out (pre-final-norm): shape={hf_decoder_out.shape}, "
        f"mean={hf_decoder_out.mean():.6f}, std={hf_decoder_out.std():.6f}"
    )
    logger.info(
        f"TT hidden (pre-final-norm):      shape={tt_hidden_trimmed.squeeze(1).shape}, "
        f"mean={tt_hidden_trimmed.mean():.6f}, std={tt_hidden_trimmed.std():.6f}"
    )

    tt_hidden_2d = tt_hidden_trimmed[:, 0, :, :]  # [1, padded_len, 5120]
    passing_hidden, pcc_hidden = comp_pcc(hf_decoder_out, tt_hidden_2d.float(), 0.80)
    logger.info(f"Hidden states PCC (pre-final-norm, all tokens): {pcc_hidden}")

    # ===== Step 4b: Compare hidden states at the SPECIFIC token position =====
    logger.info("=" * 60)
    logger.info("STEP 4b: Compare Hidden States At Last Token Position (pre-final-norm)")
    logger.info("=" * 60)
    hf_decoder_out_last = hf_decoder_out[:, seq_len - 1, :]  # [1, 5120]
    tt_hidden_last_tok = tt_hidden_trimmed[:, 0, seq_len - 1, :]  # [1, 5120]
    passing_hidden_last, pcc_hidden_last = comp_pcc(hf_decoder_out_last, tt_hidden_last_tok.float(), 0.80)
    logger.info(f"Hidden states PCC at last token (pos {seq_len-1}): {pcc_hidden_last}")
    logger.info(f"HF decoder_out[last_tok]: mean={hf_decoder_out_last.mean():.6f}, std={hf_decoder_out_last.std():.6f}")
    logger.info(f"TT hidden[last_tok]:      mean={tt_hidden_last_tok.mean():.6f}, std={tt_hidden_last_tok.std():.6f}")

    # ===== Step 4c: Compare after final norm =====
    logger.info("=" * 60)
    logger.info("STEP 4c: Compare After Final Norm")
    logger.info("=" * 60)
    # Apply HF final norm to the pre-final-norm decoder output
    with torch.no_grad():
        hf_after_final_norm = hf_model.model.norm(hf_decoder_out).float()  # [1, seq_len, 5120]
    hf_after_final_norm_last = hf_after_final_norm[:, seq_len - 1, :]  # [1, 5120]
    logger.info(
        f"HF after final norm[last]: mean={hf_after_final_norm_last.mean():.6f}, std={hf_after_final_norm_last.std():.6f}"
    )
    # Sanity check: hf_after_final_norm should match hf_hidden_states[1]
    max_diff = (hf_after_final_norm - hf_hidden_states[1]).abs().max().item()
    logger.info(f"Sanity: |hf_after_final_norm - hf_hidden_states[1]| max = {max_diff:.6f} (should be ~0)")

    # ===== Step 4d: Isolate where TTNN norm+lm_head diverges =====
    logger.info("=" * 60)
    logger.info("STEP 4d: Apply TTNN final norm on-device, compare result vs HF")
    logger.info("=" * 60)
    # Apply model-level final norm to the full sequence (same as process_output_prefill does)
    tt_normed_out, _ = tt_model.norm(tt_hidden, res=None, mode="prefill")
    ttnn.synchronize_device(mesh_device)

    # Convert TTNN normed output to CPU (without ROW_MAJOR conversion first)
    tt_normed_torch_tiled = ttnn.to_torch(
        tt_normed_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_normed_tiled_last = tt_normed_torch_tiled[:, 0, seq_len - 1, : model_args.dim]
    logger.info(
        f"TTNN normed (TILE read) at last_tok: mean={tt_normed_tiled_last.mean():.6f}, std={tt_normed_tiled_last.std():.6f}"
    )
    _, pcc_norm_tiled = comp_pcc(hf_after_final_norm_last.float(), tt_normed_tiled_last.float(), 0.80)
    logger.info(f"TTNN norm out (TILE read) vs HF norm out at last_tok PCC: {pcc_norm_tiled}")

    # Convert to ROW_MAJOR first, then to CPU (user's suggestion)
    tt_normed_rm = ttnn.to_layout(tt_normed_out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_normed_torch_rm = ttnn.to_torch(
        tt_normed_rm,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_normed_rm_last = tt_normed_torch_rm[:, 0, seq_len - 1, : model_args.dim]
    logger.info(
        f"TTNN normed (ROW_MAJOR read) at last_tok: mean={tt_normed_rm_last.mean():.6f}, std={tt_normed_rm_last.std():.6f}"
    )
    _, pcc_norm_rm = comp_pcc(hf_after_final_norm_last.float(), tt_normed_rm_last.float(), 0.80)
    logger.info(f"TTNN norm out (ROW_MAJOR read) vs HF norm out at last_tok PCC: {pcc_norm_rm}")
    logger.info(f"  -> If pcc_norm_rm >> pcc_norm_tiled: layout is the issue in to_torch")
    logger.info(f"  -> If both low: norm weights wrong or norm computation wrong")
    logger.info(f"  -> If both high: problem is in lm_head on-device")

    # ===== Step 5: Compare Logits =====
    logger.info("=" * 60)
    logger.info("STEP 5: Get TTNN Logits (norm + lm_head)")
    logger.info("=" * 60)

    logits_buf_size = 100352
    tt_out_logits_saved = torch.zeros(1, logits_buf_size)
    tt_tok = tt_model.process_output_prefill(
        tt_hidden, last_token_idx=seq_len - 1, tt_out_logits_saved=tt_out_logits_saved
    )
    ttnn.synchronize_device(mesh_device)

    vocab_size = model_args.vocab_size
    tt_logits = tt_out_logits_saved[:, :vocab_size]
    tt_token = int(tt_tok[0])
    logger.info(f"TTNN token: {tt_token} ({tokenizer.decode([tt_token])})")
    logger.info(
        f"TTNN logits: mean={tt_logits.mean():.6f}, std={tt_logits.std():.6f}, min={tt_logits.min():.4f}, max={tt_logits.max():.4f}"
    )
    logger.info(f"TTNN top-5 token ids: {tt_logits[0].topk(5).indices.tolist()}")

    hf_logits_last = hf_logits[:, seq_len - 1, :vocab_size]
    logger.info(f"HF logits (last token): mean={hf_logits_last.mean():.6f}, std={hf_logits_last.std():.6f}")
    logger.info(f"HF top-5 token ids: {hf_logits_last[0].topk(5).indices.tolist()}")

    # Sanity: CPU lm_head(final_norm(decoder_out)) should match HF logits
    with torch.no_grad():
        hf_lm_head_out = hf_model.lm_head(hf_after_final_norm_last.unsqueeze(0)).float()  # [1, 1, vocab]
        hf_lm_head_out = hf_lm_head_out.squeeze(1)  # [1, vocab]
        _, pcc_cpu_logits = comp_pcc(hf_logits_last.float(), hf_lm_head_out.float()[:, :vocab_size], 0.99)
        logger.info(f"CPU lm_head(final_norm(decoder_out)) vs HF logits PCC: {pcc_cpu_logits} (sanity, should be ~1.0)")

    # KEY DIAGNOSTIC: Apply HF's norm+lm_head to TTNN's last token hidden state
    # This isolates whether the problem is in (a) TTNN decoder output or (b) TTNN norm/lm_head
    with torch.no_grad():
        tt_last_tok_f32 = tt_hidden_last_tok.float()  # [1, 5120]
        tt_after_cpu_norm = hf_model.model.norm(tt_last_tok_f32).float()  # [1, 5120]
        tt_via_cpu_logits = hf_model.lm_head(tt_after_cpu_norm.unsqueeze(0)).float().squeeze(1)  # [1, vocab]
        _, pcc_tt_via_cpu = comp_pcc(hf_logits_last.float(), tt_via_cpu_logits.float()[:, :vocab_size], 0.80)
        logger.info(f"CPU norm+lm_head(TTNN hidden[last]) vs HF logits PCC: {pcc_tt_via_cpu}")
        logger.info(f"  (if low → TTNN decoder hidden state at last token is wrong)")
        logger.info(f"  (if high → TTNN norm/lm_head path has a bug)")
        tt_via_cpu_top5 = tt_via_cpu_logits[0, :vocab_size].topk(5).indices.tolist()
        logger.info(f"  top-5 from TTNN hidden via CPU norm+lm_head: {tt_via_cpu_top5}")

        # Also compare TTNN logits vs CPU-applied norm+lm_head on TTNN hidden
        _, pcc_tt_vs_cpu_path = comp_pcc(tt_logits.float(), tt_via_cpu_logits.float()[:, :vocab_size], 0.80)
        logger.info(f"TTNN logits vs CPU norm+lm_head(TTNN hidden) PCC: {pcc_tt_vs_cpu_path}")
        logger.info(f"  (if low → TTNN norm/lm_head itself is broken for OLMo)")

    passing_logits, pcc_logits = comp_pcc(hf_logits_last.float(), tt_logits.float(), 0.80)
    logger.info(f"Logits PCC (last token): {pcc_logits}")

    # ===== Summary =====
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Hidden states PCC (pre-final-norm, all tokens): {pcc_hidden}")
    logger.info(f"Hidden states PCC (pre-final-norm, last token): {pcc_hidden_last}")
    logger.info(f"CPU norm+lm_head(TTNN hidden) vs HF PCC:        {pcc_tt_via_cpu} <- upper bound")
    logger.info(f"TTNN norm (TILE read) vs HF norm at last_tok:   {pcc_norm_tiled}")
    logger.info(f"TTNN norm (ROW_MAJOR) vs HF norm at last_tok:   {pcc_norm_rm}")
    logger.info(f"TTNN logits vs HF logits PCC:                   {pcc_logits}")
    logger.info(f"TTNN logits vs CPU-path-on-TTNN-hidden PCC:     {pcc_tt_vs_cpu_path}")
    logger.info(f"HF token: {hf_token} ({tokenizer.decode([hf_token])})")
    logger.info(f"TT token: {tt_token} ({tokenizer.decode([tt_token])})")
    logger.info(f"Token match: {hf_token == tt_token}")
    logger.info("=" * 60)

    # Don't assert yet — we want to see the numbers first
    if not passing_hidden:
        logger.warning(f"Hidden PCC FAIL: {pcc_hidden}")
    if not passing_logits:
        logger.warning(f"Logits PCC FAIL: {pcc_logits}")

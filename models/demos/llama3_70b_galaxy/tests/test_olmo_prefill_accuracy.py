# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Quick test to compare OLMo prefill output against HuggingFace reference.
"""

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_pcc(ref, test):
    """Compute Pearson correlation coefficient between two tensors."""
    ref_flat = ref.flatten().float()
    test_flat = test.flatten().float()

    ref_mean = ref_flat.mean()
    test_mean = test_flat.mean()

    ref_centered = ref_flat - ref_mean
    test_centered = test_flat - test_mean

    numerator = (ref_centered * test_centered).sum()
    denominator = torch.sqrt((ref_centered**2).sum() * (test_centered**2).sum())

    if denominator == 0:
        return 0.0
    return (numerator / denominator).item()


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 102000000,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_olmo_prefill_vs_hf(mesh_device, reset_seeds, ensure_gc):
    """
    Compare TT OLMo prefill output against HuggingFace reference.

    This test:
    1. Runs a short prefill on both TT and HF models
    2. Compares the final layer output (logits)
    3. Reports PCC (Pearson Correlation Coefficient)
    """

    dtype = ttnn.bfloat8_b
    batch_size = 1  # Single user for comparison
    max_seq_len = 4096
    prefill_len = 128  # Short sequence for quick test
    n_layers = 1  # Use 1 layer for faster test

    logger.info("=" * 60)
    logger.info("OLMo Prefill Accuracy Test vs HuggingFace")
    logger.info("=" * 60)

    # Initialize TT model args
    model_args = TtOlmoModelArgs(
        mesh_device,
        instruct=False,
        max_seq_len=max_seq_len,
        max_batch_size=32,  # Model expects batch 32
        n_layers=n_layers,
    )

    logger.info(f"Model: OLMo-3.1-32B-Think")
    logger.info(f"Layers: {model_args.n_layers}")
    logger.info(f"Prefill length: {prefill_len}")
    logger.info(f"is_post_norm: {getattr(model_args, 'is_post_norm', False)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.TOKENIZER_PATH)

    # Load HuggingFace model (only 1 layer)
    logger.info("Loading HuggingFace model (1 layer)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_args.CKPT_DIR,
        num_hidden_layers=n_layers,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    hf_model.eval()

    # Load TT model
    logger.info("Loading TT model...")
    state_dict = model_args.load_state_dict()

    # Paged attention config
    paged_attention_config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=4096,
    )

    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    # Create test input - need exactly prefill_len tokens (must be multiple of 128)
    test_prompt = "The quick brown fox jumps over the lazy dog. " * 20  # Long enough prompt
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt")
    # Truncate or pad to exactly prefill_len tokens
    if input_ids.shape[1] < prefill_len:
        # Pad with pad_token_id (or eos_token_id if pad_token_id is None)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        padding = torch.full((1, prefill_len - input_ids.shape[1]), pad_token_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, padding], dim=1)
    else:
        input_ids = input_ids[:, :prefill_len]
    logger.info(f"Input shape: {input_ids.shape}")

    # Run HuggingFace prefill
    logger.info("Running HuggingFace prefill...")
    hf_output = hf_model(input_ids, output_hidden_states=True, use_cache=False)
    hf_logits = hf_output.logits[0, -1, :]  # Last token logits
    hf_hidden = hf_output.hidden_states[-1][0, -1, :]  # Last layer hidden states

    logger.info(f"HF logits shape: {hf_logits.shape}")
    logger.info(f"HF hidden shape: {hf_hidden.shape}")
    logger.info(f"HF logits stats: min={hf_logits.min():.4f}, max={hf_logits.max():.4f}, mean={hf_logits.mean():.4f}")

    # Get top-5 tokens from HF
    hf_top5_probs, hf_top5_ids = torch.topk(torch.softmax(hf_logits.float(), dim=-1), k=5)
    logger.info("HF Top-5 predictions:")
    for i, (prob, tok_id) in enumerate(zip(hf_top5_probs, hf_top5_ids)):
        token_text = tokenizer.decode([tok_id])
        logger.info(f"  {i+1}. {token_text!r} (id={tok_id.item()}, prob={prob.item():.4f})")

    # Run TT prefill
    logger.info("Running TT prefill...")

    # Get embeddings
    tt_embd = TtLlamaEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,
    )

    # Create input tensor - need to expand to batch 32 for TT model
    input_ids_expanded = input_ids.expand(32, -1)

    # Get embedding output
    pt_embd = hf_model.model.embed_tokens(input_ids)
    pt_embd_expanded = pt_embd.expand(32, -1, -1)  # [32, seq_len, dim]

    # Prepare for TT model
    tt_input = model_args.prepare_residual_tensor_prefill(
        pt_embd_expanded,
        force_replicated=True,
    )

    # Setup page table
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(32, paged_attention_config.max_num_blocks // 32)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Get rotary matrices
    current_pos = torch.zeros(32, dtype=torch.long)
    rot_mats = tt_model.rope_setup.get_rm_rot_mats(current_pos)

    # Run TT forward
    tt_output = tt_model(
        tt_input,
        None,  # current_pos (not used in prefill)
        rot_mats=rot_mats,
        mode="prefill",
        page_table=page_table_tt,
        user_id=0,
    )

    # Convert TT output to torch
    tt_logits_tensor = ttnn.to_torch(
        tt_output[0],
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            dims=(3, 1),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    # Get last token logits from first batch
    tt_logits = tt_logits_tensor[0, 0, -1, : model_args.vocab_size]

    logger.info(f"TT logits shape: {tt_logits.shape}")
    logger.info(f"TT logits stats: min={tt_logits.min():.4f}, max={tt_logits.max():.4f}, mean={tt_logits.mean():.4f}")

    # Get top-5 tokens from TT
    tt_top5_probs, tt_top5_ids = torch.topk(torch.softmax(tt_logits.float(), dim=-1), k=5)
    logger.info("TT Top-5 predictions:")
    for i, (prob, tok_id) in enumerate(zip(tt_top5_probs, tt_top5_ids)):
        token_text = tokenizer.decode([tok_id])
        logger.info(f"  {i+1}. {token_text!r} (id={tok_id.item()}, prob={prob.item():.4f})")

    # Compare outputs
    pcc = compute_pcc(hf_logits.float(), tt_logits.float())
    logger.info(f"\nPCC (logits): {pcc:.6f}")

    # Check if top-1 matches
    hf_top1 = hf_top5_ids[0].item()
    tt_top1 = tt_top5_ids[0].item()
    top1_match = hf_top1 == tt_top1
    logger.info(f"Top-1 match: {top1_match} (HF={hf_top1}, TT={tt_top1})")

    # Check if HF top-1 is in TT top-5
    hf_in_tt_top5 = hf_top1 in tt_top5_ids.tolist()
    logger.info(f"HF top-1 in TT top-5: {hf_in_tt_top5}")

    tt_model.tt_ccl.close()

    # Assert reasonable accuracy
    logger.info("=" * 60)
    if pcc < 0.9:
        logger.error(f"PCC {pcc:.4f} is too low! Expected >= 0.9")
    else:
        logger.info(f"PCC {pcc:.4f} is acceptable")

    assert pcc >= 0.8, f"PCC {pcc:.4f} is too low (expected >= 0.8)"

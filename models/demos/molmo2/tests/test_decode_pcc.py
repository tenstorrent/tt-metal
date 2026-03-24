# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test decode step PCC to find KV cache corruption.
Compares TTNN decode against PyTorch reference at each step.
"""

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.molmo2.reference.functional import _apply_rope, rmsnorm


def calculate_pcc(a, b):
    """Calculate Pearson Correlation Coefficient."""
    a = a.flatten().float()
    b = b.flatten().float()

    a_mean = a.mean()
    b_mean = b.mean()
    a_centered = a - a_mean
    b_centered = b - b_mean

    numerator = (a_centered * b_centered).sum()
    denominator = torch.sqrt((a_centered**2).sum() * (b_centered**2).sum())

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    pcc = (numerator / denominator).item()
    return pcc


def reference_attention_with_kv_cache(
    hidden_states,  # [B, 1, H] - single token
    wq,
    wk,
    wv,
    wo,
    q_norm_weight,
    k_norm_weight,
    kv_cache_k,
    kv_cache_v,  # [B, num_kv_heads, cached_len, head_dim]
    current_pos,  # Current position in sequence
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    rope_theta=1000000.0,
):
    """Reference attention with KV cache for decode.

    Uses the same RoPE as TTNN (half-span via ``_apply_rope``) and the same GQA
    broadcast as ``text_attention_forward`` in reference/functional.py.
    """
    batch_size = hidden_states.shape[0]
    num_kv_groups = num_heads // num_kv_heads

    # Q, K, V projections (same layout as models.demos.molmo2.reference.functional)
    q = F.linear(hidden_states, wq)
    k = F.linear(hidden_states, wk)
    v = F.linear(hidden_states, wv)

    q = q.view(batch_size, 1, num_heads, head_dim)
    k = k.view(batch_size, 1, num_kv_heads, head_dim)
    v = v.view(batch_size, 1, num_kv_heads, head_dim)

    q = rmsnorm(q, q_norm_weight)
    k = rmsnorm(k, k_norm_weight)

    position_ids = torch.full((batch_size, 1), current_pos, dtype=torch.long, device=hidden_states.device)
    q, k = _apply_rope(q, k, position_ids, head_dim, rope_theta)

    # [B, 1, heads, D] -> [B, heads, 1, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Update KV cache at current position
    kv_cache_k[:, :, current_pos : current_pos + 1, :] = k
    kv_cache_v[:, :, current_pos : current_pos + 1, :] = v

    # Get all K, V up to current position
    k_full = kv_cache_k[:, :, : current_pos + 1, :]
    v_full = kv_cache_v[:, :, : current_pos + 1, :]

    # GQA: match functional.py expand+reshape (same ordering as repeat_interleave on dim=1)
    seq_kv = current_pos + 1
    k_expanded = (
        k_full.unsqueeze(2)
        .expand(batch_size, num_kv_heads, num_kv_groups, seq_kv, head_dim)
        .reshape(batch_size, num_heads, seq_kv, head_dim)
    )
    v_expanded = (
        v_full.unsqueeze(2)
        .expand(batch_size, num_kv_heads, num_kv_groups, seq_kv, head_dim)
        .reshape(batch_size, num_heads, seq_kv, head_dim)
    )

    # Attention (decode: one query position attends over full KV length)
    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights.float(), dim=-1).to(attn_weights.dtype)
    attn_output = torch.matmul(attn_weights, v_expanded)

    # Reshape and output projection
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, num_heads * head_dim)
    output = F.linear(attn_output, wo)

    return output


def test_decode_pcc():
    """Test decode step PCC."""
    from models.demos.molmo2.demo.demo import load_model_weights

    logger.info("=" * 60)
    logger.info("Testing Decode Step PCC")
    logger.info("=" * 60)

    # Load state dict using existing function
    logger.info("Loading weights...")
    state_dict = load_model_weights()
    logger.info(f"Loaded {len(state_dict)} weight tensors")

    # Config
    hidden_dim = 4096
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    # Note: SDPA requires K sequence length to be multiple of chunk_size (256)
    # Following tt_transformers pattern: pad prefill to 128, use max_seq_len=256
    seq_len = 128  # Padded prefill length (must be multiple of 32 for TILE_LAYOUT)
    num_decode_steps = 20

    # Extract weights for layer 0
    # Molmo2 uses fused QKV: att_proj contains [Q, K, V] concatenated
    layer_idx = 0
    prefix = f"model.transformer.blocks.{layer_idx}"

    # Fused QKV weight: [q_dim + 2*kv_dim, hidden_dim] = [6144, 4096]
    att_proj = state_dict[f"{prefix}.self_attn.att_proj.weight"].float()
    q_dim = num_heads * head_dim  # 32 * 128 = 4096
    kv_dim = num_kv_heads * head_dim  # 8 * 128 = 1024

    # Split fused weight into Q, K, V
    wq = att_proj[:q_dim, :]  # [4096, 4096]
    wk = att_proj[q_dim : q_dim + kv_dim, :]  # [1024, 4096]
    wv = att_proj[q_dim + kv_dim :, :]  # [1024, 4096]
    wo = state_dict[f"{prefix}.self_attn.attn_out.weight"].float()
    q_norm = state_dict[f"{prefix}.self_attn.q_norm.weight"].float()
    k_norm = state_dict[f"{prefix}.self_attn.k_norm.weight"].float()

    # Create random prefill hidden states
    torch.manual_seed(42)
    prefill_hidden = torch.randn(1, seq_len, hidden_dim)

    # Initialize KV cache for reference
    # Note: max_seq_len must be >= padded prefill + decode steps AND multiple of 256 for SDPA
    max_seq_len = 256
    ref_k_cache = torch.zeros(1, num_kv_heads, max_seq_len, head_dim)
    ref_v_cache = torch.zeros(1, num_kv_heads, max_seq_len, head_dim)

    # Run prefill to populate KV cache (positions 0 to seq_len-1)
    logger.info(f"Running prefill for {seq_len} tokens...")
    for pos in range(seq_len):
        token_hidden = prefill_hidden[:, pos : pos + 1, :]
        _ = reference_attention_with_kv_cache(
            token_hidden,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            ref_k_cache,
            ref_v_cache,
            pos,
            num_heads,
            num_kv_heads,
            head_dim,
        )

    logger.info(f"Prefill complete. KV cache filled for positions 0-{seq_len-1}")

    # Open TTNN device
    logger.info("Opening TTNN device...")
    device = ttnn.open_device(device_id=0)

    try:
        from models.demos.molmo2.tt.text_attention import TextAttention
        from models.demos.molmo2.tt.text_model import init_decode_position, init_kv_cache
        from models.demos.molmo2.tt.text_rotary_setup import TextRotarySetup

        mesh_mapper = ttnn.ReplicateTensorToMesh(device)

        # Create TTNN attention block
        ttnn_attn = TextAttention(
            mesh_device=device,
            state_dict=state_dict,
            layer_num=layer_idx,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=ttnn.bfloat8_b,
        )

        # Create rotary setup
        rotary_setup = TextRotarySetup(
            mesh_device=device,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=1000000.0,
        )
        transformation_mats = rotary_setup.get_transformation_mats()

        # Initialize TTNN KV cache (single layer)
        ttnn_kv_cache = init_kv_cache(
            device,
            num_layers=1,
            batch_size=1,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            dtype=ttnn.bfloat8_b,
        )[0]

        # Copy reference KV cache to TTNN cache
        # Convert reference cache to bfloat8_b format and fill
        ref_k_to_copy = ref_k_cache[:, :, :seq_len, :].to(torch.bfloat16)
        ref_v_to_copy = ref_v_cache[:, :, :seq_len, :].to(torch.bfloat16)

        k_ttnn = ttnn.from_torch(
            ref_k_to_copy,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        v_ttnn = ttnn.from_torch(
            ref_v_to_copy,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
        )

        ttnn.fill_cache(ttnn_kv_cache[0], k_ttnn, batch_idx=0)
        ttnn.fill_cache(ttnn_kv_cache[1], v_ttnn, batch_idx=0)
        ttnn.deallocate(k_ttnn)
        ttnn.deallocate(v_ttnn)

        # Initialize decode position
        current_pos = init_decode_position(device, batch_size=1, initial_pos=seq_len)

        # Run decode steps
        logger.info(f"\nRunning {num_decode_steps} decode steps...")
        logger.info("-" * 60)

        for step in range(num_decode_steps):
            pos = seq_len + step

            # Generate random decode hidden state
            decode_hidden = torch.randn(1, 1, hidden_dim)

            # Reference decode
            ref_output = reference_attention_with_kv_cache(
                decode_hidden,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                ref_k_cache,
                ref_v_cache,
                pos,
                num_heads,
                num_kv_heads,
                head_dim,
            )

            # TTNN decode
            decode_hidden_ttnn = ttnn.from_torch(
                decode_hidden.unsqueeze(0).to(torch.bfloat16),  # [1, 1, 1, hidden_dim]
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
            )

            pos_tensor = torch.tensor([pos], dtype=torch.int32)
            rot_mats = rotary_setup.get_rot_mats_decode(pos_tensor)

            ttnn_output = ttnn_attn.forward_decode(
                decode_hidden_ttnn,
                rot_mats,
                transformation_mats["decode"],
                ttnn_kv_cache,
                current_pos,
            )
            ttnn.synchronize_device(device)

            ttnn_output_torch = ttnn.to_torch(ttnn_output)[0, 0, :, :]

            # Calculate PCC and assert — decode hidden states must meet >= 0.99
            pcc = calculate_pcc(ref_output.squeeze(0), ttnn_output_torch)
            logger.info(f"Step {step + 1} (pos={pos}): PCC = {pcc:.6f}")
            assert pcc >= 0.99, (
                f"Decode step {step + 1} (pos={pos}) PCC {pcc:.6f} < 0.99. "
                "Check for bfloat8_b overflow or weight mapping issues."
            )

            # Update position
            new_pos = ttnn.from_torch(
                torch.tensor([pos + 1], dtype=torch.int32),
                dtype=ttnn.int32,
                device=device,
                mesh_mapper=mesh_mapper,
            )
            ttnn.copy(new_pos, current_pos)
            ttnn.deallocate(new_pos)
            ttnn.deallocate(decode_hidden_ttnn)
            ttnn.deallocate(ttnn_output)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_decode_pcc()

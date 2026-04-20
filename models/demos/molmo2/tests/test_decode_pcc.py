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
from models.demos.molmo2.reference.functional import _apply_rope


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
    """Reference attention with KV cache for decode."""
    batch_size = hidden_states.shape[0]
    num_kv_groups = num_heads // num_kv_heads

    # Q, K, V projections
    q = F.linear(hidden_states, wq)  # [B, 1, num_heads * head_dim]
    k = F.linear(hidden_states, wk)  # [B, 1, num_kv_heads * head_dim]
    v = F.linear(hidden_states, wv)

    # Reshape
    q = q.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, 1, head_dim]
    k = k.view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)

    # QK-norm (RMSNorm)
    def rms_norm(x, weight, eps=1e-5):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight

    q = rms_norm(q, q_norm_weight)
    k = rms_norm(k, k_norm_weight)

    # RoPE: Molmo2 / TTNN use half-split rotate + duplicated cos/sin (same as reference/functional.py),
    # not interleaved even/odd pairing.
    q = q.transpose(1, 2)  # [B, 1, num_heads, head_dim]
    k = k.transpose(1, 2)  # [B, 1, num_kv_heads, head_dim]
    position_ids = torch.full(
        (batch_size, 1),
        float(current_pos),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    q, k = _apply_rope(q, k, position_ids, head_dim, rope_theta)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    # Update KV cache at current position
    kv_cache_k[:, :, current_pos : current_pos + 1, :] = k
    kv_cache_v[:, :, current_pos : current_pos + 1, :] = v

    # Get all K, V up to current position
    k_full = kv_cache_k[:, :, : current_pos + 1, :]
    v_full = kv_cache_v[:, :, : current_pos + 1, :]

    # Expand K, V for GQA
    k_expanded = k_full.repeat_interleave(num_kv_groups, dim=1)
    v_expanded = v_full.repeat_interleave(num_kv_groups, dim=1)

    # Attention
    scale = 1.0 / (head_dim**0.5)
    attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v_expanded)

    # Reshape and output projection
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, num_heads * head_dim)
    output = F.linear(attn_output, wo)

    return output


def reference_decode_intermediates(
    hidden_states,
    wq,
    wk,
    wv,
    wo,
    q_norm_weight,
    k_norm_weight,
    kv_cache_k,
    kv_cache_v,
    current_pos,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    rope_theta=1000000.0,
):
    """
    Same math as reference_attention_with_kv_cache, but return tensors after QK-norm,
    after RoPE (current token K only), attention pre-output-proj, and final output.
    Used to localize PCC loss (RoPE vs SDPA vs output proj).
    """
    batch_size = hidden_states.shape[0]
    num_kv_groups = num_heads // num_kv_heads

    q = F.linear(hidden_states, wq)
    k = F.linear(hidden_states, wk)
    v = F.linear(hidden_states, wv)

    q = q.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)

    def rms_norm(x, weight, eps=1e-5):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight

    q = rms_norm(q, q_norm_weight)
    k = rms_norm(k, k_norm_weight)

    q_for_rope = q.transpose(1, 2)
    k_for_rope = k.transpose(1, 2)
    position_ids = torch.full(
        (batch_size, 1),
        float(current_pos),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    q_rot, k_rot = _apply_rope(q_for_rope, k_for_rope, position_ids, head_dim, rope_theta)
    q_rot = q_rot.transpose(1, 2)
    k_rot = k_rot.transpose(1, 2)

    kv_cache_k = kv_cache_k.clone()
    kv_cache_v = kv_cache_v.clone()
    kv_cache_k[:, :, current_pos : current_pos + 1, :] = k_rot
    kv_cache_v[:, :, current_pos : current_pos + 1, :] = v

    k_full = kv_cache_k[:, :, : current_pos + 1, :]
    v_full = kv_cache_v[:, :, : current_pos + 1, :]
    k_expanded = k_full.repeat_interleave(num_kv_groups, dim=1)
    v_expanded = v_full.repeat_interleave(num_kv_groups, dim=1)

    scale = 1.0 / (head_dim**0.5)
    attn_weights = torch.matmul(q_rot, k_expanded.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v_expanded)
    attn_merged = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, num_heads * head_dim)
    output = F.linear(attn_merged, wo)

    return {
        "q_after_norm": q,
        "k_after_norm": k,
        "q_after_rope": q_rot,
        "k_new_after_rope": k_rot,
        "attn_pre_wo": attn_merged,
        "output": output,
    }


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

            # Snapshot KV before this token (staged reference must use the same state as one-shot ref)
            kv_k_snapshot = ref_k_cache.clone()
            kv_v_snapshot = ref_v_cache.clone()

            # Reference decode (updates ref_k_cache / ref_v_cache)
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

            # Same forward on KV snapshot for per-stage tensors (for logging / isolation)
            stages = reference_decode_intermediates(
                decode_hidden,
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                kv_k_snapshot,
                kv_v_snapshot,
                pos,
                num_heads,
                num_kv_heads,
                head_dim,
            )
            pcc_ref_vs_stages = calculate_pcc(ref_output.squeeze(0), stages["output"].squeeze(0))
            logger.info(f"  Reference self-check (full vs staged): PCC = {pcc_ref_vs_stages:.6f} " f"(should be ~1.0)")
            assert pcc_ref_vs_stages >= 0.99999, "Staged reference should match one-shot reference"

            # TTNN RoPE-only vs reference (uses same cos/sin cache + token index as forward_decode)
            if step == 0:
                q_pre = stages["q_after_norm"].transpose(1, 2).contiguous().to(torch.bfloat16)  # [1,1,H,D]
                k_pre = stages["k_after_norm"].transpose(1, 2).contiguous().to(torch.bfloat16)
                q_tt = ttnn.from_torch(
                    q_pre,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=mesh_mapper,
                )
                k_tt = ttnn.from_torch(
                    k_pre,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=mesh_mapper,
                )
                q_rr = ttnn.experimental.rotary_embedding(
                    q_tt,
                    rotary_setup.cos_matrix,
                    rotary_setup.sin_matrix,
                    int(pos),
                )
                k_rr = ttnn.experimental.rotary_embedding(
                    k_tt,
                    rotary_setup.cos_matrix,
                    rotary_setup.sin_matrix,
                    int(pos),
                )
                ttnn.synchronize_device(device)
                q_rr_t = ttnn.to_torch(q_rr)[0, 0, :num_heads, :].float()
                k_rr_t = ttnn.to_torch(k_rr)[0, 0, :num_kv_heads, :].float()
                ref_qr = stages["q_after_rope"][0, :, 0, :].float()
                ref_kr = stages["k_new_after_rope"][0, :, 0, :].float()
                pcc_rope_q = calculate_pcc(ref_qr, q_rr_t)
                pcc_rope_k = calculate_pcc(ref_kr, k_rr_t)
                logger.info(
                    f"  TTNN RoPE vs ref (step 0, pos={pos}): PCC Q = {pcc_rope_q:.6f}, " f"PCC K = {pcc_rope_k:.6f}"
                )
                ttnn.deallocate(q_tt)
                ttnn.deallocate(k_tt)
                ttnn.deallocate(q_rr)
                ttnn.deallocate(k_rr)

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

            ret = ttnn_attn.forward_decode(
                decode_hidden_ttnn,
                rot_mats,
                transformation_mats["decode"],
                ttnn_kv_cache,
                current_pos,
                return_attn_pre_wo=(step == 0),
            )
            if step == 0:
                ttnn_output, attn_pre_wo_tt = ret
            else:
                ttnn_output = ret
            ttnn.synchronize_device(device)

            ttnn_output_torch = ttnn.to_torch(ttnn_output)[0, 0, :, :]

            if step == 0:
                ref_attn_vec = stages["attn_pre_wo"].reshape(-1).float()
                attn_tt_vec = attn_pre_wo_tt.float().reshape(-1)
                min_len = min(ref_attn_vec.numel(), attn_tt_vec.numel())
                pcc_attn = calculate_pcc(ref_attn_vec[:min_len], attn_tt_vec[:min_len])
                # Isolate output projection: fp32 Wo on TTNN pre-Wo vs full reference output
                attn_row = attn_pre_wo_tt[0, 0, 0, :].float()
                out_via_fp32_wo = F.linear(attn_row.unsqueeze(0), wo).squeeze(0)
                pcc_out_fp32_wo = calculate_pcc(ref_output[0, 0].float(), out_via_fp32_wo)
                logger.info(
                    f"Step {step + 1} (pos={pos}): PCC attn (pre-Wo, ref vs TTNN) = {pcc_attn:.6f}; "
                    f"PCC if Wo were fp32 (ref vs attn_tt @ Wo_fp32) = {pcc_out_fp32_wo:.6f}"
                )

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

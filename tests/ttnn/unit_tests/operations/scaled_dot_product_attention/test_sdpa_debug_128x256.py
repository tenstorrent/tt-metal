# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug test for flash attention numerical mismatch.

Shape: (1,1,128,256) — S_q=128 (1 Q-block, B_q_t=4), S_kv=256 (2 KV-blocks, B_kv_t=4), D=64 (D_t=2).

This is the simplest case that fails: num_kv_blocks=2.

Uses torch.manual_seed(42) for reproducibility (matches acceptance test).
Computes reference with the SAME block decomposition as the kernel (2 KV-blocks)
so we can compare block-by-block.
"""

import math
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def flash_attention_reference_blocked(Q, K, V, *, scale=None, B_kv=128):
    """Flash attention with explicit KV-blocking matching the kernel.

    Q: (B, H, S_q, D), K/V: (B, H, S_kv, D) — all bf16, upcast to fp32.
    B_kv: KV block size in rows (128 = 4 tiles).
    """
    Qf = Q.float()
    Kf = K.float()
    Vf = V.float()
    B, H_q, S_q, D = Qf.shape
    _, H_kv, S_kv, _ = Kf.shape

    # GQA replication
    if H_q != H_kv:
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    output = torch.empty_like(Qf)
    num_kv_blocks = (S_kv + B_kv - 1) // B_kv

    for b in range(B):
        for h in range(H_q):
            Q_bh = Qf[b, h]  # (S_q, D)
            K_bh = Kf[b, h]  # (S_kv, D)
            V_bh = Vf[b, h]  # (S_kv, D)

            # Online softmax recurrence — block by block
            m_i = torch.full((S_q,), float("-inf"), dtype=torch.float32)
            l_i = torch.zeros((S_q,), dtype=torch.float32)
            O_i = torch.zeros((S_q, D), dtype=torch.float32)

            for kvb in range(num_kv_blocks):
                kv_start = kvb * B_kv
                kv_end = min(kv_start + B_kv, S_kv)
                K_blk = K_bh[kv_start:kv_end, :]  # (B_kv, D)
                V_blk = V_bh[kv_start:kv_end, :]  # (B_kv, D)

                # Phase 1-2: scores = Q @ K^T * scale
                scores = (Q_bh @ K_blk.T) * scale  # (S_q, B_kv)

                # Phase 4: row-max
                m_blk = scores.max(dim=-1).values  # (S_q,)

                # Phase 5: alpha = exp(m_old - m_new)
                alpha = torch.exp(m_i - m_blk)  # (S_q,)

                # Phase 6: O *= alpha
                O_i = alpha.unsqueeze(-1) * O_i

                # Phase 7: l *= alpha
                l_i = alpha * l_i

                # Phase 8: scores -= m_new
                scores = scores - m_blk.unsqueeze(-1)

                # Phase 9: exp
                exp_scores = torch.exp(scores)  # (S_q, B_kv)

                # Phase 10: row-sum
                l_blk = exp_scores.sum(dim=-1)  # (S_q,)

                # Phase 11: l += l_blk
                l_i = l_i + l_blk

                # Phase 12: O += P @ V
                O_i = O_i + exp_scores @ V_blk

                # Phase 13: m = m_new
                m_i = m_blk

            # Phase 14: normalize
            output[b, h] = O_i / l_i.unsqueeze(-1)

    return output.to(Q.dtype)


def test_sdpa_debug_128x256(device):
    """Debug test: (1,1,128,256), D=64, 2 KV-blocks — simplest failing case."""
    torch.manual_seed(42)

    Q = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
    K = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
    V = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)

    # Reference with block decomposition matching the kernel (B_kv=128 -> 2 blocks)
    expected_blocked = flash_attention_reference_blocked(Q, K, V, B_kv=128)

    # Also compute the "standard" (single-block) reference for comparison
    expected_single = flash_attention_reference_blocked(Q, K, V, B_kv=256)

    # torch F.scaled_dot_product_attention for ground truth
    scale = 1.0 / math.sqrt(64)
    expected_torch = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float(), scale=scale)

    print(f"\n=== Reference comparison ===")
    diff_blocked_vs_single = (expected_blocked.float() - expected_single.float()).abs().max().item()
    diff_blocked_vs_torch = (expected_blocked.float() - expected_torch).abs().max().item()
    diff_single_vs_torch = (expected_single.float() - expected_torch).abs().max().item()
    print(f"Max diff blocked-vs-single: {diff_blocked_vs_single:.2e}")
    print(f"Max diff blocked-vs-torch:  {diff_blocked_vs_torch:.2e}")
    print(f"Max diff single-vs-torch:   {diff_single_vs_torch:.2e}")

    # Run on device
    ttnn_Q = ttnn.from_torch(
        Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
    torch_output = ttnn.to_torch(output)

    print(f"\n=== Device vs references ===")
    diff_device_vs_blocked = (torch_output.float() - expected_blocked.float()).abs().max().item()
    diff_device_vs_torch = (torch_output.float() - expected_torch).abs().max().item()
    print(f"Max diff device-vs-blocked: {diff_device_vs_blocked:.2e}")
    print(f"Max diff device-vs-torch:   {diff_device_vs_torch:.2e}")

    # PCC computation
    output_f32 = torch_output.float().flatten()
    expected_f32 = expected_torch.flatten()
    output_centered = output_f32 - output_f32.mean()
    expected_centered = expected_f32 - expected_f32.mean()
    numerator = (output_centered * expected_centered).sum()
    denominator = torch.sqrt((output_centered**2).sum()) * torch.sqrt((expected_centered**2).sum())
    pcc = (numerator / denominator).item() if denominator > 0 else 0.0
    print(f"PCC (device vs torch): {pcc:.6f}")

    # Print first few rows for comparison
    print(f"\n=== First 4x4 output (device) ===")
    print(torch_output[0, 0, :4, :4])
    print(f"\n=== First 4x4 output (blocked ref) ===")
    print(expected_blocked[0, 0, :4, :4])
    print(f"\n=== First 4x4 output (torch ref) ===")
    print(expected_torch[0, 0, :4, :4])

    print(f"\n=== Diff (device - torch) first 4x4 ===")
    print((torch_output[0, 0, :4, :4].float() - expected_torch[0, 0, :4, :4]).abs())

    assert pcc >= 0.995, f"PCC {pcc:.6f} < threshold 0.995"

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc


class TestSDPA130CoreGrid:
    """
    Debug tests for SDPA KV chain forwarding with full 130-core grid (13x10).
    Previous implementation worked on 8x8 (64 cores) but hung on 130 cores.
    """

    def test_minimal_shape_130_cores(self, device):
        """
        Minimal test with full 130-core grid.
        Shape: B=1, NH=1, S=64, DH=64
        """
        B, NH, S, DH = 1, 1, 64, 64
        torch.manual_seed(42)

        q = torch.randn(B, NH, S, DH)
        k = torch.randn(B, NH, S, DH)
        v = torch.randn(B, NH, S, DH)

        q_tt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        k_tt = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v_tt = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Use FULL device grid
        full_grid = device.compute_with_storage_grid_size()
        print(f"\nTesting with full grid: {full_grid} = {full_grid.x * full_grid.y} cores")

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=full_grid,
            q_chunk_size=32,
            k_chunk_size=32,
            enable_kv_chain_forwarding=True,
        )

        output_tt = ttnn.transformer.scaled_dot_product_attention(
            q_tt, k_tt, v_tt, is_causal=False, program_config=program_config
        )

        output = ttnn.to_torch(output_tt)
        print(f"✓ Test completed without hang!")
        print(f"  Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")

    def test_multi_head_130_cores(self, device):
        """
        Multi-head test that should trigger chain forwarding on 130 cores.
        Shape: B=1, NH=8, S=256, DH=64

        With 130 cores and 8 heads, some heads will definitely span multiple cores.
        """
        B, NH, S, DH = 1, 8, 256, 64
        torch.manual_seed(42)

        q = torch.randn(B, NH, S, DH)
        k = torch.randn(B, NH, S, DH)
        v = torch.randn(B, NH, S, DH)

        q_tt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        k_tt = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v_tt = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        full_grid = device.compute_with_storage_grid_size()
        print(f"\nTesting multi-head with full grid: {full_grid}")
        print(f"  Q chunks per head: {S // 32} = {S // 32}")
        print(f"  Total Q chunks: {NH * (S // 32)}")

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=full_grid,
            q_chunk_size=32,
            k_chunk_size=32,
            enable_kv_chain_forwarding=True,
        )

        output_tt = ttnn.transformer.scaled_dot_product_attention(
            q_tt, k_tt, v_tt, is_causal=False, program_config=program_config
        )

        output = ttnn.to_torch(output_tt)
        print(f"✓ Multi-head test completed!")

        # Compare with PyTorch
        scale = 1.0 / (DH**0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        expected = torch.matmul(attn, v)

        passing, pcc = comp_pcc(expected, output, pcc=0.99)
        assert passing, f"PCC check failed: {pcc}"
        print(f"  ✓ PCC: {pcc:.4f}")

    def test_130_cores_chain_disabled(self, device):
        """
        Baseline test with 130 cores but chain forwarding disabled.
        This should always work as it's the original behavior.
        """
        B, NH, S, DH = 1, 8, 256, 64
        torch.manual_seed(42)

        q = torch.randn(B, NH, S, DH)
        k = torch.randn(B, NH, S, DH)
        v = torch.randn(B, NH, S, DH)

        q_tt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        k_tt = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v_tt = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        full_grid = device.compute_with_storage_grid_size()

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=full_grid,
            q_chunk_size=32,
            k_chunk_size=32,
            enable_kv_chain_forwarding=False,  # DISABLED
        )

        output_tt = ttnn.transformer.scaled_dot_product_attention(
            q_tt, k_tt, v_tt, is_causal=False, program_config=program_config
        )

        output = ttnn.to_torch(output_tt)
        print(f"\n✓ 130-core test with chain DISABLED works")
        print(f"  Output mean: {output.mean().item():.4f}")

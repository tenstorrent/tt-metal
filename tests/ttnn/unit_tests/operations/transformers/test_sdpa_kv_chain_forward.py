# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc


def run_sdpa_test(
    device, B, NH, S, DH, dtype=ttnn.bfloat16, enable_kv_chain_forwarding=True, q_chunk_size=32, k_chunk_size=32
):
    """
    Run non-causal SDPA and compare against PyTorch reference.

    Uses small shapes to ensure fast execution (<1s).
    """
    torch.manual_seed(42)

    # Create random inputs
    q = torch.randn(B, NH, S, DH)
    k = torch.randn(B, NH, S, DH)
    v = torch.randn(B, NH, S, DH)

    # PyTorch reference (non-causal)
    scale = 1.0 / (DH**0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    expected = torch.matmul(attn, v)

    # TTNN execution
    q_tt = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    k_tt = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    v_tt = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Create program config with KV chain forwarding option
    from ttnn.operations.transformer import SDPAProgramConfig

    program_config = SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),  # Tested and working with 8x8 grid
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        enable_kv_chain_forwarding=enable_kv_chain_forwarding,
    )

    # Non-causal SDPA
    output_tt = ttnn.transformer.scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        is_causal=False,
        program_config=program_config,
    )

    output = ttnn.to_torch(output_tt)

    # Verify correctness
    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    return passing, pcc


class TestSDPAKVChainForward:
    """
    Test suite for SDPA KV chain forwarding optimization.

    All tests use small shapes for fast iteration (<1s execution).
    """

    @pytest.mark.parametrize(
        "B, NH, S, DH",
        [
            # Minimal test - single head, fits on few cores
            (1, 1, 64, 64),
            # Multi-head - forces Q chunks across cores (triggers chain)
            (1, 8, 128, 64),
            # Larger sequence - more Q chunks per head
            (1, 4, 256, 64),
            # Multi-batch
            (2, 4, 128, 64),
        ],
        ids=[
            "minimal_1h",
            "multi_head_8h",
            "longer_seq_256",
            "multi_batch",
        ],
    )
    def test_sdpa_non_causal_correctness(self, device, B, NH, S, DH):
        """Test non-causal SDPA produces correct results."""
        passing, pcc = run_sdpa_test(device, B, NH, S, DH)
        assert passing, f"PCC check failed: {pcc}"

    def test_sdpa_chain_trigger_shape(self, device):
        """
        Test with shape that guarantees chain forwarding is triggered.

        With NH=8, S=256, chunk_size=32:
        - Q chunks per head = 256/32 = 8
        - Total Q chunks = 8 * 8 = 64
        - With 64 cores, each head's Q chunks span multiple cores
        """
        B, NH, S, DH = 1, 8, 256, 64
        passing, pcc = run_sdpa_test(device, B, NH, S, DH)
        assert passing, f"PCC check failed: {pcc}"

    @pytest.mark.parametrize("enable_chain", [True, False], ids=["chain_enabled", "chain_disabled"])
    def test_sdpa_with_and_without_chain(self, device, enable_chain):
        """
        Test that SDPA produces correct results with chain forwarding enabled and disabled.

        This verifies:
        1. The enable_kv_chain_forwarding config option works
        2. Both code paths produce correct results
        3. The optimization doesn't affect correctness
        """
        B, NH, S, DH = 1, 4, 128, 64
        passing, pcc = run_sdpa_test(device, B, NH, S, DH, enable_kv_chain_forwarding=enable_chain)
        assert passing, f"PCC check failed with chain={enable_chain}: {pcc}"

    def test_sdpa_chain_disabled_explicitly(self, device):
        """
        Explicitly test with chain forwarding disabled.

        Verifies that setting enable_kv_chain_forwarding=False works correctly.
        """
        B, NH, S, DH = 1, 8, 256, 64
        passing, pcc = run_sdpa_test(device, B, NH, S, DH, enable_kv_chain_forwarding=False)
        assert passing, f"PCC check failed with chain disabled: {pcc}"

    @pytest.mark.parametrize(
        "q_chunk, k_chunk",
        [
            (64, 128),  # Known working from sprint test
            (64, 256),  # Known hanging from sprint test
            (128, 256),  # Also fails in sprint test
        ],
        ids=["q64_k128", "q64_k256", "q128_k256"],
    )
    def test_different_chunk_sizes(self, device, q_chunk, k_chunk):
        """
        Test various chunk size combinations to identify which ones work.

        Based on sprint test failures, k_chunk > q_chunk may have issues.
        """
        B, NH, S, DH = 1, 10, 2368, 128  # Same as sprint test
        passing, pcc = run_sdpa_test(
            device, B, NH, S, DH, enable_kv_chain_forwarding=True, q_chunk_size=q_chunk, k_chunk_size=k_chunk
        )
        assert passing, f"PCC check failed with q_chunk={q_chunk}, k_chunk={k_chunk}: {pcc}"

    def test_sprint_exact_config(self, device):
        """
        Test exact configuration from sprint test that was passing.
        B=1, NH=10, S=2368, DH=128, q=64, k=128
        """
        B, NH, S, DH = 1, 10, 2368, 128
        torch.manual_seed(1234)

        Q = torch.randn(B, NH, S, DH)
        K = torch.randn(B, NH, S, DH)
        V = torch.randn(B, NH, S, DH)

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),  # Tested and working with 8x8 grid
            q_chunk_size=64,
            k_chunk_size=128,
            exp_approx_mode=True,
            enable_kv_chain_forwarding=True,
        )

        Q_tt = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        K_tt = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        V_tt = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Just run SDPA without checking results (like sprint test with do_check=False)
        output_tt = ttnn.transformer.scaled_dot_product_attention(
            Q_tt,
            K_tt,
            V_tt,
            is_causal=False,
            program_config=program_config,
        )

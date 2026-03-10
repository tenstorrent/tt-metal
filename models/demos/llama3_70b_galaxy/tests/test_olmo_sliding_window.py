# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Sliding Window Attention Test.

Tests sliding window attention for ring distributed SDPA against PyTorch reference.

OLMo uses hybrid attention pattern: 3 sliding (window=4096) + 1 full, repeated 16 times.

Run with:
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_sliding_window.py -v -x
"""

import torch
import pytest
from loguru import logger
import ttnn

from models.demos.llama3_70b_galaxy.reference.sliding_window import (
    sliding_window_attention,
    create_sliding_window_mask,
    SlidingWindowConfig,
)
from models.common.utility_functions import comp_pcc


def reference_sdpa_with_sliding_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sliding_window: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Reference SDPA with sliding window support.

    Args:
        q: Query [batch, n_heads, seq_len, head_dim]
        k: Key [batch, n_heads, seq_len, head_dim]
        v: Value [batch, n_heads, seq_len, head_dim]
        sliding_window: Window size (None = full attention)
        scale: Attention scale factor

    Returns:
        Output tensor [batch, n_heads, seq_len, head_dim]
    """
    return sliding_window_attention(q, k, v, sliding_window, scale)


class TestSlidingWindowSDPA:
    """Test sliding window SDPA functionality."""

    @pytest.mark.parametrize("seq_len", [64, 128, 256])
    @pytest.mark.parametrize("sliding_window", [32, 64, None])
    @pytest.mark.parametrize(
        "device_params",
        [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
        indirect=True,
    )
    def test_single_device_sliding_window(self, device, seq_len, sliding_window):
        """Test single device SDPA with sliding window against reference."""
        batch = 1
        n_heads = 8
        head_dim = 64

        # Skip invalid combinations where window > seq_len makes no sense
        if sliding_window is not None and sliding_window > seq_len:
            pytest.skip(f"Sliding window {sliding_window} > seq_len {seq_len}")

        logger.info(f"Testing: seq_len={seq_len}, sliding_window={sliding_window}")

        # Create random inputs
        torch.manual_seed(42)
        q = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16)
        k = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16)

        # Reference computation
        ref_out = reference_sdpa_with_sliding_window(q.float(), k.float(), v.float(), sliding_window=sliding_window).to(
            torch.bfloat16
        )

        # TTNN computation
        q_tt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        k_tt = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v_tt = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        tt_out = ttnn.transformer.scaled_dot_product_attention(
            q_tt,
            k_tt,
            v_tt,
            is_causal=True,
            sliding_window_size=sliding_window,
        )

        tt_out_torch = ttnn.to_torch(tt_out).to(torch.bfloat16)

        # Compare
        pcc_threshold = 0.99
        passing, pcc_message = comp_pcc(ref_out, tt_out_torch, pcc_threshold)

        logger.info(f"PCC: {pcc_message}")
        assert passing, f"PCC {pcc_message} < {pcc_threshold}"

    def test_sliding_window_mask_correctness(self):
        """Verify sliding window mask is mathematically correct."""
        seq_len = 16
        sliding_window = 4

        mask = create_sliding_window_mask(seq_len, sliding_window)

        # Check that position i can attend to positions max(0, i-window+1) to i
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    # Future positions should be masked
                    assert mask[i, j] == float("-inf"), f"Position {i} should not attend to future position {j}"
                elif i - j >= sliding_window:
                    # Outside sliding window should be masked
                    assert mask[i, j] == float("-inf"), f"Position {i} should not attend to {j} (outside window)"
                else:
                    # Within window and not future should be 0
                    assert mask[i, j] == 0, f"Position {i} should attend to {j}"

        logger.info("Sliding window mask correctness verified!")


def gather_and_reshuffle_ring_outputs(ring_outputs, ring_size, global_seq_len):
    """Gather outputs from all ring positions and reshuffle to restore global sequence order."""
    chunk_size = global_seq_len // (2 * ring_size)
    batch_size, num_heads, _, head_dim = ring_outputs[0].shape
    final_output = torch.zeros(batch_size, num_heads, global_seq_len, head_dim)

    for device_id, device_output in enumerate(ring_outputs):
        first_chunk_id = device_id
        second_chunk_id = (2 * ring_size - 1) - device_id

        first_chunk_output = device_output[:, :, :chunk_size, :]
        second_chunk_output = device_output[:, :, chunk_size : 2 * chunk_size, :]

        first_start = first_chunk_id * chunk_size
        first_end = first_start + chunk_size
        second_start = second_chunk_id * chunk_size
        second_end = second_start + chunk_size

        final_output[:, :, first_start:first_end, :] = first_chunk_output
        final_output[:, :, second_start:second_end, :] = second_chunk_output

    return final_output


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
class TestRingDistributedSlidingWindow:
    """Test ring distributed SDPA with sliding window (single device simulation)."""

    @pytest.mark.parametrize("seq_len", [512, 1024])
    @pytest.mark.parametrize("sliding_window", [256, 512, None])
    @pytest.mark.parametrize("ring_size", [4, 8])
    def test_ring_sdpa_sliding_window(self, device, seq_len, sliding_window, ring_size):
        """Test ring distributed SDPA with sliding window by simulating ring on single device."""
        batch = 1
        n_heads = 8
        n_kv_heads = 1
        head_dim = 128

        # Skip invalid combinations
        if sliding_window is not None and sliding_window > seq_len:
            pytest.skip(f"Sliding window {sliding_window} > seq_len {seq_len}")

        # Ring SDPA requires seq_len divisible by 2*ring_size
        if seq_len % (2 * ring_size) != 0:
            pytest.skip(f"seq_len {seq_len} not divisible by 2*ring_size={2*ring_size}")

        logger.info(f"Testing Ring SDPA: seq_len={seq_len}, sliding_window={sliding_window}, ring_size={ring_size}")

        # Create random inputs
        torch.manual_seed(42)
        q = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16)
        k = torch.randn(batch, n_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch, n_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)

        # Reference computation
        # For reference, expand K/V to match Q heads for proper GQA comparison
        k_expanded = k.repeat(1, n_heads // n_kv_heads, 1, 1)
        v_expanded = v.repeat(1, n_heads // n_kv_heads, 1, 1)
        ref_out = reference_sdpa_with_sliding_window(
            q.float(), k_expanded.float(), v_expanded.float(), sliding_window=sliding_window
        ).to(torch.bfloat16)

        # TTNN computation - simulate ring by running for each ring_id
        q_tt = ttnn.from_torch(q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        k_tt = ttnn.from_torch(k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        v_tt = ttnn.from_torch(v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

        # Run ring SDPA for each ring_id and collect outputs
        ring_outputs = []
        for ring_id in range(ring_size):
            tt_out = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
                q_tt,
                k_tt,
                v_tt,
                ring_size=ring_size,
                ring_id=ring_id,
                sliding_window_size=sliding_window,
            )

            tt_out_torch = ttnn.to_torch(tt_out)
            # Each device gets 2 chunks of size seq_len / (2 * ring_size)
            local_seq_len = seq_len // ring_size
            tt_out_torch = tt_out_torch[:, :, :local_seq_len, :head_dim]
            ring_outputs.append(tt_out_torch)

        # Gather and reshuffle to reconstruct full sequence
        final_output = gather_and_reshuffle_ring_outputs(ring_outputs, ring_size, seq_len)

        # Compare against reference
        pcc_threshold = 0.98
        passing, pcc_message = comp_pcc(ref_out, final_output, pcc_threshold)

        logger.info(f"Ring SDPA sliding window PCC: {pcc_message}")
        assert passing, f"Ring SDPA sliding window PCC {pcc_message} < {pcc_threshold}"


class TestOlmoLayerTypes:
    """Test OLMo layer type determination."""

    def test_layer_type_pattern(self):
        """Verify OLMo attention pattern: 3 sliding + 1 full, repeated."""
        config = SlidingWindowConfig()

        # Check pattern for all 64 layers
        for layer_id in range(64):
            layer_type = config.get_layer_type(layer_id)
            window = config.get_sliding_window_size(layer_id)

            # Every 4th layer (3, 7, 11, ..., 63) should be full attention
            if (layer_id + 1) % 4 == 0:
                assert layer_type == "full_attention", f"Layer {layer_id} should be full_attention"
                assert window is None, f"Layer {layer_id} window should be None"
            else:
                assert layer_type == "sliding_attention", f"Layer {layer_id} should be sliding_attention"
                assert window == 4096, f"Layer {layer_id} window should be 4096"

        # Verify counts
        layer_types = config.get_all_layer_types()
        sliding_count = sum(1 for t in layer_types if t == "sliding_attention")
        full_count = sum(1 for t in layer_types if t == "full_attention")

        assert sliding_count == 48, f"Expected 48 sliding layers, got {sliding_count}"
        assert full_count == 16, f"Expected 16 full layers, got {full_count}"

        logger.info("OLMo layer type pattern verified: 48 sliding, 16 full")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])

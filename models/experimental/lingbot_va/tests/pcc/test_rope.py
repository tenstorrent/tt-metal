# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.metrics import compute_pcc
from models.experimental.lingbot_va.tt.wan_RoPE import TtWanRotaryPosEmbed
from models.experimental.lingbot_va.reference.model import WanRotaryPosEmbed
from loguru import logger


def test_wan_rotary_pos_embed():
    """
    Test comparing TT TtWanRotaryPosEmbed with PyTorch reference WanRotaryPosEmbed.
    """
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

    # Parameters matching typical usage
    attention_head_dim = 128
    patch_size = (1, 2, 2)
    max_seq_len = 1024
    theta = 10000.0

    # Initialize PyTorch reference model
    logger.info("Initializing PyTorch WanRotaryPosEmbed")
    torch_rope = WanRotaryPosEmbed(
        attention_head_dim=attention_head_dim,
        patch_size=patch_size,
        max_seq_len=max_seq_len,
        theta=theta,
    )
    torch_rope.eval()

    # Initialize TT model
    logger.info("Initializing TT TtWanRotaryPosEmbed")
    tt_rope = TtWanRotaryPosEmbed(
        mesh_device=mesh_device,
        attention_head_dim=attention_head_dim,
        patch_size=patch_size,
        max_seq_len=max_seq_len,
        theta=theta,
    )

    # Create test input: grid_ids of shape [batch, 3, seq_len]
    # grid_ids[:, 0, :] = f (frame) indices
    # grid_ids[:, 1, :] = h (height) indices
    # grid_ids[:, 2, :] = w (width) indices
    B = 1
    seq_len = 1152  # Sequence length for testing
    torch.manual_seed(42)

    # Generate grid IDs - typically these would come from patch positions
    # For testing, use random integers in reasonable range
    grid_ids = torch.randint(0, 50, (B, 3, seq_len), dtype=torch.float32)
    logger.info(f"Input grid_ids shape: {grid_ids.shape}")
    logger.info(
        f"Grid IDs range: f=[{grid_ids[:, 0, :].min():.1f}, {grid_ids[:, 0, :].max():.1f}], "
        f"h=[{grid_ids[:, 1, :].min():.1f}, {grid_ids[:, 1, :].max():.1f}], "
        f"w=[{grid_ids[:, 2, :].min():.1f}, {grid_ids[:, 2, :].max():.1f}]"
    )

    # Run PyTorch reference model
    logger.info("Running PyTorch reference model")
    with torch.no_grad():
        torch_freqs_cis = torch_rope(grid_ids)

    # Convert complex tensor to cos/sin format for comparison
    # torch.polar returns complex: real = cos(freqs), imag = sin(freqs)
    # We need to extract real and imaginary parts and concatenate like TT version
    torch_freqs_cis_real = torch_freqs_cis.real  # [batch, seq_len, attention_head_dim//2]
    torch_freqs_cis_imag = torch_freqs_cis.imag  # [batch, seq_len, attention_head_dim//2]

    # TT version returns [batch, seq_len, attention_head_dim] with cos and sin concatenated
    # Reference: cos_freqs = ttnn.cos(freqs), sin_freqs = ttnn.sin(freqs)
    # Then: freqs_cis = ttnn.concat([cos_freqs, sin_freqs], dim=-1)
    # So we need to concatenate real (cos) and imag (sin) parts
    torch_output = torch.cat([torch_freqs_cis_real, torch_freqs_cis_imag], dim=-1)
    logger.info(f"PyTorch output shape: {torch_output.shape}")

    # Prepare TT input
    logger.info("Preparing TT input")
    grid_ids_tt = ttnn.from_torch(grid_ids, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    logger.info(f"TT grid_ids shape: {grid_ids_tt.shape}")

    # Run TT model
    logger.info("Running TT model")
    tt_freqs_cis = tt_rope(grid_ids_tt)

    # Convert TT output back to torch
    tt_output = ttnn.to_torch(tt_freqs_cis)
    logger.info(f"TT output shape: {tt_output.shape}")

    # Compare outputs
    logger.info("Comparing outputs")
    logger.info(f"PyTorch output shape: {torch_output.shape}, dtype: {torch_output.dtype}")
    logger.info(f"TT output shape: {tt_output.shape}, dtype: {tt_output.dtype}")

    # Ensure shapes match
    assert (
        torch_output.shape == tt_output.shape
    ), f"Shape mismatch: PyTorch {torch_output.shape} vs TT {tt_output.shape}"

    # Compute PCC
    pcc = compute_pcc(torch_output, tt_output)
    logger.info(f"PCC: {pcc:.6f}")

    # Also compute per-dimension statistics
    diff = torch.abs(torch_output - tt_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    logger.info(f"Max absolute difference: {max_diff:.6f}")
    logger.info(f"Mean absolute difference: {mean_diff:.6f}")

    # Assert minimum PCC threshold
    MIN_PCC = 0.99
    assert pcc >= MIN_PCC, f"PCC {pcc:.6f} is below threshold {MIN_PCC}"

    logger.info(f"✓ Test passed: PCC = {pcc:.6f}")

    # Cleanup
    ttnn.close_device(mesh_device)


if __name__ == "__main__":
    test_wan_rotary_pos_embed()

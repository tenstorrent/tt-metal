"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""

"""
Verify PCC accuracy of optimized encoder.

Target: PCC > 0.99
"""

import torch
import ttnn
from transformers import DPTForDepthEstimation

from tt_optimized_encoder import create_optimized_encoder


def compute_pcc(a, b):
    """Compute Pearson Correlation Coefficient."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_mean = a_flat.mean()
    b_mean = b_flat.mean()
    a_centered = a_flat - a_mean
    b_centered = b_flat - b_mean
    numerator = (a_centered * b_centered).sum()
    denominator = a_centered.norm() * b_centered.norm()
    return (numerator / (denominator + 1e-8)).item()


def pad_to_tile_multiple(tensor, multiple=32):
    """Pad sequence dimension to tile multiple."""
    B, N, C = tensor.shape
    N_padded = ((N + multiple - 1) // multiple) * multiple
    if N_padded == N:
        return tensor, N
    pad = torch.zeros(B, N_padded - N, C, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=1), N


def unpad_from_tile_multiple(tensor_tt, original_len, device):
    """Remove padding from TT tensor and convert to torch."""
    tensor_host = tensor_tt.cpu()
    if hasattr(tensor_host, "layout") and tensor_host.layout == ttnn.TILE_LAYOUT:
        tensor_host = tensor_host.to(ttnn.ROW_MAJOR_LAYOUT)
    tensor_torch = tensor_host.to_torch()
    # Handle 4D shape from ttnn
    if tensor_torch.dim() == 4:
        tensor_torch = tensor_torch.squeeze(1)
    return tensor_torch[:, :original_len, :]


def test_pcc():
    """Test PCC accuracy of optimized encoder."""
    print("=" * 60)
    print("PCC Accuracy Test - Optimized Encoder")
    print("=" * 60)

    device = ttnn.open_device(device_id=0, l1_small_size=32768)

    print("\n[1/5] Loading model...")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    model.eval()
    state_dict = model.state_dict()

    print("[2/5] Creating optimized encoder...")
    encoder = create_optimized_encoder(state_dict, device)

    print("[3/5] Preparing test input...")
    pixel_values = torch.randn(1, 3, 384, 384)

    # Get CPU embeddings
    with torch.no_grad():
        emb_out = model.dpt.embeddings(pixel_values)
        # Handle different output formats
        if hasattr(emb_out, "last_hidden_state"):
            embeddings = emb_out.last_hidden_state
        elif isinstance(emb_out, (tuple, list)):
            embeddings = emb_out[0]
        elif hasattr(emb_out, "__getitem__") and hasattr(emb_out, "to_tuple"):
            embeddings = emb_out.to_tuple()[0]
        else:
            embeddings = emb_out
        embeddings_no_cls = embeddings[:, 1:, :]  # Remove CLS for TT

    print(f"    Embeddings shape: {embeddings.shape}")
    print(f"    Embeddings no CLS shape: {embeddings_no_cls.shape}")

    print("[4/5] Running CPU reference encoder WITHOUT CLS token...")
    with torch.no_grad():
        # Run CPU encoder WITHOUT CLS (to match TT encoder)
        hidden_states = embeddings_no_cls
        cpu_outputs = []
        for i, layer in enumerate(model.dpt.encoder.layer):
            layer_out = layer(hidden_states)
            # Handle tuple output from ViT layer
            if isinstance(layer_out, tuple):
                hidden_states = layer_out[0]
            else:
                hidden_states = layer_out
            if (i + 1) in [5, 11, 17, 23]:
                cpu_outputs.append(hidden_states)

    print("[5/5] Running optimized encoder (bfloat16 mode for accuracy)...")
    # Pad and convert for TT
    emb_padded, orig_len = pad_to_tile_multiple(embeddings_no_cls)
    emb = emb_padded.to(torch.bfloat16)
    emb_tt = ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Use bfloat16 mode for better PCC
    tt_outputs = encoder(emb_tt, use_bfloat8_b=False)
    ttnn.synchronize_device(device)

    print("\n" + "=" * 60)
    print("PCC RESULTS")
    print("=" * 60)

    layer_names = ["Layer 5", "Layer 11", "Layer 17", "Layer 23"]
    pccs = []

    for i, (cpu_out, tt_out, name) in enumerate(zip(cpu_outputs, tt_outputs, layer_names)):
        # Convert TT output back to torch
        tt_torch = unpad_from_tile_multiple(tt_out, orig_len, device)

        # Compute PCC
        pcc = compute_pcc(cpu_out, tt_torch)
        pccs.append(pcc)

        status = "✅" if pcc >= 0.99 else "❌"
        print(f"  {name}: PCC = {pcc:.6f} {status}")

    avg_pcc = sum(pccs) / len(pccs)
    min_pcc = min(pccs)

    print(f"\n  Average PCC: {avg_pcc:.6f}")
    print(f"  Minimum PCC: {min_pcc:.6f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Target:    PCC > 0.99")
    print(f"  Achieved:  PCC = {min_pcc:.4f}")

    if min_pcc >= 0.99:
        print(f"\n  ✅ PCC TARGET ACHIEVED!")
    else:
        print(f"\n  ❌ PCC below target (gap: {0.99 - min_pcc:.4f})")

    ttnn.close_device(device)

    return min_pcc >= 0.99


if __name__ == "__main__":
    test_pcc()

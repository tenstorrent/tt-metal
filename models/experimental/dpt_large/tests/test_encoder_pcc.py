# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test DPT-Large encoder PCC accuracy and performance.

Validates:
- PCC > 0.99 against PyTorch/HuggingFace reference
- Encoder throughput for 20+ FPS target
"""

import sys
import time
import torch
import ttnn
from transformers import DPTForDepthEstimation

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from tt_encoder import create_encoder


def compute_pcc(a, b):
    """Compute Pearson Correlation Coefficient between two tensors."""
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
    """Pad sequence dimension to tile multiple for TTNN compatibility."""
    B, N, C = tensor.shape
    N_padded = ((N + multiple - 1) // multiple) * multiple
    if N_padded == N:
        return tensor, N
    pad = torch.zeros(B, N_padded - N, C, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=1), N


def test_encoder_pcc():
    """Test encoder PCC accuracy and performance against targets."""
    print("=" * 60)
    print("DPT-Large Encoder Test")
    print("=" * 60)

    device = ttnn.open_device(device_id=0, l1_small_size=32768)

    print("\n[1/6] Loading model...")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    model.eval()
    state_dict = model.state_dict()

    print("[2/6] Creating optimized encoder...")
    encoder = create_encoder(state_dict, device)

    print("[3/6] Preparing test input...")
    pixel_values = torch.randn(1, 3, 384, 384)

    with torch.no_grad():
        emb_out = model.dpt.embeddings(pixel_values)
        if hasattr(emb_out, "last_hidden_state"):
            embeddings = emb_out.last_hidden_state
        elif isinstance(emb_out, (tuple, list)):
            embeddings = emb_out[0]
        elif hasattr(emb_out, "__getitem__") and hasattr(emb_out, "to_tuple"):
            embeddings = emb_out.to_tuple()[0]
        else:
            embeddings = emb_out
        embeddings_no_cls = embeddings[:, 1:, :]

        # Get CPU reference outputs at DPT output layers
        hidden = embeddings_no_cls
        cpu_outputs = []
        for i, layer in enumerate(model.dpt.encoder.layer):
            layer_out = layer(hidden)
            if isinstance(layer_out, tuple):
                hidden = layer_out[0]
            else:
                hidden = layer_out
            if (i + 1) in [5, 11, 17, 23]:
                cpu_outputs.append(hidden.clone())

    emb_padded, orig_len = pad_to_tile_multiple(embeddings_no_cls)
    emb = emb_padded.to(torch.bfloat16)
    emb_tt = ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    print(f"    Input shape: {emb_padded.shape}")

    print("[4/6] Warmup...")
    for _ in range(3):
        outputs = encoder(emb_tt)
        ttnn.synchronize_device(device)

    print("[5/6] Benchmarking...")
    times = []
    for _ in range(10):
        emb_tt = ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        t0 = time.perf_counter()
        outputs = encoder(emb_tt)
        ttnn.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1000)

    avg_time = sum(times[3:]) / len(times[3:])

    print("[6/6] Checking PCC...")
    layer_names = ["Layer 5", "Layer 11", "Layer 17", "Layer 23"]
    min_pcc = 1.0

    for name, cpu_out, tt_out in zip(layer_names, cpu_outputs, outputs):
        tt_host = tt_out.cpu()
        if hasattr(tt_host, "layout") and tt_host.layout == ttnn.TILE_LAYOUT:
            tt_host = tt_host.to(ttnn.ROW_MAJOR_LAYOUT)
        tt_torch = tt_host.to_torch()
        if tt_torch.dim() == 4:
            tt_torch = tt_torch.squeeze(1)
        tt_torch = tt_torch[:, :orig_len, :]

        pcc = compute_pcc(cpu_out, tt_torch)
        min_pcc = min(min_pcc, pcc)
        status = "PASS" if pcc >= 0.99 else "FAIL"
        print(f"    {name}: PCC = {pcc:.6f} [{status}]")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Encoder time: {avg_time:.1f}ms")
    print(f"  Min PCC:      {min_pcc:.4f}")

    # Full pipeline estimate (embeddings + H2D + encoder + head)
    overhead = 5.9 + 0.7 + 13.5
    total = overhead + avg_time
    fps = 1000 / total

    print(f"\n  Full pipeline estimate: {total:.1f}ms = {fps:.1f} FPS")
    print(f"  Target: 50.0ms = 20.0 FPS, PCC > 0.99")

    pcc_pass = min_pcc >= 0.99
    fps_pass = fps >= 20

    print("\n" + "=" * 60)
    if pcc_pass and fps_pass:
        print("RESULT: PASS - Both targets achieved")
    elif pcc_pass:
        print(f"RESULT: FAIL - PCC OK but FPS too low (gap: {total - 50:.1f}ms)")
    elif fps_pass:
        print(f"RESULT: FAIL - FPS OK but PCC too low (gap: {0.99 - min_pcc:.4f})")
    else:
        print("RESULT: FAIL - Both targets missed")
    print("=" * 60)

    ttnn.close_device(device)

    return pcc_pass and fps_pass


if __name__ == "__main__":
    success = test_encoder_pcc()
    exit(0 if success else 1)

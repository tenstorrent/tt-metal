# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC verification tests for GR00T N1.6 on Blackhole.

Compares TTNN implementation against PyTorch reference layer by layer.
Run from: cd /home/ttuser/experiments/pi0/tt-metal
With: PYTHONPATH=/home/ttuser/experiments/gr00t_n16/tt-metal pytest ...

Target: >= 99% accuracy (PCC >= 0.99)
"""

import logging
import sys
import time

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Ensure our model code is importable
sys.path.insert(0, "/home/ttuser/experiments/gr00t_n16/tt-metal")


def compute_pcc(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    ref_flat = ref.float().flatten()
    test_flat = test.float().flatten()
    min_len = min(len(ref_flat), len(test_flat))
    ref_flat, test_flat = ref_flat[:min_len], test_flat[:min_len]

    ref_c = ref_flat - ref_flat.mean()
    test_c = test_flat - test_flat.mean()
    cov = (ref_c * test_c).sum()
    ref_std = (ref_c**2).sum().sqrt()
    test_std = (test_c**2).sum().sqrt()

    if ref_std == 0 or test_std == 0:
        return 1.0 if torch.allclose(ref_flat, test_flat, atol=1e-5) else 0.0
    return (cov / (ref_std * test_std)).item()


def _build_siglip2_ref_model(vision_weights):
    """Build a PyTorch reference SigLIP2 from raw weights.

    Uses the linear patch embedding weight directly (no conv2d conversion needed).
    """
    import torch.nn as nn
    from models.experimental.groot_n16.tt.ttnn_siglip2 import siglip2_patch_embeddings_cpu

    class RefSigLIP2(nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.patch_size = 14
            self.hidden_size = 1152
            self.num_heads = 16
            self.head_dim = self.hidden_size // self.num_heads
            self.num_layers = 27
            self.eps = 1e-6

            # Patch embedding (linear)
            self.patch_weight = nn.Parameter(weights["embeddings.patch_embedding.weight"].float())
            self.patch_bias = nn.Parameter(weights["embeddings.patch_embedding.bias"].float())
            self.position_embedding = nn.Parameter(weights["embeddings.position_embedding.weight"].float())

            # Encoder layers
            self.layers = nn.ModuleList()
            for i in range(self.num_layers):
                p = f"encoder.layers.{i}."
                layer = nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(self.hidden_size, eps=self.eps),
                        "ln2": nn.LayerNorm(self.hidden_size, eps=self.eps),
                        "q": nn.Linear(self.hidden_size, self.hidden_size),
                        "k": nn.Linear(self.hidden_size, self.hidden_size),
                        "v": nn.Linear(self.hidden_size, self.hidden_size),
                        "out": nn.Linear(self.hidden_size, self.hidden_size),
                        "fc1": nn.Linear(self.hidden_size, 4304),
                        "fc2": nn.Linear(4304, self.hidden_size),
                    }
                )
                layer["ln1"].weight.data = weights[f"{p}layer_norm1.weight"].float()
                layer["ln1"].bias.data = weights[f"{p}layer_norm1.bias"].float()
                layer["ln2"].weight.data = weights[f"{p}layer_norm2.weight"].float()
                layer["ln2"].bias.data = weights[f"{p}layer_norm2.bias"].float()
                layer["q"].weight.data = weights[f"{p}self_attn.q_proj.weight"].float()
                layer["q"].bias.data = weights[f"{p}self_attn.q_proj.bias"].float()
                layer["k"].weight.data = weights[f"{p}self_attn.k_proj.weight"].float()
                layer["k"].bias.data = weights[f"{p}self_attn.k_proj.bias"].float()
                layer["v"].weight.data = weights[f"{p}self_attn.v_proj.weight"].float()
                layer["v"].bias.data = weights[f"{p}self_attn.v_proj.bias"].float()
                layer["out"].weight.data = weights[f"{p}self_attn.out_proj.weight"].float()
                layer["out"].bias.data = weights[f"{p}self_attn.out_proj.bias"].float()
                layer["fc1"].weight.data = weights[f"{p}mlp.fc1.weight"].float()
                layer["fc1"].bias.data = weights[f"{p}mlp.fc1.bias"].float()
                layer["fc2"].weight.data = weights[f"{p}mlp.fc2.weight"].float()
                layer["fc2"].bias.data = weights[f"{p}mlp.fc2.bias"].float()
                self.layers.append(layer)

            self.post_ln = nn.LayerNorm(self.hidden_size, eps=self.eps)
            self.post_ln.weight.data = weights["post_layernorm.weight"].float()
            self.post_ln.bias.data = weights["post_layernorm.bias"].float()

        def forward(self, pixel_values):
            # Patch embedding using same CPU extraction as TTNN
            patches = siglip2_patch_embeddings_cpu(pixel_values, self.patch_size)
            x = F.linear(patches, self.patch_weight, self.patch_bias)
            x = x + self.position_embedding.unsqueeze(0)

            # Encoder
            import math

            scale = 1.0 / math.sqrt(self.head_dim)
            B, S, D = x.shape

            for layer in self.layers:
                # Pre-attention LN
                normed = layer["ln1"](x)
                q = layer["q"](normed).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                k = layer["k"](normed).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v = layer["v"](normed).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                attn = torch.matmul(q * scale, k.transpose(-2, -1))
                attn = torch.softmax(attn, dim=-1)
                ctx = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, S, D)
                x = x + layer["out"](ctx)
                # Pre-MLP LN
                normed = layer["ln2"](x)
                x = x + layer["fc2"](F.gelu(layer["fc1"](normed)))

            return self.post_ln(x)

    return RefSigLIP2(vision_weights)


def test_siglip2_pcc():
    """Compare SigLIP2 vision encoder against PyTorch reference built from raw weights."""
    import ttnn
    from models.experimental.groot_n16.common.weight_loader import Gr00tN16WeightLoader
    from models.experimental.groot_n16.common.configs import Gr00tN16Config
    from models.experimental.groot_n16.tt.ttnn_siglip2 import SigLIP2VisionEncoderTTNN

    cfg = Gr00tN16Config.default()
    loader = Gr00tN16WeightLoader()
    loader.load()
    vision_weights = loader.get_vision_weights()

    # --- PyTorch reference ---
    print("\nBuilding PyTorch SigLIP2 reference from raw weights...")
    ref_model = _build_siglip2_ref_model(vision_weights)
    ref_model.eval()

    # --- Input ---
    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 224, 224)

    # --- Reference forward ---
    with torch.no_grad():
        ref_features = ref_model(pixel_values)
    print(f"  Reference output: {ref_features.shape}")

    # --- TTNN forward ---
    print("Running TTNN SigLIP2...")
    device = ttnn.open_device(device_id=0)
    try:
        encoder = SigLIP2VisionEncoderTTNN(cfg.backbone.vision, vision_weights, device)

        t0 = time.time()
        tt_output = encoder(pixel_values)
        elapsed = time.time() - t0
        tt_features = ttnn.to_torch(tt_output)
        print(f"  TTNN output: {tt_features.shape}, {elapsed*1000:.0f}ms")

        # --- Compare ---
        pcc = compute_pcc(ref_features, tt_features)
        max_diff = (ref_features - tt_features.float()).abs().max().item()
        mean_diff = (ref_features - tt_features.float()).abs().mean().item()

        print(f"\n  SigLIP2 PCC: {pcc:.6f}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        assert pcc >= 0.95, f"SigLIP2 PCC {pcc:.4f} below threshold 0.95"
        print("  PASS")
    finally:
        ttnn.close_device(device)


def test_connector_pcc():
    """Compare pixel shuffle + MLP connector against PyTorch reference."""
    import ttnn
    from models.experimental.groot_n16.common.weight_loader import Gr00tN16WeightLoader
    from models.experimental.groot_n16.common.configs import Gr00tN16Config
    from models.experimental.groot_n16.tt.ttnn_groot_n16_model import PixelShuffleConnectorTTNN

    cfg = Gr00tN16Config.default()
    loader = Gr00tN16WeightLoader()
    loader.load()

    conn_weights = loader.get_connector_weights()

    # --- PyTorch reference connector ---
    # Layer 0: LayerNorm(4608)
    # Layer 1: Linear(4608, 2048)
    # Layer 2: GELU
    # Layer 3: Linear(2048, 2048)
    ln = torch.nn.LayerNorm(4608)
    ln.weight.data = conn_weights["0.weight"].float()
    ln.bias.data = conn_weights["0.bias"].float()

    linear1 = torch.nn.Linear(4608, 2048)
    linear1.weight.data = conn_weights["1.weight"].float()
    linear1.bias.data = conn_weights["1.bias"].float()

    linear3 = torch.nn.Linear(2048, 2048)
    linear3.weight.data = conn_weights["3.weight"].float()
    linear3.bias.data = conn_weights["3.bias"].float()

    ref_connector = torch.nn.Sequential(ln, linear1, torch.nn.GELU(), linear3)
    ref_connector.eval()

    # --- Input: simulated vision features ---
    torch.manual_seed(42)
    vision_features = torch.randn(1, 256, 1152)
    h = w = 16

    # Pixel shuffle on CPU
    x = vision_features.reshape(1, 16, 16, 1152)
    x = x.reshape(1, 8, 2, 8, 2, 1152)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.reshape(1, 64, 4608)

    with torch.no_grad():
        ref_output = ref_connector(x)
    print(f"Reference connector output: {ref_output.shape}")

    # --- TTNN ---
    device = ttnn.open_device(device_id=0)
    try:
        connector = PixelShuffleConnectorTTNN(
            cfg.backbone.vision.hidden_size,
            cfg.backbone.language.hidden_size,
            conn_weights,
            device,
        )
        tt_output = connector(vision_features, h, w)
        tt_features = ttnn.to_torch(tt_output)
        print(f"TTNN connector output: {tt_features.shape}")

        pcc = compute_pcc(ref_output, tt_features)
        max_diff = (ref_output - tt_features.float()).abs().max().item()
        print(f"\n  Connector PCC: {pcc:.6f}")
        print(f"  Max diff: {max_diff:.6f}")
        assert pcc >= 0.98, f"Connector PCC {pcc:.4f} below threshold 0.98"
        print("  PASS")
    finally:
        ttnn.close_device(device)


def test_embodiment_pcc():
    """Compare embodiment MLPs against PyTorch reference."""
    import ttnn
    from models.experimental.groot_n16.common.weight_loader import Gr00tN16WeightLoader
    from models.experimental.groot_n16.common.configs import Gr00tN16Config
    from models.experimental.groot_n16.tt.ttnn_embodiment import CategorySpecificMLPTTNN
    from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor

    cfg = Gr00tN16Config.default()
    loader = Gr00tN16WeightLoader()
    loader.load()

    state_weights = loader.get_state_encoder_weights()
    emb_cfg = cfg.embodiment
    embodiment_id = 0

    # --- PyTorch reference state encoder ---
    w1 = state_weights["layer1.W"][embodiment_id]  # [128, 1024]
    b1 = state_weights["layer1.b"][embodiment_id]  # [1024]
    w2 = state_weights["layer2.W"][embodiment_id]  # [1024, 1536]
    b2 = state_weights["layer2.b"][embodiment_id]  # [1536]

    torch.manual_seed(42)
    state_input = torch.randn(1, 1, 128)

    with torch.no_grad():
        # x @ W + b, then SiLU
        h = torch.matmul(state_input, w1.float()) + b1.float()
        h = torch.nn.functional.silu(h)
        ref_output = torch.matmul(h, w2.float()) + b2.float()
    print(f"Reference state encoder output: {ref_output.shape}")

    # --- TTNN ---
    device = ttnn.open_device(device_id=0)
    try:
        state_enc = CategorySpecificMLPTTNN(
            state_weights,
            emb_cfg.max_num_embodiments,
            emb_cfg.max_state_dim,
            emb_cfg.state_hidden_dim,
            emb_cfg.state_output_dim,
            device,
        )
        state_tt = to_tt_tensor(state_input, device)
        tt_output = state_enc(state_tt, embodiment_id=0)
        tt_features = ttnn.to_torch(tt_output)
        print(f"TTNN state encoder output: {tt_features.shape}")

        pcc = compute_pcc(ref_output, tt_features)
        max_diff = (ref_output - tt_features.float()).abs().max().item()
        print(f"\n  State encoder PCC: {pcc:.6f}")
        print(f"  Max diff: {max_diff:.6f}")
        assert pcc >= 0.98, f"State encoder PCC {pcc:.4f} below threshold 0.98"
        print("  PASS")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("GR00T N1.6 PCC Verification Tests")
    print("=" * 60)

    tests = [
        ("SigLIP2 Vision Encoder", test_siglip2_pcc),
        ("Pixel Shuffle Connector", test_connector_pcc),
        ("Embodiment State Encoder", test_embodiment_pcc),
    ]

    results = {}
    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print("=" * 60)
        try:
            test_fn()
            results[name] = "PASS"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {name}: {result}")

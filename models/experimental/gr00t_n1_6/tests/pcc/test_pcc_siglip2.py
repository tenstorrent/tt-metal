# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC verification for SigLIP2 vision encoder.

Compares TTNN implementation against PyTorch reference built from raw weights.
Target: >= 0.95 PCC

Run:
    cd /home/ttuser/experiments/pi0/tt-metal
    export TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole
    pytest models/experimental/gr00t_n1_6/tests/pcc/test_pcc_siglip2.py -svv
"""

import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


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
    """Build a PyTorch reference SigLIP2 from raw weights."""
    import math

    from models.experimental.gr00t_n1_6.tt.ttnn_siglip2 import siglip2_patch_embeddings_cpu

    class RefSigLIP2(nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.patch_size = 14
            self.hidden_size = 1152
            self.num_heads = 16
            self.head_dim = self.hidden_size // self.num_heads
            self.num_layers = 27
            self.eps = 1e-6

            self.patch_weight = nn.Parameter(weights["embeddings.patch_embedding.weight"].float())
            self.patch_bias = nn.Parameter(weights["embeddings.patch_embedding.bias"].float())
            self.position_embedding = nn.Parameter(weights["embeddings.position_embedding.weight"].float())

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
            import math

            patches = siglip2_patch_embeddings_cpu(pixel_values, self.patch_size)
            x = F.linear(patches, self.patch_weight, self.patch_bias)
            x = x + self.position_embedding.unsqueeze(0)

            scale = 1.0 / math.sqrt(self.head_dim)
            B, S, D = x.shape

            for layer in self.layers:
                normed = layer["ln1"](x)
                q = layer["q"](normed).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                k = layer["k"](normed).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v = layer["v"](normed).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                attn = torch.matmul(q * scale, k.transpose(-2, -1))
                attn = torch.softmax(attn, dim=-1)
                ctx = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, S, D)
                x = x + layer["out"](ctx)
                normed = layer["ln2"](x)
                x = x + layer["fc2"](F.gelu(layer["fc1"](normed)))

            return self.post_ln(x)

    return RefSigLIP2(vision_weights)


@pytest.fixture(scope="module")
def tt_device():
    import ttnn

    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.fixture(scope="module")
def weight_loader():
    from models.experimental.gr00t_n1_6.common.weight_loader import Gr00tN16WeightLoader

    loader = Gr00tN16WeightLoader()
    loader.load()
    return loader


@pytest.fixture(scope="module")
def config():
    from models.experimental.gr00t_n1_6.common.configs import Gr00tN16Config

    return Gr00tN16Config.default()


def test_siglip2_pcc(config, weight_loader, tt_device):
    """Compare SigLIP2 vision encoder against PyTorch reference."""
    import ttnn
    from models.experimental.gr00t_n1_6.tt.ttnn_siglip2 import SigLIP2VisionEncoderTTNN

    vision_weights = weight_loader.get_vision_weights()

    # PyTorch reference
    ref_model = _build_siglip2_ref_model(vision_weights)
    ref_model.eval()

    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        ref_features = ref_model(pixel_values)

    # TTNN
    encoder = SigLIP2VisionEncoderTTNN(config.backbone.vision, vision_weights, tt_device)
    t0 = time.time()
    tt_output = encoder(pixel_values)
    elapsed = time.time() - t0
    tt_features = ttnn.to_torch(tt_output)

    pcc = compute_pcc(ref_features, tt_features)
    max_diff = (ref_features - tt_features.float()).abs().max().item()

    print(f"\n  SigLIP2 PCC: {pcc:.6f}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Latency: {elapsed*1000:.0f}ms")

    assert pcc >= 0.95, f"SigLIP2 PCC {pcc:.4f} below threshold 0.95"


def test_connector_pcc(config, weight_loader, tt_device):
    """Compare pixel shuffle + MLP connector against PyTorch reference."""
    import ttnn
    from models.experimental.gr00t_n1_6.tt.ttnn_groot_n16_model import PixelShuffleConnectorTTNN

    conn_weights = weight_loader.get_connector_weights()

    # PyTorch reference
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

    torch.manual_seed(42)
    vision_features = torch.randn(1, 256, 1152)

    # Pixel shuffle on CPU
    x = vision_features.reshape(1, 16, 16, 1152)
    x = x.reshape(1, 8, 2, 8, 2, 1152)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.reshape(1, 64, 4608)

    with torch.no_grad():
        ref_output = ref_connector(x)

    # TTNN
    connector = PixelShuffleConnectorTTNN(
        config.backbone.vision.hidden_size,
        config.backbone.language.hidden_size,
        conn_weights,
        tt_device,
    )
    tt_output = connector(vision_features, 16, 16)
    tt_features = ttnn.to_torch(tt_output)

    pcc = compute_pcc(ref_output, tt_features)
    max_diff = (ref_output - tt_features.float()).abs().max().item()

    print(f"\n  Connector PCC: {pcc:.6f}")
    print(f"  Max diff: {max_diff:.6f}")

    assert pcc >= 0.98, f"Connector PCC {pcc:.4f} below threshold 0.98"

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC verification for embodiment-specific MLPs.

Compares TTNN CategorySpecificMLP against PyTorch reference.
Target: >= 0.99 PCC

Run:
    cd /home/ttuser/experiments/pi0/tt-metal
    export TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole
    pytest models/experimental/gr00t_n1_6/tests/pcc/test_pcc_embodiment.py -svv
"""

import sys
from pathlib import Path

import pytest
import torch

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


def test_state_encoder_pcc(config, weight_loader, tt_device):
    """Compare state encoder MLP against PyTorch reference."""
    import ttnn
    from models.experimental.gr00t_n1_6.tt.ttnn_embodiment import CategorySpecificMLPTTNN
    from models.experimental.gr00t_n1_6.tt.ttnn_common import to_tt_tensor

    emb_cfg = config.embodiment
    state_weights = weight_loader.get_state_encoder_weights()
    embodiment_id = 0

    # PyTorch reference
    w1 = state_weights["layer1.W"][embodiment_id].float()
    b1 = state_weights["layer1.b"][embodiment_id].float()
    w2 = state_weights["layer2.W"][embodiment_id].float()
    b2 = state_weights["layer2.b"][embodiment_id].float()

    torch.manual_seed(42)
    state_input = torch.randn(1, 1, 128)

    with torch.no_grad():
        h = torch.nn.functional.silu(torch.matmul(state_input, w1) + b1)
        ref_output = torch.matmul(h, w2) + b2

    # TTNN
    state_enc = CategorySpecificMLPTTNN(
        state_weights,
        emb_cfg.max_num_embodiments,
        emb_cfg.max_state_dim,
        emb_cfg.state_hidden_dim,
        emb_cfg.state_output_dim,
        tt_device,
    )
    state_tt = to_tt_tensor(state_input, tt_device)
    tt_output = state_enc(state_tt, embodiment_id=0)
    tt_features = ttnn.to_torch(tt_output)

    pcc = compute_pcc(ref_output, tt_features)
    max_diff = (ref_output - tt_features.float()).abs().max().item()

    print(f"\n  State encoder PCC: {pcc:.6f}")
    print(f"  Max diff: {max_diff:.6f}")

    assert pcc >= 0.99, f"State encoder PCC {pcc:.4f} below threshold 0.99"


def test_action_encoder_pcc(config, weight_loader, tt_device):
    """Compare action encoder MLP against PyTorch reference."""
    import ttnn
    from models.experimental.gr00t_n1_6.tt.ttnn_embodiment import MultiEmbodimentActionEncoderTTNN
    from models.experimental.gr00t_n1_6.tt.ttnn_common import to_tt_tensor

    emb_cfg = config.embodiment
    action_weights = weight_loader.get_action_encoder_weights()
    embodiment_id = 0

    # PyTorch reference: W1 -> SiLU -> W2 -> + timestep_emb -> W3
    w1 = action_weights["W1.W"][embodiment_id].float()
    b1 = action_weights["W1.b"][embodiment_id].float()
    w2 = action_weights["W2.W"][embodiment_id].float()
    b2 = action_weights["W2.b"][embodiment_id].float()
    w3 = action_weights["W3.W"][embodiment_id].float()
    b3 = action_weights["W3.b"][embodiment_id].float()

    torch.manual_seed(42)
    action_input = torch.randn(1, 50, 128)
    timestep_emb = torch.randn(1, 1, 1536)

    with torch.no_grad():
        h = torch.nn.functional.silu(torch.matmul(action_input, w1) + b1)
        h = torch.matmul(h, w2) + b2 + timestep_emb
        ref_output = torch.matmul(h, w3) + b3

    # TTNN
    timestep_weights = weight_loader.get_timestep_encoder_weights()
    action_enc = MultiEmbodimentActionEncoderTTNN(
        action_weights,
        emb_cfg,
        timestep_weights,
        tt_device,
    )
    action_tt = to_tt_tensor(action_input, tt_device)
    ts_tt = to_tt_tensor(timestep_emb, tt_device)
    tt_output = action_enc(action_tt, ts_tt, embodiment_id=0)
    tt_features = ttnn.to_torch(tt_output)

    pcc = compute_pcc(ref_output, tt_features)
    max_diff = (ref_output - tt_features.float()).abs().max().item()

    print(f"\n  Action encoder PCC: {pcc:.6f}")
    print(f"  Max diff: {max_diff:.6f}")

    assert pcc >= 0.99, f"Action encoder PCC {pcc:.4f} below threshold 0.99"

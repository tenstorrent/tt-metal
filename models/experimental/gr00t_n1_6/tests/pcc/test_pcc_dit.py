# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC verification for AlternateVLDiT action head.

Compares TTNN implementation against upstream Isaac-GR00T reference.
Target: >= 0.95 PCC

Run:
    cd /home/ttuser/experiments/pi0/tt-metal
    export TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole
    pytest models/experimental/gr00t_n1_6/tests/pcc/test_pcc_dit.py -svv
"""

import importlib.util
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


def test_dit_pcc(config, weight_loader, tt_device):
    """Validate AlternateVLDiT against upstream Isaac-GR00T reference."""
    import ttnn
    from models.experimental.gr00t_n1_6.tt.ttnn_dit import AlternateVLDiTTTNN
    from models.experimental.gr00t_n1_6.tt.ttnn_embodiment import TimestepEncoderTTNN
    from models.experimental.gr00t_n1_6.tt.ttnn_common import to_tt_tensor

    torch.manual_seed(42)
    hidden = torch.randn(1, 51, 1536)
    backbone = torch.randn(1, 64, 2048)

    # TTNN DiT
    tt_dit = AlternateVLDiTTTNN(config.dit, weight_loader.get_dit_weights(), tt_device)
    ts_enc = TimestepEncoderTTNN(weight_loader.get_timestep_encoder_weights(), tt_device)
    timestep_emb = ts_enc(torch.tensor([0]))

    tt_out = ttnn.to_torch(
        tt_dit(to_tt_tensor(hidden, tt_device), timestep_emb, to_tt_tensor(backbone, tt_device))
    ).float()

    # Reference DiT from Isaac-GR00T
    dit_path = "/home/ttuser/experiments/gr00t_n16/Isaac-GR00T/gr00t/model/modules/dit.py"
    spec = importlib.util.spec_from_file_location("gr00t_dit", dit_path)
    dit_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dit_mod)

    ref_dit = dit_mod.AlternateVLDiT(
        num_layers=32,
        num_attention_heads=32,
        attention_head_dim=48,
        norm_type="ada_norm",
        dropout=0.0,
        final_dropout=True,
        output_dim=1024,
        interleave_self_attention=True,
        cross_attention_dim=2048,
        attend_text_every_n_blocks=2,
    )
    dit_sd = {
        k[len("action_head.model.") :]: v
        for k, v in weight_loader.state_dict.items()
        if k.startswith("action_head.model.")
    }
    ref_dit.load_state_dict(dit_sd, strict=False)
    ref_dit.float().eval()

    with torch.no_grad():
        ref_out = ref_dit(
            hidden_states=hidden,
            encoder_hidden_states=backbone,
            timestep=torch.tensor([0]),
            image_mask=torch.ones(1, 64, dtype=torch.bool),
            backbone_attention_mask=torch.ones(1, 64, dtype=torch.bool),
        )

    pcc = compute_pcc(ref_out, tt_out)
    print(f"\n  DiT PCC: {pcc:.6f}")
    print(f"  Ref range: [{ref_out.min():.4f}, {ref_out.max():.4f}]")
    print(f"  TTNN range: [{tt_out.min():.4f}, {tt_out.max():.4f}]")

    assert pcc >= 0.95, f"DiT PCC {pcc:.4f} below threshold 0.95"

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 1b — DiffusionHead PCC test.

Loads real prediction_head weights, runs reference PyTorch forward and TT forward,
asserts PCC >= 0.99.
"""

import sys
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import MODEL_PATH
from models.experimental.vibevoice.tt.load_weights import (
    load_vibevoice_state_dict,
    split_submodule_weights,
)
from models.experimental.vibevoice.tt.ttnn_diffusion_head import (
    preprocess_diffusion_head_weights,
    TTDiffusionHead,
)
from models.experimental.vibevoice.tt.vibevoice_config import load_vibevoice_model_config

# Need reference module on sys.path
_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytestmark = pytest.mark.skipif(not Path(MODEL_PATH).is_dir(), reason="VIBEVOICE_MODEL_PATH weights missing")


@pytest.fixture(scope="module")
def loaded_weights():
    state_dict = load_vibevoice_state_dict(MODEL_PATH)
    return split_submodule_weights(state_dict)


@pytest.fixture(scope="module")
def vv_config():
    return load_vibevoice_model_config(MODEL_PATH)


def _reference_diffusion_head_forward(
    state: dict,
    x: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor,
    hidden_size: int = 1536,
    latent_size: int = 64,
    head_ffn_ratio: float = 3.0,
    frequency_embedding_size: int = 256,
    norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Run reference VibeVoiceDiffusionHead forward using loaded weights."""
    from vibevoice.modular.configuration_vibevoice import VibeVoiceDiffusionHeadConfig
    from vibevoice.modular.modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead

    cfg = VibeVoiceDiffusionHeadConfig(
        hidden_size=hidden_size,
        head_layers=len([k for k in state if k.startswith("layers.") and k.endswith(".norm.weight")]),
        head_ffn_ratio=head_ffn_ratio,
        rms_norm_eps=norm_eps,
        latent_size=latent_size,
    )
    model = VibeVoiceDiffusionHead(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        out = model(x, t, condition)
    return out


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_diffusion_head_pcc(mesh_device, loaded_weights, vv_config):
    torch.manual_seed(0)

    state = loaded_weights["diffusion_head"]
    cfg = vv_config.diffusion_head

    B = 2  # CFG batch layout
    latent_size = cfg.latent_size
    hidden_size = cfg.hidden_size

    # Fixed inputs
    x_torch = torch.randn(B, latent_size, dtype=torch.bfloat16)
    t_torch = torch.tensor([500.0, 500.0], dtype=torch.bfloat16)
    cond_torch = torch.randn(B, hidden_size, dtype=torch.bfloat16)

    # 1) Reference PyTorch forward
    ref_out = _reference_diffusion_head_forward(
        state,
        x_torch.to(torch.float32),
        t_torch.to(torch.float32),
        cond_torch.to(torch.float32),
        hidden_size=hidden_size,
        latent_size=latent_size,
        head_ffn_ratio=cfg.head_ffn_ratio,
        norm_eps=cfg.rms_norm_eps,
    )  # [B, latent_size]

    # 2) TT forward
    weights = preprocess_diffusion_head_weights(
        state,
        mesh_device,
        hidden_size=hidden_size,
        latent_size=latent_size,
        head_ffn_ratio=cfg.head_ffn_ratio,
        norm_eps=cfg.rms_norm_eps,
        num_layers=cfg.head_layers,
    )
    head_tt = TTDiffusionHead(weights)

    # Reshape to [B, 1, 1, dim] for TTNN
    x_4d = x_torch.view(B, 1, 1, latent_size)
    t_4d = t_torch.view(B, 1, 1, 1)
    cond_4d = cond_torch.view(B, 1, 1, hidden_size)

    def to_tt(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.as_tensor(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x_tt = to_tt(x_4d)
    t_tt = to_tt(t_4d)
    cond_tt = to_tt(cond_4d)

    out_tt = head_tt(x_tt, t_tt, cond_tt)
    out_torch = ttnn.to_torch(out_tt).to(torch.float32).view(B, latent_size)

    passed, pcc_val = comp_pcc(ref_out.to(torch.float32), out_torch, pcc=0.99)
    assert passed, f"DiffusionHead PCC {pcc_val:.6f} < 0.99"

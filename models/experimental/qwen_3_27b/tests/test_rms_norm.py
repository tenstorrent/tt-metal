# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Component test for TtQwen36RmsNorm.

Testing zero-centered RMS norm with weight preparation.
The module folds `1 +` into the weight at load time, so the reference here takes
the RAW (un-offset) weight and applies `(1 + w)` itself.

Single chip for now. Expanding to 8xP150 means adding shapes to the mesh_device
parametrize below -- and, once the hidden dim is fractured across chips, the
module's forward has to switch to the pre_all_gather / all_gather /
post_all_gather trio, because RMS reduces over a dim no single chip owns.

Run:
    pytest models/experimental/qwen_3_27b/tests/test_rms_norm.py -v
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.qwen_3_27b.tt.tt_rms_norm import EPS, TtQwen36RmsNorm

PCC_THRESHOLD = 0.999

# The widths this norm is used at, and a representative input shape for each.
SHAPES = [
    (1, 32, 5120),  # hidden norms: per-layer input / post-attention, and the final norm
    (1, 1, 24, 256),  # q_norm / k_norm inside gated attention: 24 heads x head_dim 256
]
SHAPE_IDS = ["hidden_5120", "qk_norm_256"]


def torch_rms_norm(x, raw_weight, eps=EPS):
    """Zero-centered reference: note it applies (1 + w) to the RAW weight itself."""
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    return x_f * torch.rsqrt(variance + eps) * (1.0 + raw_weight.float())


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_rms_norm_pcc(mesh_device, shape, reset_seeds):
    dim = shape[-1]
    x = torch.randn(*shape, dtype=torch.bfloat16)
    # Zero-centered weight: initialized to zeros, trained to small deviations.
    raw_weight = (0.1 * torch.randn(dim)).to(torch.bfloat16)

    reference = torch_rms_norm(x, raw_weight)

    tt_model = TtQwen36RmsNorm(mesh_device, raw_weight)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    tt_out = ttnn.to_torch(tt_model(tt_x)).float().reshape(reference.shape)

    passing, pcc = comp_pcc(reference, tt_out, PCC_THRESHOLD)
    logger.info(f"rms_norm PCC {shape}: {pcc}")
    assert passing, f"rms_norm PCC below {PCC_THRESHOLD}: {pcc}"

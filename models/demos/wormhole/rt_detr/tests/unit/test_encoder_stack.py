# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import torch
import torch.nn as nn

import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from tt.rtdetr_encoder import run_encoder
from tt.weight_utils import get_tt_parameters

from models.common.utility_functions import comp_pcc


def test_transformer_encoder(device):
    torch.manual_seed(0)

    batch, seq_len, d_model, nhead = 1, 49, 256, 8

    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        ),
        num_layers=6,
    ).eval()

    x = torch.randn(batch, seq_len, d_model)

    with torch.no_grad():
        ref = encoder(x)

    tt_params = get_tt_parameters(device, encoder)
    out = run_encoder(x, tt_params.layers, device)

    pcc = comp_pcc(ref, out)
    pcc = pcc[1] if isinstance(pcc, tuple) else pcc
    print(f"Encoder PCC: {pcc}")
    assert pcc >= 0.99, f"pcc too low: {pcc:.4f}"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        test_transformer_encoder(device)
        print("passed")
    finally:
        ttnn.close_device(device)

# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.demos.informer.reference import TorchInformerModel
from models.demos.informer.tt import InformerConfig, InformerModel, to_torch


def test_torch_to_ttnn_roundtrip():
    torch.manual_seed(0)
    ttnn.CONFIG.throw_exception_on_fallback = True

    cfg = InformerConfig(
        enc_in=4,
        dec_in=4,
        c_out=4,
        seq_len=24,
        label_len=12,
        pred_len=6,
        d_model=64,
        n_heads=2,
        d_ff=128,
        time_feature_dim=4,
        dtype="bfloat16",
    )

    torch_model = TorchInformerModel(cfg)
    torch_model.eval()
    state = torch_model.state_dict_ttnn()

    device = ttnn.CreateDevice(device_id=0, l1_small_size=8192)
    try:
        ttnn_model = InformerModel(cfg, device=device, seed=0)
        ttnn_model.load_state_dict(state, strict=True)

        batch = 2
        past_values = torch.randn(batch, cfg.seq_len, cfg.enc_in, dtype=torch.float32)
        past_time = torch.randn(batch, cfg.seq_len, cfg.time_feature_dim, dtype=torch.float32)
        future_time = torch.randn(batch, cfg.pred_len, cfg.time_feature_dim, dtype=torch.float32)

        with torch.no_grad():
            torch_out = torch_model(past_values, past_time, future_time)
        ttnn_out = to_torch(ttnn_model(past_values, past_time, future_time)).float()

        assert ttnn_out.shape == torch_out.shape
        assert torch.isfinite(ttnn_out).all()
    finally:
        ttnn.CloseDevice(device)

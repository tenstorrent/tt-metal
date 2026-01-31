# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.demos.informer.tt import InformerConfig, InformerModel, to_torch


def test_ttnn_informer_output_shape():
    ttnn.CONFIG.throw_exception_on_fallback = True
    device = ttnn.CreateDevice(device_id=0, l1_small_size=8192)
    try:
        cfg = InformerConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=32,
            label_len=16,
            pred_len=8,
            d_model=64,
            n_heads=2,
            d_ff=128,
            time_feature_dim=4,
            dtype="bfloat16",
        )
        model = InformerModel(cfg, device=device, seed=0)
        batch = 2
        past_values = torch.randn(batch, cfg.seq_len, cfg.enc_in, dtype=torch.float32)
        past_time = torch.randn(batch, cfg.seq_len, cfg.time_feature_dim, dtype=torch.float32)
        future_time = torch.randn(batch, cfg.pred_len, cfg.time_feature_dim, dtype=torch.float32)
        out = model(past_values, past_time, future_time)
        out_torch = to_torch(out)
        assert out_torch.shape == (batch, cfg.pred_len, cfg.c_out)
        assert torch.isfinite(out_torch).all()
    finally:
        ttnn.CloseDevice(device)

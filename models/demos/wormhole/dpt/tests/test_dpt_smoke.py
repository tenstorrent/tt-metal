"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""

import torch
import ttnn
from transformers import DPTModel


def test_smoke_untraced():
    m = DPTModel.from_pretrained("Intel/dpt-large").eval()
    state = m.state_dict()
    device = ttnn.open_device(device_id=0)
    try:
        from models.experimental.dpt_large.tt_traced_pipeline import TracedDPTFull
        pipe = TracedDPTFull(state, device, batch_size=1, image_size=384)
        x = torch.randn(1, 3, 384, 384)
        out = pipe.forward_untraced(x)
        t = ttnn.to_torch(out)
        assert t.ndim in (3, 4)
    finally:
        ttnn.close_device(device)

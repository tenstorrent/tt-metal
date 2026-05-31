# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.liquid
def test_vision_encoder():
    from models.experimental.liquid.tt.vision_encoder import siglip2_encoder

    device = ttnn.open_device(device_id=0)
    batch, img_h, img_w, img_c = 1, 448, 448, 3
    pixel_values = ttnn.from_torch(torch.randn(batch, img_h, img_w, img_c), device=device)
    result = siglip2_encoder(pixel_values, parameters={})
    assert result is not None
    ttnn.close_device(device)


@pytest.mark.liquid
def test_liquid_vl_import():
    from models.experimental.liquid.tt.liquid_vl import LiquidVL

    assert LiquidVL is not None

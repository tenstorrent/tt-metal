# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test C2PSA module (model.10) for YOLO26.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE}],
    indirect=True,
)
def test_c2psa_model10(device):
    """Test C2PSA module with actual model.9 output as input."""
    from ultralytics import YOLO
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtC2PSA
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader
    from models.common.utility_functions import comp_pcc

    torch_model = YOLO("yolo26n.pt")
    state_dict = torch_model.model.state_dict()
    weight_loader = YOLO26WeightLoader(state_dict)

    batch_size = 1

    # Create TtC2PSA
    c2psa = TtC2PSA(device, in_channels=256, out_channels=256, n=1, name="model.10")
    c2psa.load_weights(weight_loader, "model.10")

    # Get model.9 output (SPPF output) as input to C2PSA
    torch.manual_seed(42)
    x_input = torch.rand(batch_size, 3, 640, 640, dtype=torch.float32)

    with torch.no_grad():
        x_pt = x_input
        for i in range(10):  # Run through model.0-9
            x_pt = torch_model.model.model[i](x_pt)

    logger.info(f"Model.9 output (C2PSA input): shape={x_pt.shape}, mean={x_pt.mean():.4f}")

    # Run PyTorch C2PSA (model.10)
    with torch.no_grad():
        pt_c2psa_out = torch_model.model.model[10](x_pt)
    logger.info(f"PyTorch C2PSA output: shape={pt_c2psa_out.shape}, mean={pt_c2psa_out.mean():.4f}")

    # Run TtC2PSA
    x_nhwc = x_pt.permute(0, 2, 3, 1).contiguous()
    tt_input = ttnn.from_torch(
        x_nhwc.to(torch.bfloat16), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    h, w = x_pt.shape[2], x_pt.shape[3]  # 20x20
    tt_out, out_h, out_w = c2psa(tt_input, batch_size, h, w)

    # Convert output
    if tt_out.memory_config().is_sharded():
        tt_out = ttnn.sharded_to_interleaved(tt_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
    tt_out = ttnn.reshape(tt_out, [batch_size, out_h, out_w, 256])
    tt_out_torch = ttnn.to_torch(tt_out)

    logger.info(f"TtC2PSA output: shape={tt_out_torch.shape}, mean={tt_out_torch.float().mean():.4f}")

    # Compare
    pt_out_nhwc = pt_c2psa_out.permute(0, 2, 3, 1).contiguous()
    passed, pcc = comp_pcc(pt_out_nhwc, tt_out_torch.float(), 0.90)
    logger.info(f"C2PSA PCC: {pcc:.4f} - {'PASS' if passed else 'FAIL'}")

    assert passed, f"C2PSA failed with PCC {pcc}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Backbone-only PCC test for YOLO26.

Tests just the backbone without neck/head to verify core functionality.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE


def get_torch_model(variant: str):
    """Load PyTorch YOLO26 model from Ultralytics."""
    try:
        from ultralytics import YOLO

        model = YOLO(f"{variant}.pt")
        model.eval()
        return model
    except ImportError:
        pytest.skip("ultralytics not installed. Run: pip install ultralytics")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLO26_L1_SMALL_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize("variant", ["yolo26n"])
def test_yolo26_backbone_only(device, variant):
    """
    Test YOLO26 backbone only.

    This verifies the Conv, C2f, and SPPF layers work correctly.
    """
    logger.info(f"Testing YOLO26 Backbone: variant={variant}")

    input_size = 640
    batch_size = 1

    # Load PyTorch model
    torch_model = get_torch_model(variant)
    state_dict = torch_model.model.state_dict()

    # Create random input - use bfloat16 like YUNet
    torch.manual_seed(42)
    input_tensor = torch.rand(batch_size, 3, input_size, input_size, dtype=torch.bfloat16)

    # Get PyTorch backbone outputs
    with torch.no_grad():
        # Access backbone layers directly
        model_layers = torch_model.model.model
        x = input_tensor.float()  # PyTorch needs float32

        # Run through backbone (layers 0-9)
        backbone_outputs = {}
        for i in range(10):
            x = model_layers[i](x)
            if i in [4, 6, 9]:  # P3, P4, P5 outputs
                backbone_outputs[i] = x.clone()
                logger.info(f"PyTorch layer {i} output shape: {x.shape}")

    # Create TTNN backbone
    from models.experimental.yolo26.tt.ttnn_yolo26 import TtYOLO26Backbone
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader

    weight_loader = YOLO26WeightLoader(state_dict)
    tt_backbone = TtYOLO26Backbone(device, variant)
    tt_backbone.load_weights(weight_loader)

    # Convert input to TTNN format (NHWC) - use ROW_MAJOR like YUNet!
    input_nhwc = input_tensor.permute(0, 2, 3, 1).contiguous()
    tt_input = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Run TTNN backbone
    p3_data, p4_data, p5_data = tt_backbone(tt_input)

    # Convert outputs back to torch - need proper reshaping
    def ttnn_to_torch(data_tuple):
        tensor, h, w = data_tuple
        if tensor.memory_config().is_sharded():
            tensor = ttnn.sharded_to_interleaved(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if tensor.layout == ttnn.TILE_LAYOUT:
            tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
        tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
        # Infer channels from tensor shape
        total_elements = tensor.volume()
        channels = total_elements // (batch_size * h * w)
        tensor = ttnn.reshape(tensor, [batch_size, h, w, channels])
        return ttnn.to_torch(tensor)

    p3_tt = ttnn_to_torch(p3_data)
    p4_tt = ttnn_to_torch(p4_data)
    p5_tt = ttnn_to_torch(p5_data)

    logger.info(f"TTNN P3 output shape: {p3_tt.shape}")
    logger.info(f"TTNN P4 output shape: {p4_tt.shape}")
    logger.info(f"TTNN P5 output shape: {p5_tt.shape}")

    # Convert PyTorch outputs to NHWC for comparison
    torch_p3 = backbone_outputs[4].permute(0, 2, 3, 1).contiguous()
    torch_p4 = backbone_outputs[6].permute(0, 2, 3, 1).contiguous()
    torch_p5 = backbone_outputs[9].permute(0, 2, 3, 1).contiguous()

    logger.info(f"PyTorch P3 shape (NHWC): {torch_p3.shape}")
    logger.info(f"PyTorch P4 shape (NHWC): {torch_p4.shape}")
    logger.info(f"PyTorch P5 shape (NHWC): {torch_p5.shape}")

    # Calculate PCC
    from models.common.utility_functions import comp_pcc

    pcc_threshold = 0.90  # Start with lower threshold for debugging

    p3_pass, pcc_p3 = comp_pcc(torch_p3, p3_tt.float(), pcc_threshold)
    logger.info(f"P3 PCC: {pcc_p3:.6f} - {'PASS' if p3_pass else 'FAIL'}")

    p4_pass, pcc_p4 = comp_pcc(torch_p4, p4_tt.float(), pcc_threshold)
    logger.info(f"P4 PCC: {pcc_p4:.6f} - {'PASS' if p4_pass else 'FAIL'}")

    p5_pass, pcc_p5 = comp_pcc(torch_p5, p5_tt.float(), pcc_threshold)
    logger.info(f"P5 PCC: {pcc_p5:.6f} - {'PASS' if p5_pass else 'FAIL'}")

    assert p3_pass, f"P3 PCC {pcc_p3:.4f} < {pcc_threshold}"
    assert p4_pass, f"P4 PCC {pcc_p4:.4f} < {pcc_threshold}"
    assert p5_pass, f"P5 PCC {pcc_p5:.4f} < {pcc_threshold}"

    logger.info("Backbone PCC test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

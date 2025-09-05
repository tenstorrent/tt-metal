import torch
import pytest
from typing import Dict
import ttnn

from models.experimental.panoptic_deeplab.reference.pytorch_semseg import PanopticDeepLabSemSegHead, ShapeSpec
from models.experimental.panoptic_deeplab.tt.tt_semseg import TtPanopticDeepLabSemSegHead
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
# Add this to your test file and run it
def test_ttnn_wholeSemSeg(device):
    compute_grid = device.compute_with_storage_grid_size()

    print(f"compute_grid: {compute_grid.x}x{compute_grid.y}")
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    batch_size = 1
    num_classes = 19
    final_output_size = (512, 1024)
    common_stride = 4

    target_project_channels = [32, 64]

    target_decoder_channels = [256, 256, 256]
    target_head_channels = 256

    # 2. Create ALL shared torch.Tensor weights for both models
    # --- Weights for the base decoder part (ASPP, fuse, project) ---
    w_aspp_k1 = torch.randn(256, 2048, 1, 1, dtype=torch.bfloat16)
    w_aspp_k3 = torch.randn(256, 2048, 3, 3, dtype=torch.bfloat16)
    # ASPP concat before final proj is 4*256(atrous) + 256(pool) = 1280 channels -> 256 out
    w_aspp_k1_out5 = torch.randn(256, 1280, 1, 1, dtype=torch.bfloat16)

    w_res3_proj = torch.randn(64, 512, 1, 1, dtype=torch.bfloat16)
    w_res3_fuse0 = torch.randn(256, 320, 3, 3, dtype=torch.bfloat16)
    w_res3_fuse1 = torch.randn(256, 256, 3, 3, dtype=torch.bfloat16)

    w_res2_proj = torch.randn(32, 256, 1, 1, dtype=torch.bfloat16)
    w_res2_fuse0 = torch.randn(256, 288, 3, 3, dtype=torch.bfloat16)
    w_res2_fuse1 = torch.randn(256, 256, 3, 3, dtype=torch.bfloat16)

    project_conv_weights = {"res2": w_res2_proj, "res3": w_res3_proj}
    fuse_conv_0_weights = {"res2": w_res2_fuse0, "res3": w_res3_fuse0}
    fuse_conv_1_weights = {"res2": w_res2_fuse1, "res3": w_res3_fuse1}

    w_panoptic_head_0 = torch.randn(256, 256, 3, 3, dtype=torch.bfloat16)
    w_panoptic_head_1 = torch.randn(target_head_channels, 256, 3, 3, dtype=torch.bfloat16)

    w_panoptic_predictor = torch.randn(num_classes, target_head_channels, 1, 1, dtype=torch.bfloat16)

    # 3. Prepare PyTorch Model and Inputs
    # --- Create PyTorch inputs and ShapeSpec ---
    torch_features: Dict[str, torch.Tensor] = {
        "res2": torch.randn(1, 256, 128, 256, dtype=torch.bfloat16),
        "res3": torch.randn(1, 512, 64, 128, dtype=torch.bfloat16),
        "res5": torch.randn(1, 2048, 32, 64, dtype=torch.bfloat16),
    }
    res2_shape = ShapeSpec()
    res2_shape.channels = 256
    res2_shape.stride = 4
    res3_shape = ShapeSpec()
    res3_shape.channels = 512
    res3_shape.stride = 8
    res5_shape = ShapeSpec()
    res5_shape.channels = 2048
    res5_shape.stride = 16

    input_shape_pytorch: Dict[str, ShapeSpec] = {
        "res2": res2_shape,
        "res3": res3_shape,
        "res5": res5_shape,
    }
    # --- Instantiate and run PyTorch model ---
    torch_model = PanopticDeepLabSemSegHead(
        input_shape=input_shape_pytorch,
        project_channels=target_project_channels,
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.0,
        decoder_channels=target_decoder_channels,
        common_stride=common_stride,
        train_size=final_output_size,
        norm="SyncBN",
        head_channels=target_head_channels,
        loss_weight=1.0,
        loss_type="cross_entropy",
        loss_top_k=0.2,
        ignore_value=255,
        num_classes=num_classes,
        use_depthwise_separable_conv=False,
        shared_weight_tensor_kernel1=w_aspp_k1,
        shared_weight_tensor_kernel3=w_aspp_k3,
        shared_weight_tensor_kernel1_output5=w_aspp_k1_out5,
        project_conv_weights=project_conv_weights,
        fuse_conv_0_weights=fuse_conv_0_weights,
        fuse_conv_1_weights=fuse_conv_1_weights,
        panoptic_head_0_weight=w_panoptic_head_0,
        panoptic_head_1_weight=w_panoptic_head_1,
        panoptic_predictor_weight=w_panoptic_predictor,
    )

    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    # 4. Prepare TTNN Model and Inputs
    # --- Create TTNN inputs and ShapeSpec ---
    ttnn_features: Dict[str, ttnn.Tensor] = {}
    for name, tensor in torch_features.items():
        ttnn_features[name] = ttnn.from_torch(
            tensor.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16  # NCHW -> NHWC
        )
    input_shape_ttnn: Dict[str, ShapeSpec] = {
        "res2": res2_shape,
        "res3": res3_shape,
        "res5": res5_shape,
    }
    ttnn_model = TtPanopticDeepLabSemSegHead(
        input_shape=input_shape_ttnn,
        device=device,
        project_channels=target_project_channels,
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.0,
        decoder_channels=target_decoder_channels,
        common_stride=common_stride,
        train_size=final_output_size,
        norm="SyncBN",
        head_channels=target_head_channels,
        num_classes=num_classes,
        shared_weight_tensor_kernel1=w_aspp_k1,
        shared_weight_tensor_kernel3=w_aspp_k3,
        shared_weight_tensor_kernel1_output5=w_aspp_k1_out5,
        project_conv_weights=project_conv_weights,
        fuse_conv_0_weights=fuse_conv_0_weights,
        fuse_conv_1_weights=fuse_conv_1_weights,
        panoptic_head_0_weight=w_panoptic_head_0,
        panoptic_head_1_weight=w_panoptic_head_1,
        panoptic_predictor_weight=w_panoptic_predictor,
    )

    torch_out, _ = torch_model(torch_features)

    ttnn_out_tt, _ = ttnn_model(ttnn_features)

    ttnn_out_torch = ttnn.to_torch(ttnn_out_tt).permute(0, 3, 1, 2)  # NHWC -> NCHW

    passed, msg = assert_with_pcc(torch_out, ttnn_out_torch, pcc=0.98)
    print(f"PCC Result: {msg}")

    assert passed, f"Comparison FAILED : {msg}"
    print(f"✅ TEST PASSED")

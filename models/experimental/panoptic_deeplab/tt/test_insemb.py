import pytest
import torch
from typing import Dict

from .tt_pytorch_insemb import PanopticDeepLabInsEmbedHead
from .tt_pytorch_semSeg import ShapeSpec

from .tt_insemb import TtPanopticDeepLabInsEmbedHead

from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_ttnn_wholeInsEmbed(device):
    compute_grid = device.compute_with_storage_grid_size()

    print(f"compute_grid: {compute_grid.x}x{compute_grid.y}")
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    target_project_channels = [32, 64]
    target_decoder_channels = [256, 256, 256]
    target_head_channels = 32

    w_aspp_k1 = torch.randn(256, 2048, 1, 1, dtype=torch.bfloat16)
    w_aspp_k3 = torch.randn(256, 2048, 3, 3, dtype=torch.bfloat16)
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

    decoder_output_channels = target_decoder_channels[0]

    # Center Head branch
    w_center_head_0 = torch.randn(decoder_output_channels, decoder_output_channels, 3, 3, dtype=torch.bfloat16)
    w_center_head_1 = torch.randn(target_head_channels, decoder_output_channels, 3, 3, dtype=torch.bfloat16)
    w_center_predictor = torch.randn(1, target_head_channels, 1, 1, dtype=torch.bfloat16)  # Izlaz je 1 kanal

    # Offset Head branch
    w_offset_head_0 = torch.randn(decoder_output_channels, decoder_output_channels, 3, 3, dtype=torch.bfloat16)
    w_offset_head_1 = torch.randn(target_head_channels, decoder_output_channels, 3, 3, dtype=torch.bfloat16)
    w_offset_predictor = torch.randn(2, target_head_channels, 1, 1, dtype=torch.bfloat16)  # Izlaz su 2 kanala

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

    torch_model = PanopticDeepLabInsEmbedHead(
        input_shape=input_shape_pytorch,
        project_channels=target_project_channels,
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.0,
        decoder_channels=target_decoder_channels,
        common_stride=4,
        train_size=(512, 1024),
        norm="SyncBN",
        head_channels=target_head_channels,
        center_loss_weight=200.0,
        offset_loss_weight=0.01,
        use_depthwise_separable_conv=False,
        shared_weight_tensor_kernel1=w_aspp_k1,
        shared_weight_tensor_kernel3=w_aspp_k3,
        shared_weight_tensor_kernel1_output5=w_aspp_k1_out5,
        project_conv_weights=project_conv_weights,
        fuse_conv_0_weights=fuse_conv_0_weights,
        fuse_conv_1_weights=fuse_conv_1_weights,
        center_head_0_weight=w_center_head_0,
        center_head_1_weight=w_center_head_1,
        center_predictor_weight=w_center_predictor,
        offset_head_0_weight=w_offset_head_0,
        offset_head_1_weight=w_offset_head_1,
        offset_predictor_weight=w_offset_predictor,
    )
    torch_model = torch_model.to(dtype=torch.bfloat16)
    torch_model.eval()

    ttnn_features: Dict[str, ttnn.Tensor] = {
        name: ttnn.from_torch(tensor.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        for name, tensor in torch_features.items()
    }

    ttnn_model = TtPanopticDeepLabInsEmbedHead(
        input_shape=input_shape_pytorch,
        device=device,
        project_channels=target_project_channels,
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.0,
        decoder_channels=target_decoder_channels,
        common_stride=4,
        train_size=(512, 1024),
        norm="SyncBN",
        head_channels=target_head_channels,
        shared_weight_tensor_kernel1=w_aspp_k1,
        shared_weight_tensor_kernel3=w_aspp_k3,
        shared_weight_tensor_kernel1_output5=w_aspp_k1_out5,
        project_conv_weights=project_conv_weights,
        fuse_conv_0_weights=fuse_conv_0_weights,
        fuse_conv_1_weights=fuse_conv_1_weights,
        center_head_0_weight=w_center_head_0,
        center_head_1_weight=w_center_head_1,
        center_predictor_weight=w_center_predictor,
        offset_head_0_weight=w_offset_head_0,
        offset_head_1_weight=w_offset_head_1,
        offset_predictor_weight=w_offset_predictor,
    )

    torch_center_out, torch_offset_out, _, _ = torch_model(torch_features)

    ttnn_center_out_tt, ttnn_offset_out_tt, _, _ = ttnn_model(ttnn_features)

    print("\n--- Comparing Center Prediction ---")
    ttnn_center_out_torch = ttnn.to_torch(ttnn_center_out_tt).permute(0, 3, 1, 2)
    passed_center, msg_center = assert_with_pcc(torch_center_out, ttnn_center_out_torch, pcc=0.96)
    print(f"PCC Result (Center): {msg_center}")
    assert passed_center, f"Center comparison FAILED: {msg_center}"
    print("✅ Center Prediction PASSED")

    print("\n--- Comparing Offset Prediction ---")
    ttnn_offset_out_torch = ttnn.to_torch(ttnn_offset_out_tt).permute(0, 3, 1, 2)
    passed_offset, msg_offset = assert_with_pcc(torch_offset_out, ttnn_offset_out_torch, pcc=0.97)
    print(f"PCC Result (Offset): {msg_offset}")
    assert passed_offset, f"Offset comparison FAILED: {msg_offset}"
    print("✅ Offset Prediction PASSED")

    print("\n✅✅✅ TEST PASSED ✅✅✅")

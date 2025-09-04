import torch
import pytest
from typing import Dict
import ttnn

# Assume your model classes are in a file named `your_model_file.py`
# If they are in the same file, you can remove the next line.
from .tt_pytorch_semSeg import DeepLabV3PlusHead, PanopticDeepLabSemSegHead, ShapeSpec
from .tt_semseg import TtPanopticDeepLabSemSegHead
from tests.ttnn.utils_for_testing import assert_with_pcc

# --- Test for DeepLabV3PlusHead ---


def test_deeplabv3():
    """
    Tests DeepLabV3PlusHead with a specific, realistic input/output scenario.
    - Input: 3 multi-scale feature maps from a backbone.
    - Output: A final logit map of a specific size and class count.
    This test validates that the model's internal layers (ASPP, project, fuse)
    are configured correctly and that the final output shape is as expected.
    """
    torch.manual_seed(0)

    # 1. Define the EXACT input and output parameters from your scenario
    batch_size = 1

    final_output_size = (512, 1024)  # H, W of the final prediction
    common_stride = 4  # The stride of the highest-resolution feature map

    # --- Create the mock input Tensors and ShapeSpec dictionary ---
    # These are created manually to match your exact input shapes.

    res2 = torch.randn(batch_size, 256, 128, 256)
    res3 = torch.randn(batch_size, 512, 64, 128)
    res5 = torch.randn(batch_size, 2048, 32, 64)

    features: Dict[str, torch.Tensor] = {
        "res2": res2,
        "res3": res3,
        "res5": res5,
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

    input_shape: Dict[str, ShapeSpec] = {
        "res2": res2_shape,
        "res3": res3_shape,
        "res5": res5_shape,
    }

    # 2. Instantiate the DeepLabV3PlusHead
    # We deduce the necessary parameters to handle the given inputs and produce the desired output.
    # A standard DeepLabV3+ configuration is used for decoder/project channels.

    # NOTE: Your __init__ requires shared weight tensors. We create dummy ones for this test.
    # You will need to pass the real ones during actual model creation.

    w_aspp_k1 = torch.randn(256, 2048, 1, 1)
    w_aspp_k3 = torch.randn(256, 2048, 3, 3)
    w_aspp_k1_out5 = torch.randn(256, 1280, 1, 1)  # 1280 = 5 * 256

    # --- Weights for Decoder Stages (Refined list) ---
    w_shared_fuse0 = torch.randn(256, 304, 3, 3)  # 304 = 48 + 256
    w_shared_fuse1 = torch.randn(256, 256, 3, 3)
    w_res3_proj = torch.randn(48, 512, 1, 1)
    w_res2_proj = torch.randn(48, 256, 1, 1)

    # --- Weight for Predictor ---
    # w_predictor = torch.randn(19, 256, 1, 1) We do decoder_only = True

    model = DeepLabV3PlusHead(
        input_shape=input_shape,
        # len(in_features) - 1 = 2 project channels needed
        project_channels=[48, 48],
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.1,
        # len(in_features) = 3 decoder channels needed
        decoder_channels=[256, 256, 256],
        common_stride=common_stride,
        norm="SyncBN",
        # train_size should match the final output for correct ASPP global pooling
        train_size=final_output_size,
        num_classes=None,
        # Pass the required dummy shared weights
        shared_weight_tensor_kernel1=w_aspp_k1,
        shared_weight_tensor_kernel3=w_aspp_k3,
        shared_weight_tensor_kernel1_output5=w_aspp_k1_out5,
        shared_fuse_conv_0_weight=w_shared_fuse0,
        shared_fuse_conv_1_weight=w_shared_fuse1,
        res3_project_conv_weight=w_res3_proj,
        res2_project_conv_weight=w_res2_proj,
        predictor_weight=None,
    )
    model.eval()  # Set to evaluation mode

    # 3. Run the forward pass
    # In inference mode, the forward pass returns (predictions, {})
    predictions = model(features)

    # 4. Perform Checks
    assert predictions is not None
    assert isinstance(predictions, torch.Tensor)

    # The final assertion validates the output shape against your specified target
    expected_channels = 256  # from decoder_channels[0]
    expected_height = 128  # from res2 input
    expected_width = 256  # from res2 input

    expected_shape = (batch_size, expected_channels, expected_height, expected_width)
    assert (
        predictions.shape == expected_shape
    ), f"Shape mismatch! Expected {expected_shape}, but got {predictions.shape}"


# --- Test for PanopticDeepLabSemSegHead ---


def test_semSeg():
    """
    Tests the complete PanopticDeepLabSemSegHead based on the real-world usage.
    It verifies the full pipeline:
    1. The base DeepLabV3PlusHead runs in decoder_only mode.
    2. The Panoptic head's specific layers are applied.
    3. The final output is a correctly shaped logit map.
    """
    torch.manual_seed(0)

    # 1. Define the same input and output parameters as the real scenario
    batch_size = 1
    num_classes = 19
    final_output_size = (512, 1024)
    common_stride = 4

    res2 = torch.randn(batch_size, 256, 128, 256)
    res3 = torch.randn(batch_size, 512, 64, 128)
    res5 = torch.randn(batch_size, 2048, 32, 64)

    # --- Use the same mock input features and shapes ---
    features: Dict[str, torch.Tensor] = {
        "res2": res2,
        "res3": res3,
        "res5": res5,
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

    input_shape: Dict[str, ShapeSpec] = {
        "res2": res2_shape,
        "res3": res3_shape,
        "res5": res5_shape,
    }

    # 2. Create ALL necessary weights for the entire pipeline
    # --- Weights for the base decoder part (ASPP, fuse, project) ---

    w_aspp_k1 = torch.randn(256, 2048, 1, 1)
    w_aspp_k3 = torch.randn(256, 2048, 3, 3)
    w_aspp_k1_out5 = torch.randn(256, 1280, 1, 1)
    w_shared_fuse0 = torch.randn(256, 304, 3, 3)
    w_shared_fuse1 = torch.randn(256, 256, 3, 3)
    w_res3_proj = torch.randn(48, 512, 1, 1)
    w_res2_proj = torch.randn(48, 256, 1, 1)

    # --- NEW: Weights for the Panoptic-specific head and predictor ---
    # Assuming head_channels=128
    w_panoptic_head_0 = torch.randn(256, 256, 3, 3)
    w_panoptic_head_1 = torch.randn(128, 256, 3, 3)
    w_panoptic_predictor = torch.randn(19, 128, 1, 1)

    # 3. Instantiate the PanopticDeepLabSemSegHead
    # NOTE: You will need to modify the Panoptic... __init__ to accept these weights
    model = PanopticDeepLabSemSegHead(
        # --- Parameters that will be passed down to the base class ---
        input_shape=input_shape,
        project_channels=[48, 48],
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.1,
        decoder_channels=[256, 256, 256],
        common_stride=common_stride,
        train_size=final_output_size,
        use_depthwise_separable_conv=False,
        # --- Parameters for this specific Panoptic head ---
        norm="SyncBN",
        head_channels=128,
        loss_weight=1.0,
        loss_type="cross_entropy",
        loss_top_k=0.2,
        ignore_value=255,
        num_classes=num_classes,
        # --- All the weights needed for deterministic testing ---
        # Base decoder weights
        shared_weight_tensor_kernel1=w_aspp_k1,
        shared_weight_tensor_kernel3=w_aspp_k3,
        shared_weight_tensor_kernel1_output5=w_aspp_k1_out5,
        shared_fuse_conv_0_weight=w_shared_fuse0,
        shared_fuse_conv_1_weight=w_shared_fuse1,
        res3_project_conv_weight=w_res3_proj,
        res2_project_conv_weight=w_res2_proj,
        # Panoptic-specific weights
        panoptic_head_0_weight=w_panoptic_head_0,
        panoptic_head_1_weight=w_panoptic_head_1,
        panoptic_predictor_weight=w_panoptic_predictor,
    )
    model.eval()

    # 4. Run the forward pass
    # Panoptic head is in inference mode, so it returns (predictions, {})
    predictions, _ = model(features)

    # 5. Perform Checks
    assert predictions is not None
    assert isinstance(predictions, torch.Tensor)
    # The final assertion checks for the full-size logit map
    expected_shape = (batch_size, num_classes, final_output_size[0], final_output_size[1])
    assert (
        predictions.shape == expected_shape
    ), f"Shape mismatch! Expected {expected_shape}, but got {predictions.shape}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_ttnn_semSeg(device):
    """
    Compares the TTNN implementation of PanopticDeepLabSemSegHead against the
    PyTorch reference model to ensure numerical consistency.
    """
    torch.manual_seed(0)

    # 1. Define Model Configuration
    batch_size = 1
    num_classes = 19
    final_output_size = (512, 1024)
    common_stride = 4

    # 2. Create ALL shared torch.Tensor weights for both models
    # --- Weights for the base decoder part (ASPP, fuse, project) ---
    w_aspp_k1 = torch.randn(256, 2048, 1, 1, dtype=torch.bfloat16)
    w_aspp_k3 = torch.randn(256, 2048, 3, 3, dtype=torch.bfloat16)
    w_aspp_k1_out5 = torch.randn(256, 1280, 1, 1, dtype=torch.bfloat16)
    w_shared_fuse0 = torch.randn(256, 304, 3, 3, dtype=torch.bfloat16)
    w_shared_fuse1 = torch.randn(256, 256, 3, 3, dtype=torch.bfloat16)
    w_res3_proj = torch.randn(48, 512, 1, 1, dtype=torch.bfloat16)
    w_res2_proj = torch.randn(48, 256, 1, 1, dtype=torch.bfloat16)
    # --- Weights for the Panoptic-specific head and predictor ---
    w_panoptic_head_0 = torch.randn(256, 256, 3, 3, dtype=torch.bfloat16)
    w_panoptic_head_1 = torch.randn(128, 256, 3, 3, dtype=torch.bfloat16)
    w_panoptic_predictor = torch.randn(19, 128, 1, 1, dtype=torch.bfloat16)

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
    # Create weight dictionaries that match the current implementation
    # We need different fusion weights for res2 and res3 stages with correct input channel dimensions
    w_res2_fuse0 = torch.randn(256, 304, 3, 3, dtype=torch.bfloat16)  # 304 = 48 (res2 proj) + 256 (from res3)
    w_res2_fuse1 = torch.randn(256, 256, 3, 3, dtype=torch.bfloat16)
    w_res3_fuse0 = torch.randn(256, 304, 3, 3, dtype=torch.bfloat16)  # 304 = 48 (res3 proj) + 256 (from ASPP)
    w_res3_fuse1 = torch.randn(256, 256, 3, 3, dtype=torch.bfloat16)

    project_conv_weights = {"res2": w_res2_proj, "res3": w_res3_proj}
    fuse_conv_0_weights = {"res2": w_res2_fuse0, "res3": w_res3_fuse0}
    fuse_conv_1_weights = {"res2": w_res2_fuse1, "res3": w_res3_fuse1}

    # --- Instantiate and run PyTorch model ---
    torch_model = PanopticDeepLabSemSegHead(
        input_shape=input_shape_pytorch,
        project_channels=[48, 48],
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.0,  # Disable dropout for comparison
        decoder_channels=[256, 256, 256],
        common_stride=common_stride,
        train_size=final_output_size,
        norm="SyncBN",
        head_channels=128,
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
    torch_output, _ = torch_model(torch_features)
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
    # --- Instantiate and run TTNN model ---
    ttnn_model = TtPanopticDeepLabSemSegHead(
        input_shape=input_shape_ttnn,
        device=device,
        project_channels=[48, 48],
        aspp_dilations=[6, 12, 18],
        aspp_dropout=0.0,
        decoder_channels=[256, 256, 256],
        common_stride=common_stride,
        train_size=final_output_size,
        norm="SyncBN",
        head_channels=128,
        num_classes=num_classes,
        # Pass all the same shared weights
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
    ttnn_output_tt, _ = ttnn_model(ttnn_features)
    # 5. Convert TTNN output and Compare
    ttnn_output_torch = ttnn.to_torch(ttnn_output_tt)
    ttnn_output_torch = ttnn_output_torch.permute(0, 3, 1, 2)  # NHWC -> NCHW

    # Basic shape check
    assert (
        torch_output.shape == ttnn_output_torch.shape
    ), f"Shape mismatch: PyTorch is {torch_output.shape}, TTNN is {ttnn_output_torch.shape}"

    # Numerical consistency check
    pcc_passed, pcc_message = assert_with_pcc(torch_output, ttnn_output_torch, pcc=0.99)
    print(f"PCC: {pcc_message}")
    assert pcc_passed, f"PCC check failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
# Add this to your test file and run it
def test_ttnn_wholeSemSeg(device):
    compute_grid = device.compute_with_storage_grid_size()

    print(f"compute_grid: {compute_grid.x}x{compute_grid.y}")
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)

    # 1. Define Model Configuration
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

    # torch_output, _ = torch_model(torch_features)

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

    # Run PyTorch model for this stage
    torch_out, _ = torch_model(torch_features)

    # Run TTNN model for this stage
    ttnn_out_tt, _ = ttnn_model(ttnn_features)

    # Convert and compare
    ttnn_out_torch = ttnn.to_torch(ttnn_out_tt).permute(0, 3, 1, 2)  # NHWC -> NCHW

    passed, msg = assert_with_pcc(torch_out, ttnn_out_torch, pcc=0.98)
    print(f"PCC Result: {msg}")

    assert passed, f"Comparison FAILED : {msg}"
    print(f"✅ TEST PASSED")

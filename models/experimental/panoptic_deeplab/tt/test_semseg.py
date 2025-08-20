import torch
from typing import Dict

# Assume your model classes are in a file named `your_model_file.py`
# If they are in the same file, you can remove the next line.
from .tt_pytorch_semSeg import DeepLabV3PlusHead, PanopticDeepLabSemSegHead, ShapeSpec

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
        norm="LN",
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
        norm="LN",
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

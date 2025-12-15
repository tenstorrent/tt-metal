import ttnn
import tests.ttnn.unit_tests.operations.conv2d_bn2d.utils as utils
import torch
import torch.nn as nn
from loguru import logger
from pytorchcv.model_provider import get_model as ptcv_get_model


def ttnn_conv_cpu_bn(input_torch, conv_weights_torch, cpu_bn_layer):
    device = utils.DeviceGetter.get_device((1, 1))

    logger.info(f"Input shape (NCHW): {input_torch.shape}")

    # ========================================================================
    # CPU PREPROCESSING for Conv2d (using torch)
    # ========================================================================
    logger.info("CPU preprocessing: NCHW -> flattened format for conv2d...")
    # Permute from NCHW [1, 32, 112, 112] to NHWC [1, 112, 112, 32]
    input_nhwc = input_torch.permute(0, 2, 3, 1)
    # Reshape to flatten spatial dimensions [1, 1, H*W, C] = [1, 1, 12544, 32]
    input_flattened = input_nhwc.reshape(1, 1, 12544, 32)
    logger.info(f"After CPU preprocessing: {input_flattened.shape}")

    # ========================================================================
    # DEVICE OPERATION: Conv2d (ttnn)
    # ========================================================================
    logger.info("Transferring input to device for conv2d...")
    # Transfer to device
    input_ttnn = ttnn.from_torch(
        input_flattened,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    conv_weights_ttnn = ttnn.from_torch(
        conv_weights_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    logger.info("Running ttnn.conv2d on device...")
    conv_output_ttnn = ttnn.conv2d(
        input_tensor=input_ttnn,
        weight_tensor=conv_weights_ttnn,
        device=device,
        in_channels=32,
        out_channels=64,
        batch_size=1,
        input_height=112,
        input_width=112,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    logger.info(f"Conv output shape on device: {conv_output_ttnn.shape}")

    # Transfer back to CPU
    logger.info("Transferring conv output back to CPU...")
    conv_output_torch = ttnn.to_torch(conv_output_ttnn)

    # ========================================================================
    # CPU POSTPROCESSING for Conv2d (using torch)
    # ========================================================================
    logger.info("CPU postprocessing: flattened -> NCHW format...")
    # Reshape from [1, 1, 12544, 64] to [1, 112, 112, 64]
    conv_reshaped = conv_output_torch.reshape(1, 112, 112, 64)
    # Permute from NHWC [1, 112, 112, 64] to NCHW [1, 64, 112, 112]
    conv_nchw = conv_reshaped.permute(0, 3, 1, 2)
    logger.info(f"After CPU postprocessing (NCHW): {conv_nchw.shape}")

    # ========================================================================
    # CPU OPERATION: Batch Norm (PyTorch on CPU)
    # ========================================================================
    logger.info("Running CPU batch_norm (NOT on device)...")
    logger.info(">>> Using PyTorch batch_norm implementation <<<")

    with torch.no_grad():
        bn_output_torch = cpu_bn_layer(conv_nchw)

    logger.info(f"Batch norm output shape (CPU): {bn_output_torch.shape}")

    return bn_output_torch


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.s1_p_conv = model.encoder.backbone.stage1.unit1.pw_conv.conv
        self.s1_p_bn = model.encoder.backbone.stage1.unit1.pw_conv.bn

    def forward(self, x):
        x = self.s1_p_conv(x)
        x = self.s1_p_bn(x)
        return x


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_float = x.to(torch.float32) if x.dtype != torch.float32 else x
    y_float = y.to(torch.float32) if y.dtype != torch.float32 else y

    x_flat, y_flat = x_float.flatten(), y_float.flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    if denom == 0:
        return float("nan")
    else:
        return ((vx @ vy) / denom).item()


def test_conv2d_tt_bn2d_cpu():
    # Load CPU model for batch_norm and reference
    model = ptcv_get_model("lwopenpose2d_mobilenet_cmupan_coco", pretrained=True).to(torch.bfloat16)
    cpu_model = Wrapper(model)
    cpu_model.eval()

    # Load tensors (on CPU as torch tensors)
    logger.info("\nLoading tensors on CPU...")
    conv_weights_ttnn = ttnn.load_tensor("./tensors/arg4.tensorbin")
    input_ttnn = ttnn.load_tensor("./tensors/arg5.tensorbin")

    # Convert to torch immediately
    conv_weights_torch = ttnn.to_torch(conv_weights_ttnn)
    input_torch = ttnn.to_torch(input_ttnn)

    logger.info(f"Input shape: {input_torch.shape}")
    logger.info(f"Conv weights shape: {conv_weights_torch.shape}")

    # Run TT conv2d + CPU batch_norm
    logger.info("\n" + "=" * 80)
    logger.info("Running TT conv2d → CPU batch_norm...")
    logger.info("=" * 80)
    tt_output = ttnn_conv_cpu_bn(input_torch, conv_weights_torch, cpu_model.s1_p_bn)
    logger.info(f"Output shape: {tt_output.shape}")
    logger.info(f"Output dtype: {tt_output.dtype}")

    # CPU reference (full chain on CPU)
    logger.info("\n" + "=" * 80)
    logger.info("Running CPU reference (full model on CPU)...")
    logger.info("=" * 80)
    cpu_input = torch.load("cpu_input.pt", map_location="cpu")
    logger.info(f"CPU input shape: {cpu_input.shape}")
    logger.info(f"CPU input dtype: {cpu_input.dtype}")

    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)

    logger.info(f"CPU output shape: {cpu_output.shape}")
    logger.info(f"CPU output dtype: {cpu_output.dtype}")

    # Verify inputs match
    logger.info("\n=== Verification: Comparing inputs ===")
    inputs_match = torch.allclose(input_torch, cpu_input)
    logger.info(f"Inputs match (allclose): {inputs_match}")
    assert inputs_match, "TT and CPU inputs do not match!"

    conv_weights_match = torch.allclose(conv_weights_torch, cpu_model.s1_p_conv.weight.data)
    logger.info(f"Conv weights match (allclose): {conv_weights_match}")
    assert conv_weights_match, "TT and CPU conv weights do not match!"

    logger.info("✓ All input tensors and parameters match between TT and CPU!")

    # Compare outputs
    logger.info("\n=== Output Comparison ===")
    pcc = compute_pcc(tt_output, cpu_output)

    logger.info("\n" + "=" * 80)
    logger.info("PCC={}", {pcc})
    logger.info("=" * 80)

    return pcc

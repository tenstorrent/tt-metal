import ttnn

import tests.ttnn.unit_tests.operations.conv2d_bn2d.utils as utils
import torch
import torch.nn as nn
from loguru import logger
from pytorchcv.model_provider import get_model as ptcv_get_model


def ttnn_model_sanity(input_torch, conv_weights_torch, bn_mean_torch, bn_var_torch, bn_weight_torch, bn_bias_torch):
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
    # DEVICE OPERATION 1: Conv2d (ttnn)
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

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        fp32_dest_acc_en=True,  # ← Enable FP32 dest
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
        compute_config=compute_config,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    logger.info(f"Conv output shape on device: {conv_output_ttnn.shape}")

    # Transfer back to CPU
    logger.info("Transferring conv output back to CPU...")
    conv_output_torch = ttnn.to_torch(conv_output_ttnn)

    # ========================================================================
    # CPU POSTPROCESSING for Conv2d / PREPROCESSING for Batch Norm (using torch)
    # ========================================================================
    logger.info("CPU postprocessing: flattened -> NCHW format for batch_norm...")
    # Reshape from [1, 1, 12544, 64] to [1, 112, 112, 64]
    conv_reshaped = conv_output_torch.reshape(1, 112, 112, 64)
    # Permute from NHWC [1, 112, 112, 64] to NCHW [1, 64, 112, 112]
    conv_nchw = conv_reshaped.permute(0, 3, 1, 2)
    logger.info(f"After CPU postprocessing (NCHW): {conv_nchw.shape}")

    # ========================================================================
    # DEVICE OPERATION 2: Batch Norm (ttnn)
    # ========================================================================
    logger.info("Transferring conv output to device for batch_norm...")
    # Transfer conv output to device with TILE layout (required by batch_norm)
    conv_nchw_ttnn = ttnn.from_torch(
        conv_nchw,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # Prepare BN parameters on device
    logger.info("Preparing batch norm parameters on device...")
    bn_mean_ttnn = ttnn.from_torch(
        bn_mean_torch.reshape(1, 64, 1, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    bn_var_ttnn = ttnn.from_torch(
        bn_var_torch.reshape(1, 64, 1, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    bn_weight_ttnn = ttnn.from_torch(
        bn_weight_torch.reshape(1, 64, 1, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    bn_bias_ttnn = ttnn.from_torch(
        bn_bias_torch.reshape(1, 64, 1, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    logger.info("Running ttnn.batch_norm on device...")
    bn_output_ttnn = ttnn.batch_norm(
        conv_nchw_ttnn,
        running_mean=bn_mean_ttnn,
        running_var=bn_var_ttnn,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=bn_weight_ttnn,
        bias=bn_bias_ttnn,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    logger.info(f"Batch norm output shape on device: {bn_output_ttnn.shape}")

    # Transfer back to CPU
    logger.info("Transferring batch_norm output back to CPU...")
    bn_output_torch = ttnn.to_torch(bn_output_ttnn)

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


def test_sanity():
    logger.info("=" * 80)
    logger.info("SANITY TEST: Conv2d + BatchNorm on device, all else on CPU")
    logger.info("=" * 80)

    # Load tensors (on CPU as torch tensors)
    logger.info("\nLoading tensors on CPU...")
    bn_var_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    bn_mean_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    bn_bias_ttnn = ttnn.load_tensor("./tensors/arg2.tensorbin")
    bn_weight_ttnn = ttnn.load_tensor("./tensors/arg3.tensorbin")
    conv_weights_ttnn = ttnn.load_tensor("./tensors/arg4.tensorbin")
    input_ttnn = ttnn.load_tensor("./tensors/arg5.tensorbin")

    # Convert to torch immediately
    bn_var_torch = ttnn.to_torch(bn_var_ttnn)
    bn_mean_torch = ttnn.to_torch(bn_mean_ttnn)
    bn_bias_torch = ttnn.to_torch(bn_bias_ttnn)
    bn_weight_torch = ttnn.to_torch(bn_weight_ttnn)
    conv_weights_torch = ttnn.to_torch(conv_weights_ttnn)
    input_torch = ttnn.to_torch(input_ttnn)

    logger.info(f"Input shape: {input_torch.shape}")
    logger.info(f"Conv weights shape: {conv_weights_torch.shape}")
    logger.info(
        f"BN parameters loaded (shapes): mean={bn_mean_torch.shape}, var={bn_var_torch.shape}, weight={bn_weight_torch.shape}, bias={bn_bias_torch.shape}"
    )

    # TT run (only conv2d and batch_norm on device)
    logger.info("\n" + "=" * 80)
    logger.info("Running TT inference (only conv2d + batch_norm on device)...")
    logger.info("=" * 80)
    tt_output = ttnn_model_sanity(
        input_torch, conv_weights_torch, bn_mean_torch, bn_var_torch, bn_weight_torch, bn_bias_torch
    )
    logger.info(f"TT output shape: {tt_output.shape}")
    logger.info(f"TT output dtype: {tt_output.dtype}")

    # CPU run
    logger.info("\n" + "=" * 80)
    logger.info("Running CPU inference (full model on CPU)...")
    logger.info("=" * 80)
    model = ptcv_get_model("lwopenpose2d_mobilenet_cmupan_coco", pretrained=True).to(torch.bfloat16)
    cpu_model = Wrapper(model)
    cpu_model.eval()

    cpu_input = torch.load("cpu_input.pt", map_location="cpu")
    logger.info(f"CPU input shape: {cpu_input.shape}")
    logger.info(f"CPU input dtype: {cpu_input.dtype}")

    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)

    logger.info(f"CPU output shape: {cpu_output.shape}")
    logger.info(f"CPU output dtype: {cpu_output.dtype}")

    # Verify inputs and parameters match
    logger.info("\n=== Verification: Comparing TT and CPU tensors ===")

    inputs_match = torch.allclose(input_torch, cpu_input)
    logger.info(f"Inputs match (allclose): {inputs_match}")
    assert inputs_match, "TT and CPU inputs do not match!"

    conv_weights_match = torch.allclose(conv_weights_torch, cpu_model.s1_p_conv.weight.data)
    logger.info(f"Conv weights match (allclose): {conv_weights_match}")
    assert conv_weights_match, "TT and CPU conv weights do not match!"

    bn_weight_match = torch.allclose(bn_weight_torch, cpu_model.s1_p_bn.weight.data)
    logger.info(f"BN weight match (allclose): {bn_weight_match}")
    assert bn_weight_match, "TT and CPU BN weights do not match!"

    bn_bias_match = torch.allclose(bn_bias_torch, cpu_model.s1_p_bn.bias.data)
    logger.info(f"BN bias match (allclose): {bn_bias_match}")
    assert bn_bias_match, "TT and CPU BN biases do not match!"

    bn_mean_match = torch.allclose(bn_mean_torch, cpu_model.s1_p_bn.running_mean.data)
    logger.info(f"BN running_mean match (allclose): {bn_mean_match}")
    assert bn_mean_match, "TT and CPU BN running_mean do not match!"

    bn_var_match = torch.allclose(bn_var_torch, cpu_model.s1_p_bn.running_var.data)
    logger.info(f"BN running_var match (allclose): {bn_var_match}")
    assert bn_var_match, "TT and CPU BN running_var do not match!"

    logger.info("✓ All input tensors and model parameters match between TT and CPU!")

    # Compare outputs
    logger.info("\n=== Output Comparison ===")

    # Compute PCC
    pcc = compute_pcc(tt_output, cpu_output)

    logger.info("\n" + "=" * 80)
    logger.info(f"Output PCC: {pcc}")

    logger.info("=" * 80)

    return pcc

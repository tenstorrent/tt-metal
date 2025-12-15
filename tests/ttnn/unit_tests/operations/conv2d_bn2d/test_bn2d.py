import ttnn

import tests.ttnn.unit_tests.operations.conv2d_bn2d.utils as utils
import torch
import torch.nn as nn
from loguru import logger
from pytorchcv.model_provider import get_model as ptcv_get_model


def ttnn_batch_norm_only(conv_output_torch, bn_mean_torch, bn_var_torch, bn_weight_torch, bn_bias_torch):
    """
    Sanity test: ONLY batch_norm runs on ttnn device.
    Input (conv output) generated on CPU, all preprocessing/postprocessing done with torch.
    """
    device = utils.DeviceGetter.get_device((1, 1))

    logger.info(f"Conv output shape (NCHW, input to batch_norm): {conv_output_torch.shape}")

    # ========================================================================
    # DEVICE OPERATION: Batch Norm (ttnn)
    # ========================================================================
    logger.info("Transferring conv output to device for batch_norm...")
    # Transfer conv output to device with TILE layout (required by batch_norm)
    conv_nchw_ttnn = ttnn.from_torch(
        conv_output_torch,
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
    """CPU model with conv2d + batch_norm."""

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


def test_bn2d():
    """
    SANITY TEST: Only batch_norm runs on ttnn device.
    Uses ttnn_conv_output.pt (ACTUAL ttnn conv2d output) as input.
    This tests batch_norm with realistic imperfect input from conv2d.
    All other operations (reshape, permute, preprocessing) done on CPU with torch.
    """
    logger.info("=" * 80)
    logger.info("SANITY TEST: Batch Norm only on device, all else on CPU")
    logger.info("=" * 80)

    # Load BN parameters (on CPU as torch tensors)
    logger.info("\nLoading BN parameters on CPU...")
    bn_var_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    bn_mean_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    bn_bias_ttnn = ttnn.load_tensor("./tensors/arg2.tensorbin")
    bn_weight_ttnn = ttnn.load_tensor("./tensors/arg3.tensorbin")

    # Convert to torch immediately
    bn_var_torch = ttnn.to_torch(bn_var_ttnn)
    bn_mean_torch = ttnn.to_torch(bn_mean_ttnn)
    bn_bias_torch = ttnn.to_torch(bn_bias_ttnn)
    bn_weight_torch = ttnn.to_torch(bn_weight_ttnn)

    logger.info(
        f"BN parameters loaded (shapes): mean={bn_mean_torch.shape}, var={bn_var_torch.shape}, weight={bn_weight_torch.shape}, bias={bn_bias_torch.shape}"
    )

    # Load batch_norm input (conv output) - same for both ttnn and CPU
    logger.info("\n" + "=" * 80)
    logger.info("Loading batch_norm input (TTNN conv2d output)...")
    logger.info("=" * 80)

    # Use actual ttnn conv2d output instead of perfect CPU data
    bn_input = torch.load("ttnn_conv_output.pt", map_location="cpu")
    logger.info(f"Batch norm input shape: {bn_input.shape}")
    logger.info(f"Batch norm input dtype: {bn_input.dtype}")
    logger.info("bn_input={}", bn_input)
    logger.info("Note: This is ACTUAL ttnn conv2d output (not perfect CPU data)")

    # Load CPU model
    model = ptcv_get_model("lwopenpose2d_mobilenet_cmupan_coco", pretrained=True).to(torch.bfloat16)
    cpu_model = Wrapper(model)
    cpu_model.eval()

    # TT run (only batch_norm on device)
    logger.info("\n" + "=" * 80)
    logger.info("Running TT inference (only batch_norm on device)...")
    logger.info("=" * 80)
    tt_output = ttnn_batch_norm_only(bn_input, bn_mean_torch, bn_var_torch, bn_weight_torch, bn_bias_torch)
    logger.info(f"TT output shape: {tt_output.shape}")
    logger.info(f"TT output dtype: {tt_output.dtype}")

    # CPU run (only batch_norm with same input)
    logger.info("\n" + "=" * 80)
    logger.info("Running CPU inference (only batch_norm)...")
    logger.info("=" * 80)

    with torch.no_grad():
        # Run only batch_norm with the same input
        cpu_output = cpu_model.s1_p_bn(bn_input)

    logger.info(f"CPU output shape: {cpu_output.shape}")
    logger.info(f"CPU output dtype: {cpu_output.dtype}")

    # Verify inputs and BN parameters match
    logger.info("\n=== Verification: Comparing TT and CPU tensors ===")

    # 1. Verify batch_norm input is the same
    logger.info(f"Batch norm input (same for TT and CPU): shape={bn_input.shape}, dtype={bn_input.dtype}")

    # 2. Verify BN parameters match
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

    logger.info("✓ Batch norm input and parameters match between TT and CPU!")

    # Compare outputs
    logger.info("\n=== Output Comparison ===")

    # Compute PCC
    pcc = compute_pcc(tt_output, cpu_output)

    logger.info("\n" + "=" * 80)
    logger.info(f"Batch Norm Output PCC: {pcc}")
    logger.info("=" * 80)

    return pcc

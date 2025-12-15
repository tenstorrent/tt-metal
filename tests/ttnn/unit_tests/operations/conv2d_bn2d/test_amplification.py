import ttnn
import tests.ttnn.unit_tests.operations.conv2d_bn2d.utils as utils
import torch
from loguru import logger
from pytorchcv.model_provider import get_model as ptcv_get_model


def ttnn_model_with_intermediate(input_torch, conv_weights_torch, cpu_bn_layer):
    device = utils.DeviceGetter.get_device((1, 1))

    logger.info(f"Input shape (NCHW): {input_torch.shape}")

    # ========================================================================
    # CPU PREPROCESSING for Conv2d
    # ========================================================================
    logger.info("CPU preprocessing: NCHW -> flattened format for conv2d...")
    input_nhwc = input_torch.permute(0, 2, 3, 1)
    input_flattened = input_nhwc.reshape(1, 1, 12544, 32)
    logger.info(f"After CPU preprocessing: {input_flattened.shape}")

    # ========================================================================
    # TT Conv2d
    # ========================================================================
    logger.info("Transferring input to device for conv2d...")
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
    # CPU POSTPROCESSING for Conv2d
    # ========================================================================
    logger.info("CPU postprocessing: flattened -> NCHW format...")
    conv_reshaped = conv_output_torch.reshape(1, 112, 112, 64)
    conv_nchw = conv_reshaped.permute(0, 3, 1, 2)
    logger.info(f"After CPU postprocessing (NCHW): {conv_nchw.shape}")

    # ========================================================================
    # CPU Batch Norm
    # ========================================================================
    logger.info("Running CPU batch_norm...")
    with torch.no_grad():
        bn_output_torch = cpu_bn_layer(conv_nchw)

    logger.info(f"Batch norm output shape (CPU): {bn_output_torch.shape}")

    # Return BOTH conv output and final output
    return conv_nchw, bn_output_torch


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.s1_p_conv = model.encoder.backbone.stage1.unit1.pw_conv.conv
        self.s1_p_bn = model.encoder.backbone.stage1.unit1.pw_conv.bn

    def forward(self, x):
        x1 = self.s1_p_conv(x)
        x2 = self.s1_p_bn(x1)
        return x1, x2


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


def test_amplification():
    # Load CPU model
    model = ptcv_get_model("lwopenpose2d_mobilenet_cmupan_coco", pretrained=True).to(torch.bfloat16)
    cpu_model = Wrapper(model)
    cpu_model.eval()

    # Load tensors
    logger.info("\nLoading tensors on CPU...")
    conv_weights_ttnn = ttnn.load_tensor("./tensors/arg4.tensorbin")
    input_ttnn = ttnn.load_tensor("./tensors/arg5.tensorbin")

    conv_weights_torch = ttnn.to_torch(conv_weights_ttnn)
    input_torch = ttnn.to_torch(input_ttnn)

    logger.info(f"Input shape: {input_torch.shape}")
    logger.info(f"Conv weights shape: {conv_weights_torch.shape}")

    # ========================================================================
    # Run TT model (returns conv output + final output)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Running TT model (conv + bn)...")
    logger.info("=" * 80)
    tt_conv_output, tt_final_output = ttnn_model_with_intermediate(input_torch, conv_weights_torch, cpu_model.s1_p_bn)

    # ========================================================================
    # Run CPU model (returns conv output + final output)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Running CPU model (conv + bn)...")
    logger.info("=" * 80)
    cpu_input = torch.load("cpu_input.pt", map_location="cpu")
    logger.info(f"CPU input shape: {cpu_input.shape}")

    cpu_conv_output, cpu_final_output = cpu_model(cpu_input)

    # ========================================================================
    # Verify inputs match
    # ========================================================================
    logger.info("\n=== Verification: Comparing inputs ===")
    inputs_match = torch.allclose(input_torch, cpu_input)
    logger.info(f"Inputs match (allclose): {inputs_match}")
    assert inputs_match, "TT and CPU inputs do not match!"

    conv_weights_match = torch.allclose(conv_weights_torch, cpu_model.s1_p_conv.weight.data)
    logger.info(f"Conv weights match (allclose): {conv_weights_match}")
    assert conv_weights_match, "TT and CPU conv weights do not match!"

    # ========================================================================
    # CRITICAL COMPARISON: Before vs After Batch Norm
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Conv Output (BEFORE batch_norm)")
    logger.info("=" * 80)
    pcc_conv = compute_pcc(tt_conv_output, cpu_conv_output)
    logger.info(f"PCC (TT conv vs CPU conv): {pcc_conv}")

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Final Output (AFTER batch_norm)")
    logger.info("=" * 80)
    pcc_final = compute_pcc(tt_final_output, cpu_final_output)
    logger.info(f"PCC (TT final vs CPU final): {pcc_final}")

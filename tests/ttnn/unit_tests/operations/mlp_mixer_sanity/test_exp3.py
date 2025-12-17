import ttnn
import tests.ttnn.unit_tests.operations.mlp_mixer_sanity.utils as utils
import torch
from loguru import logger
import timm


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


def test_fc2_only():
    """
    Test only the last FC2 operation (matmul + add) on both TT and CPU
    """
    # Get device
    device = utils.DeviceGetter.get_device((1, 1))

    # Load input to fc2
    tt_x6_input = torch.load("tt_x6_torch.pt", map_location="cpu")
    logger.info("tt_x6_input: {}", tt_x6_input)
    logger.info("tt_x6_input shape: {}", tt_x6_input.shape)
    logger.info("tt_x6_input dtype: {}", tt_x6_input.dtype)

    # Load fc2 weights and bias
    fc2_weight_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    fc2_weight_torch = ttnn.to_torch(fc2_weight_ttnn).squeeze()

    fc2_bias_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    fc2_bias_torch = ttnn.to_torch(fc2_bias_ttnn).squeeze()

    logger.info("fc2_weight shape: {}", fc2_weight_torch.shape)
    logger.info("fc2_bias shape: {}", fc2_bias_torch.shape)

    # ===== TT MODEL: FC2 using TTNN =====
    # Convert input to TTNN
    tt_x6_ttnn = ttnn.from_torch(
        tt_x6_input,
        device=device,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # Convert fc2 weight to TTNN with TILE layout
    fc2_weight_device = ttnn.from_torch(
        fc2_weight_torch,
        device=device,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # Convert fc2 bias to TTNN
    fc2_bias_device = ttnn.from_torch(
        fc2_bias_torch,
        device=device,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    ttnn_reshape_8 = ttnn.reshape(
        tt_x6_ttnn,
        [768, 384],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # FC2 matmul
    tt_matmul_output = ttnn.matmul(
        ttnn_reshape_8,
        fc2_weight_device,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )

    # FC2 bias add
    tt_output = ttnn.add(
        tt_matmul_output,
        fc2_bias_device,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(tt_output)  # [768, 196]
    logger.info("TT output shape: {}", tt_output_torch.shape)

    # ===== CPU MODEL: FC2 using torch =====
    # Add batch dimension for torch linear
    cpu_x6_input = tt_x6_input.to(torch.bfloat16)  # [1, 768, 384]

    # FC2 using torch.nn.functional.linear
    cpu_output = torch.nn.functional.linear(
        cpu_x6_input, weight=fc2_weight_torch.to(torch.bfloat16), bias=fc2_bias_torch.to(torch.bfloat16)
    )  # [1, 768, 196]

    logger.info("CPU output shape: {}", cpu_output.shape)

    # ===== COMPARE OUTPUTS =====
    logger.info("\n=== FC2 Output Comparison ===")
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"PCC: {pcc}")

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


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 196, 1]),
        fill_value=9.9837779998779297e-07,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_0 = ttnn.permute(
        ttnn_full_0,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_full_0, False)
    util_create_list_0 = [ttnn_permute_0]
    return util_create_list_0


def main_const_eval_1(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_0 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_to_device_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_to_layout_0,
        [1, 1, 196],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_repeat_0 = ttnn.repeat(ttnn_reshape_0, ttnn.Shape([1, 768, 1]))
    ttnn.deallocate(ttnn_reshape_0, False)
    util_create_list_1 = [ttnn_repeat_0]
    return util_create_list_1


def main_const_eval_2(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_1 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_to_device_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_1, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_to_layout_1,
        [1, 1, 384],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_repeat_1 = ttnn.repeat(ttnn_reshape_1, ttnn.Shape([1, 768, 1]))
    ttnn.deallocate(ttnn_reshape_1, False)
    util_create_list_2 = [ttnn_repeat_1]
    return util_create_list_2


def main_const_eval_3(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_2 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_to_device_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_2, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_2,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_2,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_permute_1,
        [768, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_1, False)
    util_create_list_3 = [ttnn_reshape_3]
    return util_create_list_3


def main_const_eval_4(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_3 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_to_device_3,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_3, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_to_layout_3,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn_permute_2 = ttnn.permute(
        ttnn_reshape_4,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_4, False)
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_permute_2,
        [768, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_2, False)
    util_create_list_4 = [ttnn_reshape_5]
    return util_create_list_4


CACHED_main_const_eval_0 = None
CACHED_main_const_eval_1 = None
CACHED_main_const_eval_2 = None
CACHED_main_const_eval_3 = None
CACHED_main_const_eval_4 = None


def _main(input, norm1_weight, norm1_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    """
    Hybrid implementation: Use torch for all ops except last matmul+add (fc2)
    """
    input_0 = input[0]
    input_1 = input[1]
    input_2 = input[2]
    input_3 = input[3]
    input_4 = input[4]
    input_5 = input[5]
    input_6 = input[6]

    # Get device for final TTNN operations
    device = utils.DeviceGetter.get_device((1, 1))

    # ===== USE TORCH OPERATIONS FOR EVERYTHING EXCEPT LAST MATMUL+ADD =====

    # Convert input from TTNN to torch
    torch_input = ttnn.to_torch(input_6).to(torch.bfloat16)  # [1, 196, 768]

    # x1: LayerNorm using torch
    torch_x1 = torch.nn.functional.layer_norm(
        torch_input, normalized_shape=(768,), weight=norm1_weight, bias=norm1_bias, eps=1e-6
    )  # [1, 196, 768]

    # x2: Transpose using torch
    torch_x2 = torch_x1.transpose(1, 2)  # [1, 768, 196]

    # x3: FC1 using torch
    torch_x3 = torch.nn.functional.linear(torch_x2, weight=fc1_weight, bias=fc1_bias)  # [1, 768, 384]

    # x4: GELU using torch
    torch_x4 = torch.nn.functional.gelu(torch_x3)  # [1, 768, 384]

    # x5: Dropout (identity in eval) using torch
    torch_x5 = torch_x4  # [1, 768, 384]

    # x6: Norm (identity layer) using torch
    torch_x6 = torch_x5  # [1, 768, 384]

    # ===== NOW USE TTNN FOR LAST FC2 (MATMUL + ADD) =====

    # Convert torch tensor to TTNN for fc2
    torch_x6_2d = torch_x6.squeeze(0)  # [768, 384]
    ttnn_x6 = ttnn.from_torch(
        torch_x6_2d,
        device=device,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # FC2 matmul using TTNN
    ttnn_matmul_fc2 = ttnn.matmul(
        ttnn_x6,
        input_1,  # fc2 weight
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )

    # FC2 bias add using TTNN
    fc2_bias_ttnn = ttnn.from_torch(
        fc2_bias,
        device=device,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    ttnn_x7 = ttnn.add(
        ttnn_matmul_fc2,
        fc2_bias_ttnn,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # Convert torch intermediates to TTNN format for return (to match test expectations)
    # These need to be in the same format as the original TTNN implementation
    tt_x1 = ttnn.from_torch(
        torch_x1.squeeze(0).transpose(0, 1),  # [196, 768] -> [768, 196]
        device=device,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    tt_x2 = ttnn.from_torch(
        torch_x2.squeeze(0),  # [768, 196]
        device=device,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    tt_x3 = ttnn.from_torch(
        torch_x3.squeeze(0),  # [768, 384]
        device=device,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    tt_x4 = ttnn.from_torch(
        torch_x4.squeeze(0),  # [768, 384]
        device=device,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    tt_x5 = tt_x4  # Same tensor
    tt_x6 = tt_x4  # Same tensor
    tt_x7 = ttnn_x7

    # Return all 7 intermediate tensors for comparison
    util_create_list_9 = [tt_x1, tt_x2, tt_x3, tt_x4, tt_x5, tt_x6, tt_x7]
    return util_create_list_9


def load_inputs_for__main():
    utils_DeviceGetter_get_device_5 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_5,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_5,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_5,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_10 = [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
        utils_load_tensor_4,
        utils_load_tensor_5,
        utils_load_tensor_6,
    ]
    return util_create_list_10


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm1 = model.blocks[11].norm1
        self.fc1 = model.blocks[11].mlp_tokens.fc1
        self.act = model.blocks[11].mlp_tokens.act
        self.drop1 = model.blocks[11].mlp_tokens.drop1
        self.norm = model.blocks[11].mlp_tokens.norm
        self.fc2 = model.blocks[11].mlp_tokens.fc2

    def forward(self, x):
        x1 = self.norm1(x)
        x2 = x1.transpose(1, 2)
        x3 = self.fc1(x2)
        x4 = self.act(x3)
        x5 = self.drop1(x4)
        x6 = self.norm(x5)
        x7 = self.fc2(x6)

        return x1, x2, x3, x4, x5, x6, x7


def test_mlp_mixer():
    # cpu run - load input
    cpu_input = torch.load("cpu_input.pt", map_location="cpu")
    logger.info("cpu_input={}", cpu_input)
    logger.info("cpu_input.shape={}", cpu_input.shape)
    logger.info("cpu_input.dtype={}", cpu_input.dtype)

    # Load the actual weights and biases used by ttnn
    arg0_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    arg0_torch = ttnn.to_torch(arg0_ttnn).squeeze()  # fc2 bias

    arg1_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    arg1_torch = ttnn.to_torch(arg1_ttnn).squeeze()  # fc2 weight

    arg2_ttnn = ttnn.load_tensor("./tensors/arg2.tensorbin")
    arg2_torch = ttnn.to_torch(arg2_ttnn).squeeze()  # fc1 bias

    arg3_ttnn = ttnn.load_tensor("./tensors/arg3.tensorbin")
    arg3_torch = ttnn.to_torch(arg3_ttnn).squeeze()  # fc1 weight

    arg4_ttnn = ttnn.load_tensor("./tensors/arg4.tensorbin")
    arg4_torch = ttnn.to_torch(arg4_ttnn).squeeze()  # norm1 bias

    arg5_ttnn = ttnn.load_tensor("./tensors/arg5.tensorbin")
    arg5_torch = ttnn.to_torch(arg5_ttnn).squeeze()  # norm1 weight

    arg6_ttnn = ttnn.load_tensor("./tensors/arg6.tensorbin")
    arg6_torch = ttnn.to_torch(arg6_ttnn)  # input

    # Load the base model first
    base_model = timm.create_model("mixer_b16_224.goog_in21k", pretrained=True)
    base_model.eval()

    # Create CPU model wrapper
    cpu_model = Wrapper(base_model)
    cpu_model.to(torch.bfloat16)
    cpu_model.eval()

    # tt run - use torch ops for everything except last matmul+add
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_intermediates = _main(
        load_inputs_for__main_0,
        norm1_weight=arg5_torch.to(torch.bfloat16),
        norm1_bias=arg4_torch.to(torch.bfloat16),
        fc1_weight=arg3_torch.to(torch.bfloat16),
        fc1_bias=arg2_torch.to(torch.bfloat16),
        fc2_weight=arg1_torch.to(torch.bfloat16),
        fc2_bias=arg0_torch.to(torch.bfloat16),
    )

    with torch.no_grad():
        cpu_x1, cpu_x2, cpu_x3, cpu_x4, cpu_x5, cpu_x6, cpu_x7 = cpu_model(cpu_input)

    # Verify inputs and parameters match between TT and CPU
    logger.info("\n=== Verification: Comparing TT and CPU tensors ===")

    # 1. Compare main input (arg6)
    inputs_match = torch.allclose(arg6_torch, cpu_input)
    logger.info(f"Inputs match (allclose): {inputs_match}")

    # 2. Compare arg0 - fc2 bias
    fc2_bias_match = torch.allclose(arg0_torch.squeeze(), cpu_model.fc2.bias.data)
    logger.info(f"fc2 bias match (allclose): {fc2_bias_match}")

    # 3. Compare arg1 - fc2 weight (transposed for matmul)
    fc2_weight_match = torch.allclose(arg1_torch.squeeze(), cpu_model.fc2.weight.data)
    logger.info(f"fc2 weight match (allclose): {fc2_weight_match}")

    # 4. Compare arg2 - fc1 bias
    fc1_bias_match = torch.allclose(arg2_torch.squeeze(), cpu_model.fc1.bias.data)
    logger.info(f"fc1 bias match (allclose): {fc1_bias_match}")

    # 5. Compare arg3 - fc1 weight (transposed for matmul)
    fc1_weight_match = torch.allclose(arg3_torch.squeeze(), cpu_model.fc1.weight.data)
    logger.info(f"fc1 weight match (allclose): {fc1_weight_match}")

    # 6. Compare arg4 - norm1 bias
    norm1_bias_match = torch.allclose(arg4_torch.squeeze(), cpu_model.norm1.bias.data)
    logger.info(f"norm1 bias match (allclose): {norm1_bias_match}")

    # 7. Compare arg5 - norm1 weight
    norm1_weight_match = torch.allclose(arg5_torch.squeeze(), cpu_model.norm1.weight.data)
    logger.info(f"norm1 weight match (allclose): {norm1_weight_match}")

    # Compare intermediate outputs - convert from device only at comparison time
    logger.info("\n=== Intermediate Outputs Comparison ===")

    # x1: after_norm1
    # TT shape: [768, 196], CPU shape: [1, 196, 768]
    tt_x1_torch = ttnn.to_torch(tt_intermediates[0])
    cpu_x1_reshaped = cpu_x1.squeeze(0).transpose(0, 1)  # [1, 196, 768] -> [196, 768] -> [768, 196]
    pcc_x1 = compute_pcc(tt_x1_torch, cpu_x1_reshaped)
    logger.info(f"x1: After norm1 - TT shape: {tt_x1_torch.shape}, CPU shape: {cpu_x1_reshaped.shape}, PCC: {pcc_x1}")

    # x2: after_transpose
    # TT: [768, 196] (already in transposed format), CPU: [1, 768, 196]
    tt_x2_torch = ttnn.to_torch(tt_intermediates[1])
    cpu_x2_reshaped = cpu_x2.squeeze(0)  # [1, 768, 196] -> [768, 196]
    pcc_x2 = compute_pcc(tt_x2_torch, cpu_x2_reshaped)
    logger.info(
        f"x2: After transpose - TT shape: {tt_x2_torch.shape}, CPU shape: {cpu_x2_reshaped.shape}, PCC: {pcc_x2}"
    )

    # x3: after_fc1
    # TT shape: [768, 384], CPU shape: [1, 768, 384]
    tt_x3_torch = ttnn.to_torch(tt_intermediates[2])
    cpu_x3_reshaped = cpu_x3.squeeze(0)  # [1, 768, 384] -> [768, 384]
    pcc_x3 = compute_pcc(tt_x3_torch, cpu_x3_reshaped)
    logger.info(f"x3: After fc1 - TT shape: {tt_x3_torch.shape}, CPU shape: {cpu_x3_reshaped.shape}, PCC: {pcc_x3}")

    # x4: after_gelu
    # TT shape: [768, 384], CPU shape: [1, 768, 384]
    tt_x4_torch = ttnn.to_torch(tt_intermediates[3])
    cpu_x4_reshaped = cpu_x4.squeeze(0)  # [1, 768, 384] -> [768, 384]
    pcc_x4 = compute_pcc(tt_x4_torch, cpu_x4_reshaped)
    logger.info(f"x4: After gelu - TT shape: {tt_x4_torch.shape}, CPU shape: {cpu_x4_reshaped.shape}, PCC: {pcc_x4}")

    # x5: after_drop1 (same as gelu in eval mode)
    tt_x5_torch = ttnn.to_torch(tt_intermediates[4])
    cpu_x5_reshaped = cpu_x5.squeeze(0)  # [1, 768, 384] -> [768, 384]
    pcc_x5 = compute_pcc(tt_x5_torch, cpu_x5_reshaped)
    logger.info(f"x5: After drop1 - TT shape: {tt_x5_torch.shape}, CPU shape: {cpu_x5_reshaped.shape}, PCC: {pcc_x5}")

    # x6: after_norm (identity layer)
    tt_x6_torch = ttnn.to_torch(tt_intermediates[5])
    cpu_x6_reshaped = cpu_x6.squeeze(0)  # [1, 768, 384] -> [768, 384]
    pcc_x6 = compute_pcc(tt_x6_torch, cpu_x6_reshaped)
    logger.info(f"x6: After norm - TT shape: {tt_x6_torch.shape}, CPU shape: {cpu_x6_reshaped.shape}, PCC: {pcc_x6}")

    # x7: after_fc2 (final output)
    # TT shape: [768, 196], CPU shape: [1, 768, 196]
    tt_x7_torch = ttnn.to_torch(tt_intermediates[6])
    cpu_x7_reshaped = cpu_x7.squeeze(0)  # [1, 768, 196] -> [768, 196]
    pcc_x7 = compute_pcc(tt_x7_torch, cpu_x7_reshaped)
    logger.info(f"x7: After fc2 - TT shape: {tt_x7_torch.shape}, CPU shape: {cpu_x7_reshaped.shape}, PCC: {pcc_x7}")

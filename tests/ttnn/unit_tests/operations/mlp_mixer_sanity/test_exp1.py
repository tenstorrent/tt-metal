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


def _main(input):
    global CACHED_main_const_eval_4
    global CACHED_main_const_eval_3
    global CACHED_main_const_eval_2
    global CACHED_main_const_eval_1
    global CACHED_main_const_eval_0
    input_0 = input[0]
    input_1 = input[1]
    input_2 = input[2]
    input_3 = input[3]
    input_4 = input[4]
    input_5 = input[5]
    input_6 = input[6]
    const_0 = main_const_eval_0
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(const_0, CACHED_main_const_eval_0)
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapperZeroArg_0
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    const_1 = main_const_eval_1
    util_create_list_5 = [input_0]
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(const_1, util_create_list_5, CACHED_main_const_eval_1)
    CACHED_main_const_eval_1 = utils_constEvalFuncWrapper_0
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_2 = main_const_eval_2
    util_create_list_6 = [input_2]
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(const_2, util_create_list_6, CACHED_main_const_eval_2)
    CACHED_main_const_eval_2 = utils_constEvalFuncWrapper_1
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    const_3 = main_const_eval_3
    util_create_list_7 = [input_4]
    utils_constEvalFuncWrapper_2 = utils.constEvalFuncWrapper(const_3, util_create_list_7, CACHED_main_const_eval_3)
    CACHED_main_const_eval_3 = utils_constEvalFuncWrapper_2
    utils_constEvalFuncWrapper_2_0 = utils_constEvalFuncWrapper_2[0]
    const_4 = main_const_eval_4
    util_create_list_8 = [input_5]
    utils_constEvalFuncWrapper_3 = utils.constEvalFuncWrapper(const_4, util_create_list_8, CACHED_main_const_eval_4)
    CACHED_main_const_eval_4 = utils_constEvalFuncWrapper_3
    utils_constEvalFuncWrapper_3_0 = utils_constEvalFuncWrapper_3[0]
    ttnn_to_layout_4 = ttnn.to_layout(
        input_6,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_0 = ttnn.mean(
        ttnn_to_layout_4,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    ttnn_permute_3 = ttnn.permute(
        ttnn_mean_0,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_mean_0, False)
    ttnn_to_layout_5 = ttnn.to_layout(
        input_6,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_6, False)
    ttnn_permute_4 = ttnn.permute(
        ttnn_to_layout_5,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_5, False)
    ttnn_neg_0 = ttnn.neg(
        ttnn_permute_3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_3, False)
    ttnn_add_0 = ttnn.add(
        ttnn_permute_4,
        ttnn_neg_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_neg_0, False)
    ttnn.deallocate(ttnn_permute_4, False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_add_0,
        ttnn_add_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_1 = ttnn.mean(
        ttnn_multiply_0,
        [1],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_0, False)
    ttnn_add_1 = ttnn.add(
        ttnn_mean_1,
        utils_constEvalFuncWrapperZeroArg_0_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_mean_1, False)
    ttnn_rsqrt_0 = ttnn.rsqrt(
        ttnn_add_1,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_1, False)
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_rsqrt_0,
        [1, 196],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rsqrt_0, False)
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_add_0,
        [768, 196],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_reshape_7,
        ttnn_reshape_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_7, False)
    ttnn.deallocate(ttnn_reshape_6, False)
    ttnn_multiply_2 = ttnn.multiply(
        ttnn_multiply_1,
        utils_constEvalFuncWrapper_3_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_1, False)
    ttnn_add_2 = ttnn.add(
        ttnn_multiply_2,
        utils_constEvalFuncWrapper_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_2, False)
    # x1: after_norm1 - keep for comparison
    tt_x1 = ttnn_add_2

    # x2: after_transpose - in TT model, ttnn_add_2 is already in transposed format [768, 196]
    tt_x2 = ttnn_add_2  # Same as x1 since data is already transposed

    ttnn_matmul_0 = ttnn.matmul(
        ttnn_add_2,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    # Don't deallocate ttnn_add_2, we need it for comparison
    ttnn_add_3 = ttnn.add(
        ttnn_matmul_0,
        utils_constEvalFuncWrapper_1_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    # x3: after_fc1 - keep for comparison
    tt_x3 = ttnn_add_3

    ttnn_gelu_0 = ttnn.gelu(
        ttnn_add_3,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    # Don't deallocate ttnn_add_3, we need it for comparison
    # x4: after_gelu - keep for comparison
    tt_x4 = ttnn_gelu_0

    # x5: after_drop1 - drop1 is identity in eval mode, same as after_gelu
    tt_x5 = ttnn_gelu_0

    # x6: after_norm - norm is identity layer, same as after_gelu
    tt_x6 = ttnn_gelu_0

    ttnn_reshape_8 = ttnn.reshape(
        ttnn_gelu_0,
        [768, 384],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    # Don't deallocate ttnn_gelu_0, we need it for comparison
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_reshape_8,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_8, False)
    ttnn_add_4 = ttnn.add(
        ttnn_matmul_1,
        utils_constEvalFuncWrapper_0_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_1, False)
    # x7: after_fc2 (final output) - keep for comparison
    tt_x7 = ttnn_add_4

    # Return all 7 intermediate tensors for comparison (don't deallocate them)
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
    # tt run
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_intermediates = _main(load_inputs_for__main_0)

    # cpu run - load input
    cpu_input = torch.load("cpu_input.pt", map_location="cpu")
    logger.info("cpu_input={}", cpu_input)
    logger.info("cpu_input.shape={}", cpu_input.shape)
    logger.info("cpu_input.dtype={}", cpu_input.dtype)

    # Load the actual weights and biases used by ttnn
    arg0_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    arg0_torch = ttnn.to_torch(arg0_ttnn)

    arg1_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    arg1_torch = ttnn.to_torch(arg1_ttnn)

    arg2_ttnn = ttnn.load_tensor("./tensors/arg2.tensorbin")
    arg2_torch = ttnn.to_torch(arg2_ttnn)

    arg3_ttnn = ttnn.load_tensor("./tensors/arg3.tensorbin")
    arg3_torch = ttnn.to_torch(arg3_ttnn)

    arg4_ttnn = ttnn.load_tensor("./tensors/arg4.tensorbin")
    arg4_torch = ttnn.to_torch(arg4_ttnn)

    arg5_ttnn = ttnn.load_tensor("./tensors/arg5.tensorbin")
    arg5_torch = ttnn.to_torch(arg5_ttnn)

    arg6_ttnn = ttnn.load_tensor("./tensors/arg6.tensorbin")
    arg6_torch = ttnn.to_torch(arg6_ttnn)

    # Load the base model first
    base_model = timm.create_model("mixer_b16_224.goog_in21k", pretrained=True)
    base_model.eval()

    # Create CPU model wrapper
    cpu_model = Wrapper(base_model)
    cpu_model.to(torch.bfloat16)
    cpu_model.eval()

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

    # torch.save(tt_x6_torch,"tt_x6_torch.pt")
    # logger.info("tt_x6_torch={}",tt_x6_torch)
    # logger.info("tt_x6_torch.shape={}",tt_x6_torch.shape)
    # logger.info("tt_x6_torch.dtype={}",tt_x6_torch.dtype)

    cpu_x6_reshaped = cpu_x6.squeeze(0)  # [1, 768, 384] -> [768, 384]
    pcc_x6 = compute_pcc(tt_x6_torch, cpu_x6_reshaped)
    logger.info(f"x6: After norm - TT shape: {tt_x6_torch.shape}, CPU shape: {cpu_x6_reshaped.shape}, PCC: {pcc_x6}")

    # x7: after_fc2 (final output)
    # TT shape: [768, 196], CPU shape: [1, 768, 196]
    tt_x7_torch = ttnn.to_torch(tt_intermediates[6])
    cpu_x7_reshaped = cpu_x7.squeeze(0)  # [1, 768, 196] -> [768, 196]
    pcc_x7 = compute_pcc(tt_x7_torch, cpu_x7_reshaped)
    logger.info(f"x7: After fc2 - TT shape: {tt_x7_torch.shape}, CPU shape: {cpu_x7_reshaped.shape}, PCC: {pcc_x7}")

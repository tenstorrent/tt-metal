import torch
import ttnn
import tests.ttnn.unit_tests.operations.linear_sigmoid.utils as utils
from loguru import logger
from scipy.stats import pearsonr
from transformers import YolosForObjectDetection


def main_const_eval_0(v1):
    v2 = v1[0]
    v3 = ttnn.reshape(
        v2,
        [1, 1, 4],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v4 = ttnn.repeat(v3, ttnn.Shape([1, 100, 1]))
    ttnn.deallocate(v3, False)
    v5 = [v4]
    return v5


CACHED_main_const_eval_0 = None


def _main(v1):
    global CACHED_main_const_eval_0
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = main_const_eval_0
    v6 = [v2]
    v7 = utils.constEvalFuncWrapper(v5, v6, CACHED_main_const_eval_0)
    CACHED_main_const_eval_0 = v7
    v8 = v7[0]
    v9 = ttnn.reshape(
        v4,
        [100, 384],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v4, False)
    v10 = ttnn.matmul(
        v9,
        v3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(v9, False)
    v11 = ttnn.add(
        v10,
        v8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = [v11]
    return v12


def load_inputs_for__main():
    v1 = utils.DeviceGetter.get_device((1, 1))
    v2 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v3 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v4 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v5 = [v2, v3, v4]
    return v5


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.linear_2 = model.bbox_predictor.layers[2]

    def forward(self, r2):
        pred_boxes = self.linear_2(r2)

        return pred_boxes


def compute_pcc(tt, cpu):
    a = tt.detach().cpu().numpy().flatten()
    b = cpu.detach().cpu().numpy().flatten()
    pcc, _ = pearsonr(a, b)

    return pcc


def test_linear_sig():
    # tt run
    v1 = load_inputs_for__main()
    tt_output = _main(v1)

    logger.info("tt_output={}", tt_output)

    # cpu run
    model_kwargs = {"return_dict": False}
    model_kwargs["torch_dtype"] = torch.bfloat16
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small", **model_kwargs)
    model.eval()

    cpu_model = Wrapper(model)

    x1 = torch.load("linear_sigmoid_ip.pt", map_location="cpu")

    with torch.no_grad():
        cpu_output = cpu_model(x1)

    for i in range(len(tt_output)):
        tt_output[i] = ttnn.to_torch(tt_output[i])

    logger.info("cpu_output={}", cpu_output)

    # pcc check
    pcc_values = compute_pcc(tt_output[0].float(), cpu_output.float())

    logger.info(f"PCC : {pcc_values}")

    # Input, Weights and bias cross check

    # Verify input match
    input_ttnn = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils.DeviceGetter.get_device((1, 1)),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    input_torch = ttnn.to_torch(input_ttnn)
    input_match = torch.allclose(input_torch, x1)
    logger.info(f"Input match: {input_match}")

    # Verify weights match
    weights_torch = ttnn.to_torch(v1[1])

    # Verify bias match
    bias_torch = ttnn.to_torch(v1[0]).squeeze()

    # Verify weights match
    weights_match = torch.allclose(weights_torch, cpu_model.linear_2.weight.data)
    logger.info(f"Weights match: {weights_match}")

    # Verify bias match
    bias_match = torch.allclose(bias_torch, cpu_model.linear_2.bias.data)
    logger.info(f"Bias match: {bias_match}")

    assert input_match, "Input does not match!"
    assert weights_match, "Weights do not match!"
    assert bias_match, "Bias does not match!"

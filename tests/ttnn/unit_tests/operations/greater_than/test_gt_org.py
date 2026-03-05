import ttnn
import tests.ttnn.unit_tests.operations.greater_than.utils as utils
import torch
import torch.nn.functional as F
from loguru import logger

_CONST_EVAL_CACHE = {}


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=0.0099999997764825821,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_0 = [ttnn_full_0]
    return util_create_list_0


def _main(input):
    global _CONST_EVAL_CACHE
    const_0 = main_const_eval_0
    const_1 = "main_const_eval_0"
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(
        const_0, _CONST_EVAL_CACHE, const_1
    )
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    ttnn_to_layout_0 = ttnn.to_layout(
        input[0],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input[0], False)
    ttnn_softmax_0 = ttnn.softmax(
        ttnn_to_layout_0,
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_slice_0 = ttnn.slice(
        ttnn_softmax_0,
        [0, 0, 1],
        [1, 8732, 2],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_softmax_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_slice_0,
        [8732],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_0, False)
    ttnn_gt_0 = ttnn.gt(
        ttnn_reshape_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    util_create_list_1 = [ttnn_gt_0]
    return util_create_list_1


def load_inputs_for__main():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_2 = [utils_load_tensor_0]
    return util_create_list_2


class Cpu_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score_thresh = 0.01

    def forward(self, cls_logits):
        pred_scores = F.softmax(cls_logits, dim=-1)
        for scores in pred_scores:
            for label in range(1, 2):
                score = scores[:, label]
                keep_idxs = score > self.score_thresh
        return keep_idxs


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_float = x.to(torch.float64) if x.dtype != torch.float64 else x
    y_float = y.to(torch.float64) if y.dtype != torch.float64 else y

    x_flat, y_flat = x_float.flatten(), y_float.flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    if denom == 0:
        return float("nan")
    else:
        return ((vx @ vy) / denom).item()


def test_gt_org():
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_output = _main(load_inputs_for__main_0)

    # CPU run
    logger.info("Running CPU inference...")
    cpu_model = Cpu_model()
    cpu_model.eval()

    # Load the same input tensor for CPU inference
    logger.info("Loading input tensor for CPU inference...")
    tt_input_ttnn = ttnn.load_tensor("arg0.tensorbin")
    cpu_input = ttnn.to_torch(tt_input_ttnn).to(torch.float32)

    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)

    # Compare outputs
    logger.info("=== Output Comparison ===")
    tt_output_torch = ttnn.to_torch(tt_output[0])

    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"Output PCC: {pcc}")

import ttnn
import tests.ttnn.unit_tests.operations.silu.utils as utils
import torch
import torch.nn as nn
from loguru import logger
from tests.ttnn.unit_tests.operations.silu.model_utils import attempt_load
import urllib.request

_CONST_EVAL_CACHE = {}


def _main(input):
    ttnn_to_layout_0 = ttnn.to_layout(
        input[0],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input[0], False)
    ttnn_silu_0 = ttnn.silu(
        ttnn_to_layout_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    util_create_list_0 = [ttnn_silu_0]
    return util_create_list_0


def load_inputs_for__main():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_1 = [utils_load_tensor_0]
    return util_create_list_1


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


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.act = model.model[22].cv3[1][0].act

    def forward(self, x):
        x = self.act(x)
        return x

def test_silu_org():
    # tt run
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_input_torch = ttnn.to_torch(load_inputs_for__main_0[0])  # before _main (it deallocates input)
    _main_0 = _main(load_inputs_for__main_0)
    tt_output = _main_0[0]

    # cpu run
    cpu_input = torch.load('act_ip.pt', map_location="cpu")

    # Compare inputs
    logger.info("\n=== Input Comparison ===")
    logger.info("tt_input_torch: {}", tt_input_torch)
    logger.info("tt_input_torch.shape: {}", tt_input_torch.shape)
    logger.info("tt_input_torch.dtype: {}", tt_input_torch.dtype)

    logger.info("cpu_input: {}", cpu_input)
    logger.info("cpu_input.shape: {}", cpu_input.shape)
    logger.info("cpu_input.dtype: {}", cpu_input.dtype)

    input_allclose = torch.allclose(tt_input_torch, cpu_input)
    logger.info(f"Input allclose: {input_allclose}")
    
    # Construct weights URL dynamically from variant
    weights_url = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s-converted.pt"
    weight_path,_ = urllib.request.urlretrieve(weights_url)

    # Load model
    model = attempt_load(weight_path, "cpu", inplace=True, fuse=True)
    model.eval()
    model.to(torch.bfloat16)
    
    model = Wrapper(model)
    model.eval()
    logger.info("model={}",model)
    
    with torch.no_grad():
        cpu_output = model(cpu_input)

    # Compare outputs
    logger.info("\n=== Output Comparison ===")
    tt_output_torch = ttnn.to_torch(tt_output)

    logger.info("tt_output_torch: {}", tt_output_torch)
    logger.info("tt_output_torch.shape: {}", tt_output_torch.shape)
    logger.info("tt_output_torch.dtype: {}", tt_output_torch.dtype)

    logger.info("cpu_output: {}", cpu_output)
    logger.info("cpu_output.shape: {}", cpu_output.shape)
    logger.info("cpu_output.dtype: {}", cpu_output.dtype)

    # Compute PCC
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"Output PCC: {pcc}")


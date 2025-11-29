import ttnn
import tests.ttnn.unit_tests.operations.topk_indices.utils as utils
from scipy.stats import pearsonr
import torch
from loguru import logger


def _main(input):
    input_0 = input[0]
    v_1, v_2 = ttnn.sort(
        input_0,
        1,
        True,
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v_1, False)
    ttnn.deallocate(input_0, False)
    ttnn_typecast_0 = ttnn.typecast(
        v_2,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v_2, False)
    ttnn_slice_0 = ttnn.slice(
        ttnn_typecast_0,
        [0, 0],
        [1, 100],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    util_create_list_0 = [ttnn_slice_0]
    return util_create_list_0


def load_inputs_for__main():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_1 = [utils_load_tensor_0]
    return util_create_list_1


class TopKIndices(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return torch.topk(x, self.k)[1]


def compute_pcc(tt, cpu):
    a = tt.detach().cpu().numpy().flatten()
    b = cpu.detach().cpu().numpy().flatten()
    pcc, _ = pearsonr(a, b)

    return pcc


def test_sort_slice():
    # tt run
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_output = _main(load_inputs_for__main_0)

    tt_output = ttnn.to_torch(tt_output[0])

    # cpu run
    model = TopKIndices(k=100)
    model.eval()
    x = torch.load("topk_ip.pt", map_location="cpu")
    cpu_output = model(x)

    # pcc check
    pcc_values = compute_pcc(tt_output[0].float(), cpu_output.float())

    logger.info(f"PCC : {pcc_values}")

    # Verify input match
    input_ttnn = utils.load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils.DeviceGetter.get_device((1, 1)),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    input_torch = ttnn.to_torch(input_ttnn)
    input_match = torch.allclose(input_torch, x)
    logger.info(f"Input match: {input_match}")

    # input details
    torch.set_printoptions(threshold=float("inf"), precision=6, sci_mode=False, linewidth=200)

    logger.info("input_cpu=\n{}", x)
    logger.info("input_tt=\n{}", input_torch)

    logger.info("input_cpu.shape={}", x.shape)
    logger.info("input_tt.shape={}", input_torch.shape)

    logger.info("input_cpu.dtype={}", x.dtype)
    logger.info("input_tt.dtype={}", input_torch.dtype)

    # output details

    logger.info("cpu_output=\n{}", cpu_output)
    logger.info("tt_output[0]=\n{}", tt_output[0])

    logger.info("cpu_output.shape={}", cpu_output.shape)
    logger.info("tt_output[0].shape={}", tt_output[0].shape)

    logger.info("cpu_output.dtype={}", cpu_output.dtype)
    logger.info("tt_output[0].dtype={}", tt_output[0].dtype)

import ttnn
import torch
import tests.ttnn.unit_tests.operations.topk.utils as utils

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
    v_0, v_1 = ttnn.sort(
        ttnn_to_layout_0,
        1,
        True,
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_0, False)
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_typecast_0 = ttnn.typecast(
        v_1,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_1, False)
    ttnn_slice_0 = ttnn.slice(
        ttnn_typecast_0,
        [0, 0],
        [1, 300],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    util_create_list_0 = [ttnn_slice_0]
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
    x_flat, y_flat = x.flatten().to(torch.float64), y.flatten().to(torch.float64)
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return ((vx @ vy) / denom).item()


def test_topk_org():
    # Load input on host for CPU inference
    raw_host = utils.load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    input_torch_bf16 = ttnn.to_torch(raw_host)
    print(f"Input shape: {input_torch_bf16.shape}, dtype: {input_torch_bf16.dtype}")

    # CPU inference
    _, cpu_indices = torch.topk(input_torch_bf16, 300, dim=1)
    cpu_indices_i32 = cpu_indices.to(torch.int32)

    # Device inference
    load_inputs_for__main_0 = load_inputs_for__main()
    device_result = _main(load_inputs_for__main_0)
    device_indices_torch = ttnn.to_torch(device_result[0]).to(torch.int32)

    # Comparison
    pcc = compute_pcc(cpu_indices_i32, device_indices_torch)
    print(f"PCC: {pcc}")

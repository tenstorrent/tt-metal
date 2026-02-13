import os
import ttnn
import utils
import torch
import torch.nn.functional as F

_CONST_EVAL_CACHE = {}
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _main(input):
    ttnn_to_layout_0 = ttnn.to_layout(
        input[0],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input[0], False)
    ttnn_softmax_0 = ttnn.softmax(
        ttnn_to_layout_0,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    util_create_list_0 = [ttnn_softmax_0]
    return util_create_list_0


def load_inputs_for__main():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    tensor_path = os.path.join(_SCRIPT_DIR, "tensors", "arg0.tensorbin")
    utils_load_tensor_0 = utils.load_tensor(
        tensor_path,
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_1 = [utils_load_tensor_0]
    return util_create_list_1


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_flat, y_flat = x.flatten(), y.flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return ((vx @ vy) / denom).item()


class SoftmaxModule(torch.nn.Module):
    def forward(self, attention_scores):
        return F.softmax(attention_scores, dim=-1)


def main():
    # Load input tensor
    tensor_path = os.path.join(_SCRIPT_DIR, "tensors", "arg0.tensorbin")
    attention_scores = ttnn.to_torch(ttnn.load_tensor(tensor_path))

    # TT Run
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_output = _main(load_inputs_for__main_0)

    # CPU Run
    model = SoftmaxModule()
    model.eval()

    # CPU Inference
    with torch.no_grad():
        cpu_output = model(attention_scores)

    # Compare results
    tt_output_torch = ttnn.to_torch(tt_output[0])
    tt_output_fp64 = tt_output_torch.to(torch.float64)
    cpu_output_fp64 = cpu_output.to(torch.float64)

    pcc = compute_pcc(tt_output_fp64, cpu_output_fp64)
    print(f"Output PCC: {pcc}")

    # Assert PCC threshold
    pcc_threshold = 0.99
    assert pcc >= pcc_threshold, f"PCC check failed: {pcc} < {pcc_threshold}"
    print(f"PCC check passed: {pcc} >= {pcc_threshold}")


if __name__ == "__main__":
    main()

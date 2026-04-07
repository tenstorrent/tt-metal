import ttnn
import torch

device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))


def layer_norm_pre_all_gather(input):
    var_0 = input[0]
    ttnn_layer_norm_pre_all_gather_0 = ttnn.layer_norm_pre_all_gather(
        var_0,
        dtype=ttnn.DataType.BFLOAT16,
        residual_input_tensor=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        program_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        recip_tensor=None,
    )
    ttnn.deallocate(var_0, False)
    return [ttnn_layer_norm_pre_all_gather_0]


def create_ones_ttnn():
    ttnn_ones_0 = ttnn.ones(
        shape=ttnn.Shape([1, 1, 37, 72]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=device
    )
    print("TTNN input: ", ttnn_ones_0)
    return [ttnn_ones_0]


def create_ones_torch_to_ttnn():
    torch_ones = torch.ones((1, 1, 37, 72), dtype=torch.bfloat16)
    ttnn_ones_0 = ttnn.from_torch(torch_ones, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=device)
    print("TTNN input: ", ttnn_ones_0)
    return [ttnn_ones_0]


def main():
    inputs_ones_torch_to_ttnn = create_ones_torch_to_ttnn()
    inputs_ones_ttnn = create_ones_ttnn()
    layer_norm_pre_all_gather_0 = layer_norm_pre_all_gather(inputs_ones_torch_to_ttnn)
    layer_norm_pre_all_gather_1 = layer_norm_pre_all_gather(inputs_ones_ttnn)
    layer_norm_pre_all_gather_0_torch = ttnn.to_torch(layer_norm_pre_all_gather_0[0])
    layer_norm_pre_all_gather_1_torch = ttnn.to_torch(layer_norm_pre_all_gather_1[0])
    print("Output from TTNN with torch input: ", layer_norm_pre_all_gather_0_torch)
    print("Output from TTNN with TTNN input: ", layer_norm_pre_all_gather_1_torch)
    assert torch.allclose(
        layer_norm_pre_all_gather_0_torch, layer_norm_pre_all_gather_1_torch, rtol=1e-3, atol=1e-3
    ), "Outputs do not match!"
    return 0


if __name__ == "__main__":
    main()

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 1, 16],
    ],
)
def test_tanh_range(device, shape):
    torch_input_tensor_a = torch.tensor(
        [
            [
                [
                    [
                        -0.1,
                        0.25,
                        0.75,
                        -0.8359375,
                        -0.5,
                        0.9,
                        -2,
                        -3,
                        -1.5,
                        -2.5,
                        -3.5,
                        -3.75,
                        -3.359375,
                        -1.8828125,
                        -3.255,
                        -4,
                    ]
                ]
            ]
        ],
        dtype=torch.bfloat16,
    )
    torch_output_tensor = torch.tanh(torch_input_tensor_a)
    torch.set_printoptions(linewidth=200, threshold=10000, precision=10, sci_mode=False, edgeitems=17)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.tanh(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG, accuracy=True)
    # output_tensor = ttnn.tanhshrink(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    output_tensor = ttnn.to_torch(output_tensor)

    # for i in range(1):            # Batch size
    #     for j in range(1):        # Channels
    #         for k in range(shape[-2]):   # Height
    #             for l in range(shape[-1]):  # Width
    #                 print(f"{i}-{j}-{k}-{l} input: {torch_input_tensor_a[i][j][k][l]} \t TT_out: {output_tensor[i][j][k][l]} \t torch: {torch_output_tensor[i][j][k][l]} \n")

    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    print("pcc_msg", pcc_msg)  # pcc_msg 0.9997324342481332
    assert pcc


# 0-0-0-0 input: -0.10009765625    TT_out: TorchTensor(-0.0996093750, dtype=torch.bfloat16)        torch: -0.099609375

# 0-0-0-1 input: 0.25      TT_out: TorchTensor(0.2451171875, dtype=torch.bfloat16)         torch: 0.2451171875

# 0-0-0-2 input: 0.75      TT_out: TorchTensor(0.6406250000, dtype=torch.bfloat16)         torch: 0.63671875

# 0-0-0-3 input: -0.8359375        TT_out: TorchTensor(-0.6875000000, dtype=torch.bfloat16)        torch: -0.68359375

# 0-0-0-4 input: -0.5      TT_out: TorchTensor(-0.4628906250, dtype=torch.bfloat16)        torch: -0.462890625

# 0-0-0-5 input: 0.8984375         TT_out: TorchTensor(0.7148437500, dtype=torch.bfloat16)         torch: 0.71484375

# 0-0-0-6 input: -2.0      TT_out: TorchTensor(-1., dtype=torch.bfloat16)          torch: -0.96484375

# 0-0-0-7 input: -3.0      TT_out: TorchTensor(-1., dtype=torch.bfloat16)          torch: -0.99609375

# 0-0-0-8 input: -1.5      TT_out: TorchTensor(-0.9531250000, dtype=torch.bfloat16)        torch: -0.90625

# 0-0-0-9 input: -2.5      TT_out: TorchTensor(-1., dtype=torch.bfloat16)          torch: -0.98828125

# 0-0-0-10 input: -3.5     TT_out: TorchTensor(-1., dtype=torch.bfloat16)          torch: -1.0

# 0-0-0-11 input: -3.75    TT_out: TorchTensor(-1., dtype=torch.bfloat16)          torch: -1.0

# 0-0-0-12 input: -3.359375        TT_out: TorchTensor(-1., dtype=torch.bfloat16)          torch: -0.99609375

# 0-0-0-13 input: -1.8828125       TT_out: TorchTensor(-0.9882812500, dtype=torch.bfloat16)        torch: -0.953125

# 0-0-0-14 input: -3.25    TT_out: TorchTensor(-1., dtype=torch.bfloat16)          torch: -0.99609375

# 0-0-0-15 input: -4.0     TT_out: TorchTensor(-1., dtype=torch.bfloat16)          torch: -1.0

# pcc_msg 0.9997324342481332


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([2, 4, 320, 1024])),
    ),
)
def test_tanh_accuracy(device, input_shapes):
    torch_input_tensor = torch.rand((input_shapes), dtype=torch.bfloat16) * 2 - 1
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.tanh(input_tensor, accuracy=True)
    output_tensor = ttnn.to_torch(output)
    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    print("pcc_msg", pcc_msg)  # pcc_msg 0.9997324342481332
    assert pcc

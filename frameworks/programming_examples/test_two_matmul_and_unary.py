import torch
from pymetal import ttlib


def test_two_matmul_and_unary():
    # Create and initialize device on pcie slot 0
    device = ttlib.device.CreateDevice(ttlib.device.Arch.GRAYSKULL, 0)
    ttlib.device.InitializeDevice(device)
    host = ttlib.device.GetHost()

    torch.manual_seed(1234)

    # batch * C is the true batch size for the inputs
    batch = 2
    C = 1
    M = 32
    K = 64
    N = 96

    # We will run matmul on input_0 and input_1, so the W dim of input_0 must match the H dim of input_1 (K)
    # We will then run matmul on the output and input_2, so the W dim of the output must match the H dim of input_2 (N)
    input_0 = torch.randn((batch, C, M, K))
    input_1 = torch.randn((batch, C, K, N))
    input_2 = torch.randn((batch, C, N, M))

    # Create the tensor on host, convert to tile layout as ops expect inputs in tile layout, and then move to device
    input_t0 = (
        ttlib.tensor.Tensor(
            input_0.reshape(-1).tolist(),
            input_0.size(),
            ttlib.tensor.DataType.BFLOAT16,
            ttlib.tensor.Layout.ROW_MAJOR,
        )
        .to(ttlib.tensor.Layout.TILE)
        .to(device)
    )

    input_t1 = (
        ttlib.tensor.Tensor(
            input_1.reshape(-1).tolist(),
            input_1.size(),
            ttlib.tensor.DataType.BFLOAT16,
            ttlib.tensor.Layout.ROW_MAJOR,
        )
        .to(ttlib.tensor.Layout.TILE)
        .to(device)
    )

    input_t2 = (
        ttlib.tensor.Tensor(
            input_2.reshape(-1).tolist(),
            input_2.size(),
            ttlib.tensor.DataType.BFLOAT16,
            ttlib.tensor.Layout.ROW_MAJOR,
        )
        .to(ttlib.tensor.Layout.TILE)
        .to(device)
    )

    # Run batched matrix mult on input 0 and input 1
    # Output is size [batch, C, M, N]
    output_t0 = ttlib.tensor.bmm(input_t0, input_t1)

    # Run batched matrix mult on bmm out 0 and input 2
    # Output is size [batch, C, M, M]
    output_t1 = ttlib.tensor.bmm(output_t0, input_t2)

    # Run gelu on bmm out 1
    # Output is size [batch, C, M, M]
    output_t2 = ttlib.tensor.gelu(output_t1)

    # Move final output from device back to host, convert back to row major, then convert to Pytorch tensor
    output_t2 = output_t2.to(host).to(ttlib.tensor.Layout.ROW_MAJOR)
    final_output = torch.Tensor(output_t2.data()).reshape(output_t2.shape())

    ttlib.device.CloseDevice(device)

    print(final_output)

    return


if __name__ == "__main__":
    test_two_matmul_and_unary()

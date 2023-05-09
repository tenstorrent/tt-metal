from libs import tt_lib as ttm


def tt_baddbmm(device, input, batch1, batch2, beta=1.0, alpha=1.0) -> ttm.tensor.Tensor:

    # print(f"input shape {input.shape()}")
    # print(f"batch1 shape {batch1.shape}")
    # print(f"batch2 shape {batch2.shape}")

    if beta != 1.0:
        input = ttm.tensor.mul(beta, input)

    tmp = ttm.tensor.bmm(batch1, batch2)

    if alpha != 1.0:
        tmp = ttm.tensor.mul(alpha, tmp)

    result = ttm.tensor.add(input, tmp)

    return result

import tt_lib
import python_api_testing.models.bloom.bloom_utils as bloom_utils


def tt_baddbmm(
    device, input, batch1, batch2, beta=1.0, alpha=1.0
) -> tt_lib.tensor.Tensor:
    if beta != 1.0:
        input = tt_lib.tensor.mul(beta, input)

    tmp = bloom_utils.tt_bmm(batch1, batch2, device)

    if alpha != 1.0:
        tmp = tt_lib.tensor.mul(alpha, tmp)

    result = tt_lib.tensor.add(input, tmp)

    return result

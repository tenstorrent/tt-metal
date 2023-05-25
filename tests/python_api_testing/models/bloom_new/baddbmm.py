from libs import tt_lib as ttm
import python_api_testing.models.bloom_new.bloom_utils as bloom_utils


def tt_baddbmm(device, input, batch1, batch2, beta=1.0, alpha=1.0) -> ttm.tensor.Tensor:

    if beta != 1.0:
        input = ttm.tensor.mul(beta, input)

    tmp = bloom_utils.tt_bmm(batch1, batch2, device)

    if alpha != 1.0:
        tmp = ttm.tensor.mul(alpha, tmp)

    result = ttm.tensor.add(input, tmp)

    return result

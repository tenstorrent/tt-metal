import torch
import tt_lib
import python_api_testing.models.bloom.bloom_utils as bloom_utils


def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def tt_bloom_gelu_forward(x, device):
    z = x

    k1 = torch.full(x.shape(), 0.5)
    tt_k1 = bloom_utils.torch2tt_tensor(k1, device)

    k2 = torch.full(x.shape(), 0.044715)
    tt_k2 = bloom_utils.torch2tt_tensor(k2, device)

    k3 = torch.full(x.shape(), 0.79788456)
    tt_k3 = bloom_utils.torch2tt_tensor(k3, device)

    # 0.5*x
    factor1 = tt_lib.tensor.mul(tt_k1, z)  # exp(z)

    # x*x
    pow2 = tt_lib.tensor.mul(z, z)

    # (x + 0.044715 * torch.pow(x, 3)))
    # torch.pow(x, 3))
    pow3 = tt_lib.tensor.mul(pow2, z)
    factor3 = tt_lib.tensor.mul(tt_k2, pow3)

    # (x + 0.044715 * torch.pow(x, 3)))
    factor3 = tt_lib.tensor.add(factor3, z)

    sumtanh = tt_lib.tensor.mul(tt_k3, factor3)
    tanh = tt_lib.tensor.tanh(sumtanh)

    k4 = torch.full(x.shape(), 1.0)
    tt_k4 = bloom_utils.torch2tt_tensor(k4, device)

    total = tt_lib.tensor.add(tt_k4, tanh)
    output = tt_lib.tensor.mul(factor1, total)

    return output

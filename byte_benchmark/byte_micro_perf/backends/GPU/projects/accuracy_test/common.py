import torch
from collections import namedtuple


BITS_INFO = namedtuple(
    "BITS_INFO",
    [
        "total_bits",
        "exponent_bits",
        "mantissa_bits",
        "equivalent_dtype",
        "exponent_mask",
        "mantissa_mask",
        "exponent_offset",
        "mantissa_denominator",
    ],
)

float_info_mapping = {
    torch.float64: BITS_INFO(64, 11, 52, torch.int64, (1 << 11) - 1, (1 << 52) - 1, (1 << (11 - 1)) - 1, 1 << 52),
    torch.float32: BITS_INFO(32, 8, 23, torch.int32, (1 << 8) - 1, (1 << 23) - 1, (1 << (8 - 1)) - 1, 1 << 23),
    torch.float16: BITS_INFO(16, 5, 10, torch.int16, (1 << 5) - 1, (1 << 10) - 1, (1 << (5 - 1)) - 1, 1 << 10),
    torch.bfloat16: BITS_INFO(16, 8, 7, torch.int16, (1 << 8) - 1, (1 << 7) - 1, (1 << (8 - 1)) - 1, 1 << 7),
}


def create_float(sign, exponent, mantissa, torch_dtype=torch.float32, shape=(1,)):
    if torch_dtype not in float_info_mapping:
        print(f"{torch_dtype} not supported")
        return None
    bits_info = float_info_mapping[torch_dtype]

    exponent_tensor = torch.full(shape, exponent, dtype=bits_info.equivalent_dtype) << bits_info.mantissa_bits
    mantissa_tensor = torch.full(shape, mantissa, dtype=bits_info.equivalent_dtype)
    value_tensor = exponent_tensor + mantissa_tensor
    if sign:
        value_tensor *= -1
    float_tensor = value_tensor.view(torch_dtype)
    return float_tensor


def parse_float(data):
    torch_dtype = data.dtype
    if torch_dtype not in float_info_mapping:
        print(f"{torch_dtype} not supported")
        return None
    bits_info = float_info_mapping[torch_dtype]

    viewed_data = data.view(bits_info.equivalent_dtype)
    exponent_tensor = (viewed_data >> bits_info.mantissa_bits) & bits_info.exponent_mask
    mantissa_tensor = viewed_data & bits_info.mantissa_mask

    return exponent_tensor, mantissa_tensor


def parse_float_value(data):
    torch_dtype = data.dtype
    if torch_dtype not in float_info_mapping:
        print(f"{torch_dtype} not supported")
        return None
    bits_info = float_info_mapping[torch_dtype]

    exponent_tensor, mantissa_tensor = parse_float(data)
    exponent = exponent_tensor.flatten()[0].item()
    mantissa = mantissa_tensor.flatten()[0].item()
    true_exponent = exponent - bits_info.exponent_offset
    true_fraction = true_exponent - bits_info.mantissa_bits

    return exponent, mantissa, true_exponent, true_fraction

import os
import torch
from datetime import timedelta
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import namedtuple
import traceback

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

dtype_mapping = {
    torch.float64: BITS_INFO(64, 11, 52, torch.int64, (1 << 11) - 1, (1 << 52) - 1, (1 << (11 - 1)) - 1, 1 << 52),
    torch.float32: BITS_INFO(32, 8, 23, torch.int32, (1 << 8) - 1, (1 << 23) - 1, (1 << (8 - 1)) - 1, 1 << 23),
    torch.float16: BITS_INFO(16, 5, 10, torch.int16, (1 << 5) - 1, (1 << 10) - 1, (1 << (5 - 1)) - 1, 1 << 10),
    torch.bfloat16: BITS_INFO(16, 8, 7, torch.int16, (1 << 8) - 1, (1 << 7) - 1, (1 << (8 - 1)) - 1, 1 << 7),
}


def create_float(sign, exponent, mantissa, torch_dtype=torch.float32, shape=(1,)):
    if torch_dtype not in dtype_mapping:
        print(f"{torch_dtype} not supported")
        return None
    bits_info = dtype_mapping[torch_dtype]
    exponent_tensor = torch.full(shape, exponent, dtype=bits_info.equivalent_dtype) << bits_info.mantissa_bits
    mantissa_tensor = torch.full(shape, mantissa, dtype=bits_info.equivalent_dtype)
    value_tensor = exponent_tensor + mantissa_tensor
    if sign:
        value_tensor *= -1
    float_tensor = value_tensor.view(torch_dtype)
    return float_tensor


def parse_float(data):
    torch_dtype = data.dtype
    bits_info = dtype_mapping[torch_dtype]

    viewed_data = data.view(bits_info.equivalent_dtype)
    exponent_tensor = (viewed_data >> bits_info.mantissa_bits) & bits_info.exponent_mask
    mantissa_tensor = viewed_data & bits_info.mantissa_mask

    return exponent_tensor, mantissa_tensor


def parse_float_value(data):
    torch_dtype = data.dtype
    bits_info = dtype_mapping[torch_dtype]

    exponent_tensor, mantissa_tensor = parse_float(data)
    exponent = exponent_tensor.flatten()[0].item()
    mantissa = mantissa_tensor.flatten()[0].item()
    true_exponent = exponent - bits_info.exponent_offset
    true_fraction = true_exponent - bits_info.mantissa_bits

    return exponent, mantissa, true_exponent, true_fraction


def subprocess_func(rank: int, *args):
    try:
        (world_size,) = args
        torch.cuda.set_device(rank)

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "49373"
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank, timeout=timedelta(seconds=1800))

        default_sign = 0
        default_exponent = 127
        default_mantissa = 64

        # 八个数
        # 0: 大数，100,0000，不断增加exponent
        # 1: 小数，在大数值域体现为0011
        # 2-7: 小数，在大数值域体现为0001
        # 顺序累加后：最终值为1001，且1001会不断后移，直到超过mantissa表示范围，会看出不同累加位数的差异。
        values = [
            # m: 100,0000
            create_float(default_sign, default_exponent, default_mantissa, torch.bfloat16, shape=[1024]),
            # m: 0011, shifting
            create_float(default_sign, default_exponent + 1, 64, torch.bfloat16, shape=[1024]),
            # m: 0001, shifting
            create_float(default_sign, default_exponent, 0, torch.bfloat16, shape=[1024]),
            create_float(default_sign, default_exponent, 0, torch.bfloat16, shape=[1024]),
            create_float(default_sign, default_exponent, 0, torch.bfloat16, shape=[1024]),
            create_float(default_sign, default_exponent, 0, torch.bfloat16, shape=[1024]),
            create_float(default_sign, default_exponent, 0, torch.bfloat16, shape=[1024]),
            create_float(default_sign, default_exponent, 0, torch.bfloat16, shape=[1024]),
        ]

        for exponent in range(default_exponent, default_exponent + 13):
            values[0] = create_float(default_sign, exponent, default_mantissa, torch.bfloat16, shape=[1024])

            # float64
            fp64_golden_value = torch.zeros_like(values[0], dtype=torch.float64)
            for i in range(world_size):
                fp64_golden_value += values[i].type(torch.float64)

            fp64_exponent, fp64_mantissa = parse_float(fp64_golden_value)
            fp64_exponent = fp64_exponent.flatten()[0].item()
            fp64_mantissa = fp64_mantissa.flatten()[0].item()
            true_fp64_exponent = fp64_exponent - dtype_mapping[torch.float64].exponent_offset
            true_fp64_fraction = true_fp64_exponent - dtype_mapping[torch.float64].mantissa_bits

            # float32
            fp32_golden_value = torch.zeros_like(values[0], dtype=torch.float32)
            for i in range(world_size):
                fp32_golden_value += values[i].type(torch.float32)

            fp32_exponent, fp32_mantissa = parse_float(fp32_golden_value)
            fp32_exponent = fp32_exponent.flatten()[0].item()
            fp32_mantissa = fp32_mantissa.flatten()[0].item()
            true_fp32_exponent = fp32_exponent - dtype_mapping[torch.float32].exponent_offset
            true_fp32_fraction = true_fp32_exponent - dtype_mapping[torch.float32].mantissa_bits

            # float32累加 cast回 bfloat16
            cast_bf16_golden_value = fp32_golden_value.type(torch.bfloat16)
            cast_bf16_exponent, cast_bf16_mantissa = parse_float(cast_bf16_golden_value)
            cast_bf16_exponent = cast_bf16_exponent.flatten()[0].item()
            cast_bf16_mantissa = cast_bf16_mantissa.flatten()[0].item()
            true_cast_bf16_exponent = cast_bf16_exponent - dtype_mapping[torch.bfloat16].exponent_offset
            true_cast_bf16_fraction = true_cast_bf16_exponent - dtype_mapping[torch.bfloat16].mantissa_bits

            # bfloat16
            bf16_golden_value = torch.zeros_like(values[0], dtype=torch.bfloat16).cuda()
            for i in range(world_size):
                bf16_golden_value += values[i].type(torch.bfloat16).cuda()

            bf16_exponent, bf16_mantissa = parse_float(bf16_golden_value)
            bf16_exponent = bf16_exponent.flatten()[0].item()
            bf16_mantissa = bf16_mantissa.flatten()[0].item()
            true_bf16_exponent = bf16_exponent - dtype_mapping[torch.bfloat16].exponent_offset
            true_bf16_fraction = true_bf16_exponent - dtype_mapping[torch.bfloat16].mantissa_bits

            # nccl
            data = values[rank].cuda()
            # dist.reduce(data, 0)
            dist.all_reduce(data)
            nccl_exponent, nccl_mantissa = parse_float(data)
            nccl_exponent = nccl_exponent.flatten()[0].item()
            nccl_mantissa = nccl_mantissa.flatten()[0].item()
            true_nccl_exponent = nccl_exponent - dtype_mapping[torch.bfloat16].exponent_offset
            true_nccl_fraction = true_nccl_exponent - dtype_mapping[torch.bfloat16].mantissa_bits

            if rank == 0:
                print(f"e={exponent}, m={default_mantissa}, exponent={exponent-127}: ")
                print(
                    f"fp64:\t\t{fp64_golden_value.flatten()[0].item()}\t{fp64_exponent}\t{bin(fp64_mantissa)} ({dtype_mapping[torch.float64].mantissa_bits}) (e^{true_fp64_exponent}={2 ** true_fp64_exponent}) (e^{true_fp64_fraction}={2 ** true_fp64_fraction})"
                )
                print(
                    f"fp32:\t\t{fp32_golden_value.flatten()[0].item()}\t{fp32_exponent}\t{bin(fp32_mantissa)} ({dtype_mapping[torch.float32].mantissa_bits}) (e^{true_fp32_exponent}={2 ** true_fp32_exponent}) (e^{true_fp32_fraction}={2 ** true_fp32_fraction})"
                )
                print(
                    f"cast bf16:\t{cast_bf16_golden_value.flatten()[0].item()}\t{cast_bf16_exponent}\t{bin(cast_bf16_mantissa)} ({dtype_mapping[torch.bfloat16].mantissa_bits}) (e^{true_cast_bf16_exponent}={2 ** true_cast_bf16_exponent}) (e^{true_cast_bf16_fraction}={2 ** true_cast_bf16_fraction})"
                )
                print(
                    f"bf16:\t\t{bf16_golden_value.flatten()[0].item()}\t{bf16_exponent}\t{bin(bf16_mantissa)} ({dtype_mapping[torch.bfloat16].mantissa_bits}) (e^{true_bf16_exponent}={2 ** true_bf16_exponent}) (e^{true_bf16_fraction}={2 ** true_bf16_fraction})"
                )
                print(
                    f"nccl bf16:\t{data.cpu().flatten()[0].item()}\t{nccl_exponent}\t{bin(nccl_mantissa)} ({dtype_mapping[torch.bfloat16].mantissa_bits}) (e^{true_nccl_exponent}={2 ** true_nccl_exponent}) (e^{true_nccl_fraction}={2 ** true_nccl_fraction})"
                )
                print("")
    except Exception:
        traceback.print_exc()

    dist.destroy_process_group()


if __name__ == "__main__":
    device_num = 8
    mp.set_start_method("spawn", force=True)
    _subprocess = mp.spawn(fn=subprocess_func, args=(device_num,), nprocs=device_num, join=False, daemon=False)
    for process in _subprocess.processes:
        process.join()

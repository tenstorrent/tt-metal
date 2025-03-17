# filepath: /home/ubuntu/projects/tt-metal/test_bw_gelu_benchmark.py
import time
import math
import torch
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range
import timeit


# from 512 to 16k
INPUT_SHAPES = [
    torch.Size([1, 1, 32, 32]),
    torch.Size([1, 3, 32, 32]),
    torch.Size([1, 3, 128, 128]),
    # torch.Size([1, 3, 512, 512]),
    # torch.Size([1, 3, 1024, 1024]),
    # torch.Size([1, 3, 2048, 2048]),
    # torch.Size([1, 3, 4096, 4096]),
    # torch.Size([1, 3, 8192, 8192]),
    # torch.Size([1, 3, 16384, 16384]),
    # torch.Size([1, 3, 22912, 22912]),
]

BENCH_FUNCS = [
    (ttnn.gelu, "gelu"),
    (ttnn.silu, "silu"),
    (ttnn.tanh, "tanh"),
]


def benchmark_with_warmup(iterations):
    # Setup
    device_id = 0
    dispatch_core_type = ttnn.device.DispatchCoreType.ETH
    device = ttnn.open_device(
        device_id=device_id, l1_small_size=8192, dispatch_core_config=ttnn.device.DispatchCoreConfig(dispatch_core_type)
    )

    ttnn.enable_program_cache(device)  # Enable cache

    try:
        # Timed runs
        times = []
        _, tt_tensor = data_gen_with_range(INPUT_SHAPES[0], -100, 100, device, True)
        for func, name in BENCH_FUNCS:
            # Compile program
            for input_shape in INPUT_SHAPES:
                _, tt_tensor = data_gen_with_range(input_shape, -100, 100, device, True)
                # print(f"Warm up for func: {name}, shape: {input_shape}")
                _ = func(tt_tensor)
                start = time.time()
                for _ in range(iterations):
                    _ = func(tt_tensor)
                end = time.time()
                times.append(end - start)
                avg_time = sum(times) / len(times)
                avg_time = (end - start) / iterations
                print(f"Shape: {input_shape} Average {name} time over {iterations} runs: {avg_time*1000:.12f} ms")
                # print(timeit.timeit("ttnn.tanh(tt_tensor)", globals=globals(), number=100))

    finally:
        # Cleanup
        ttnn.close_device(device)


if __name__ == "__main__":
    benchmark_with_warmup(1000)

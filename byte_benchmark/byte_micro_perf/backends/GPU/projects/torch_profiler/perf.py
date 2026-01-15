import os
import torch
import shutil
import pathlib
import argparse

PWD_DIR = pathlib.Path.cwd()
FILE_DIR = pathlib.Path(__file__).parent.absolute()
PROFILING_DIR = FILE_DIR.joinpath("profiling")

os.chdir(FILE_DIR)


def trace_handler(prof):
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=-1))
    torch.profiler.tensorboard_trace_handler(f"{PROFILING_DIR}")(prof)


if __name__ == "__main__":
    device_count = torch.cuda.device_count()
    device_list = list(range(device_count))
    if device_count == 0:
        raise RuntimeError("No CUDA device found.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=7, choices=device_list)
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    args = parser.parse_args()

    if PROFILING_DIR.exists():
        shutil.rmtree(PROFILING_DIR)

    torch.cuda.set_device(args.device)

    # prepare inputs
    M = args.M
    K = args.K
    N = args.N
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
        on_trace_ready=trace_handler,
        with_stack=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        with_flops=True,
        record_shapes=True,
    ) as prof:
        for _ in range(20):
            for i in range(10):
                torch.matmul(a, b, out=c)
            torch.cuda.synchronize()
            prof.step()

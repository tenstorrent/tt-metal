"""Sweep channel_chunk_size for the fused KDA conv at Qwen3.6-27B's exact shapes.

T=32 is one prefill chunk; T=128 is decode's K*B user-major packed window."""

import argparse

import torch

import ttnn
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

KW, VW = 2048, 6144
C = 2 * KW + VW  # 10240


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=32)
    SEQ = parser.parse_args().sequence
    device = ttnn.open_device(device_id=0)
    try:
        g = torch.Generator().manual_seed(7)
        inp = torch.randn(1, SEQ, C, generator=g, dtype=torch.bfloat16)
        hist = torch.randn(1, 3, C, generator=g, dtype=torch.bfloat16)
        taps = [torch.randn(1, 1, C, generator=g, dtype=torch.bfloat16) for _ in range(4)]
        inp_tt = ttnn.from_torch(inp, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        hist_tt = ttnn.from_torch(hist, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        taps_tt = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in taps]
        ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        for chunk in (64, 128, 160, 256, 320, 512, 640, 1024, 1280, 2048):
            try:
                pc = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=chunk)
                for _ in range(3):
                    outs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
                        inp_tt,
                        hist_tt,
                        *taps_tt,
                        KW,
                        KW,
                        VW,
                        program_config=pc,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=ckc,
                    )
                    for o in outs:
                        ttnn.deallocate(o)
                ttnn.synchronize_device(device)

                def run():
                    return ttnn.experimental.kda.qkv_causal_conv1d_silu(
                        inp_tt,
                        hist_tt,
                        *taps_tt,
                        KW,
                        KW,
                        VW,
                        program_config=pc,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=ckc,
                    )

                times = []
                for _ in range(7):
                    outs, record = profile_realtime_program(device, run)
                    times.append(record["duration_ns"] / 1000.0)
                    for o in outs:
                        ttnn.deallocate(o)
                times.sort()
                print(
                    f"CHUNK {chunk:6d}  device_median_us={times[len(times)//2]:8.2f}  device_min_us={times[0]:8.2f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"CHUNK {chunk:6d}  FAILED: {str(exc).splitlines()[0][:110]}", flush=True)
    finally:
        ttnn.close_device(device)


main()

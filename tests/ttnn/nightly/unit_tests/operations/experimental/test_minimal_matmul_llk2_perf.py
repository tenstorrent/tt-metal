# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Device time of minimal_matmul on the 1.0 and the LLK 2.0 compute kernel (TT_MINIMAL_MATMUL_LLK2), at the Wan2.2 DiT's
# per-device shapes (480p, 16384 tokens per device on a 2x2 mesh). Each case logs one line,
# "LLK2_MM_PERF <case> <kernel> median_ns=<n> runs=<n>", and appends it to $LLK2_MM_PERF_OUT when that is set.

import os
import statistics

import pytest
import torch
import ttnn
from loguru import logger

from tt_py_test_utils_common.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

M = 16384
CASES = {
    "qkv": (5120, 7680, "bias"),
    "to_q_out": (5120, 2560, "bias"),
    "ff1_gelu_tanh": (5120, 6912, "gelu_tanh"),
    "ff2": (6912, 5120, "bias"),
    "out_addcmul": (5120, 2560, "addcmul"),
}
RUNS = 5


@pytest.mark.parametrize("kernel", ["llk1", "llk2"])
@pytest.mark.parametrize("case", list(CASES.keys()))
def test_llk2_minimal_matmul_perf(device, monkeypatch, case, kernel):
    monkeypatch.setenv("TT_MINIMAL_MATMUL_LLK2", "1" if kernel == "llk2" else "0")
    K, N, form = CASES[case]
    torch.manual_seed(0)

    def tensor(rows, cols):
        return ttnn.from_torch(
            torch.randn((rows, cols), dtype=torch.float32), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )

    x, w, bias = tensor(M, K), tensor(K, N), tensor(1, N)
    a, b = (tensor(M, N), tensor(1, N)) if form == "addcmul" else (None, None)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    config = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=2,
        subblock_w=2,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )

    def run():
        if form == "addcmul":
            return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x, w, 1.0, a, b, bias_tensor=bias, config=config, compute_kernel_config=compute_config
            )
        return ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=w,
            bias_tensor=bias,
            fused_activation=ttnn.UnaryOpType.GELU_TANH if form == "gelu_tanh" else None,
            compute_kernel_config=compute_config,
            config=config,
        )

    run()  # compile outside the profiler window (and in a mock prebuild, which stops at the next line)
    ttnn.synchronize_device(device)
    require_realtime_profiler("the LLK 2.0 minimal_matmul perf comparison")
    _, per_program = profile_realtime_program_merged(device, lambda: [run() for _ in range(RUNS)])
    durations = [entry["duration_ns"] for _, entry in sorted(per_program.items())]
    assert len(durations) == RUNS, f"expected {RUNS} programs, got {len(durations)}"
    median = statistics.median(durations)
    line = f"LLK2_MM_PERF {case} {kernel} median_ns={median:.0f} runs={RUNS}"
    logger.info(line)
    out = os.environ.get("LLK2_MM_PERF_OUT")
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")

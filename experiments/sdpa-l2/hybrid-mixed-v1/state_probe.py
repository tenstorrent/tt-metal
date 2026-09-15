# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native FP32 round-trip/update in a BF16-configured kernel."""

import json
import os
from pathlib import Path
import torch
import ttnn

HERE = Path(__file__).resolve().parent
PREFIX = "experiments/sdpa-l2/hybrid-mixed-v1/"


def invoke(device, values, repeats):
    x = ttnn.from_torch(values.reshape(1, 1, 96, 32), device=device, layout=ttnn.TILE_LAYOUT)
    assert torch.equal(ttnn.to_torch(x), values.reshape(1, 1, 96, 32)), "input upload mismatch"
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    specs = [(0, 3, ttnn.float32, 4096), (1, 1, ttnn.bfloat16, 2048), (16, 1, ttnn.float32, 4096)]
    cbs = [
        ttnn.CBDescriptor(
            total_size=n * s,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=d, page_size=s)],
        )
        for i, n, d, s in specs
    ]
    ra, wa = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    ra[0][0] = [x.buffer_address()]
    wa[0][0] = [y.buffer_address()]
    pd = ttnn._ttnn.program_descriptor
    modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
    modes[0] = pd.UnpackToDestMode.UnpackToDestFp32
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=bool(os.getenv("HYBRID_STATIC_CONTROL")),
        dst_full_sync_en=False,
        math_approx_mode=True,
    )
    config.unpack_to_dest_mode = modes
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "state_reader.cpp",
                core_ranges=grid,
                compile_time_args=[3] + ttnn.TensorAccessorArgs(x).get_compile_time_args(),
                runtime_args=ra,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "state_writer.cpp",
                core_ranges=grid,
                compile_time_args=[1] + ttnn.TensorAccessorArgs(y).get_compile_time_args(),
                runtime_args=wa,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "state_compute.cpp",
                core_ranges=grid,
                compile_time_args=[repeats],
                defines=[],
                config=config,
            ),
        ],
    )
    ttnn.generic_op([x, y], desc)
    result = ttnn.to_torch(y).reshape(32, 32).clone()
    ttnn.deallocate(x)
    ttnn.deallocate(y)
    return result


device = ttnn.open_device(device_id=0)
device.enable_program_cache()
try:
    records = []
    for seed in (1236, 1237, 1238):
        torch.manual_seed(seed)
        for pattern in ("roundtrip", "small_increment", "rescale", "cancellation"):
            x = torch.randn(32, 32)
            delta = torch.zeros_like(x)
            scale = torch.ones_like(x)
            repeats = 0 if pattern == "roundtrip" else 1
            if pattern == "small_increment":
                x.fill_(1)
                delta.fill_(2**-20)
                repeats = 512
            if pattern == "rescale":
                scale = (1 - torch.rand_like(x) * 0.1).bfloat16().float()
                delta = torch.randn_like(x) * 0.01
            if pattern == "cancellation":
                delta = -x + torch.randn_like(x) * 2**-20
            ref = x.clone()
            for _ in range(repeats):
                ref = (ref.double() * scale.double() + delta.double()).float()
            actual = invoke(device, torch.stack((x, delta, scale)), repeats)
            print("SAMPLES", actual.flatten()[:8].tolist(), ref.flatten()[:8].tolist(), flush=True)
            bad = actual.view(torch.int32) != ref.view(torch.int32)
            print(
                "BAD",
                actual[bad][:12].tolist(),
                ref[bad][:12].tolist(),
                actual.view(torch.int32)[bad][:12].tolist(),
                ref.view(torch.int32)[bad][:12].tolist(),
                flush=True,
            )
            record = dict(
                seed=seed,
                pattern=pattern,
                max_abs=(actual - ref).abs().max().item(),
                bit_mismatches=(actual.view(torch.int32) != ref.view(torch.int32)).sum().item(),
                finite=bool(torch.isfinite(actual).all()),
            )
            print(json.dumps(record), flush=True)
            records.append(record)
            assert record["finite"] and record["max_abs"] < 2**-20, record
            if pattern in ("roundtrip", "small_increment"):
                assert record["bit_mismatches"] == 0, record
    (HERE / "state-results.json").write_text(json.dumps(records, indent=2) + "\n")
finally:
    ttnn.close_device(device)

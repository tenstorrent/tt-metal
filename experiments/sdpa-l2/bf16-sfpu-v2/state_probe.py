# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise comparison against the frozen compensated-state implementation."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
PREFIX = "experiments/sdpa-l2/bf16-sfpu-v2/"


def invoke(device, x, pairs, reference):
    src = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 2 * pairs * 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cbs = [
        ttnn.CBDescriptor(
            total_size=n * 2048,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i, n in ((0, 3 * pairs + 1), (16, 2 * pairs))
    ]
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0], writer[0][0] = [src.buffer_address()], [out.buffer_address()]
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "state_reader.cpp",
                core_ranges=grid,
                compile_time_args=[3 * pairs + 1] + ttnn.TensorAccessorArgs(src).get_compile_time_args(),
                runtime_args=reader,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "state_writer.cpp",
                core_ranges=grid,
                compile_time_args=[2 * pairs] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "state_compute.cpp",
                core_ranges=grid,
                compile_time_args=[pairs],
                defines=[("PROBE_REFERENCE", "1")] if reference else [],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    fp32_dest_acc_en=False,
                    dst_full_sync_en=False,
                    math_approx_mode=True,
                ),
            ),
        ],
    )
    ttnn.generic_op([src, out], desc)
    result = ttnn.to_torch(out).clone()
    ttnn.deallocate(src)
    ttnn.deallocate(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    output = HERE / (args.label + ".jsonl")
    assert not output.exists()
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    try:
        with output.open("x") as log:
            for pairs in (1, 2):
                for seed in (1236, 1237, 1238):
                    for pattern in ("normal", "rescale", "cancellation", "wide_exponent", "round_boundary"):
                        torch.manual_seed(seed)
                        tiles = torch.randn(3 * pairs + 1, 32, 32)
                        for b in range(pairs):
                            tiles[3 * b + 1] *= 2**-9
                        tiles[-1].fill_(1)
                        if pattern in ("rescale", "cancellation", "wide_exponent"):
                            tiles[-1] = torch.exp(-torch.rand(32, 32) * 12).bfloat16().float()
                        if pattern == "cancellation":
                            for b in range(pairs):
                                tiles[3 * b + 2] = -tiles[3 * b].bfloat16().float() * tiles[-1]
                        if pattern == "wide_exponent":
                            for b in range(pairs):
                                scale = 2.0 ** torch.randint(-40, 41, (32, 32))
                                tiles[3 * b : 3 * b + 3] *= scale
                        if pattern == "round_boundary":
                            for b in range(pairs):
                                tiles[3 * b] = (1 + torch.randint(0, 128, (32, 32)) / 128) * torch.where(
                                    tiles[3 * b] < 0, -1, 1
                                )
                                tiles[3 * b + 1] = 0
                                tiles[3 * b + 2] = torch.tensor([1 / 256, -1 / 256, 1 / 512, -1 / 512]).repeat(32, 8)
                        x = tiles.bfloat16().reshape(1, 1, (3 * pairs + 1) * 32, 32)
                        old = invoke(device, x, pairs, True)
                        new = invoke(device, x, pairs, False)
                        mismatch = (old.view(torch.int16) != new.view(torch.int16)).sum().item()
                        row = dict(
                            pairs=pairs,
                            seed=seed,
                            pattern=pattern,
                            mismatches=mismatch,
                            max_abs=float((old.float() - new.float()).abs().max()),
                            output_sha256=hashlib.sha256(new.view(torch.int16).numpy().tobytes()).hexdigest(),
                        )
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        print(json.dumps(row), flush=True)
                        assert mismatch == 0, row
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

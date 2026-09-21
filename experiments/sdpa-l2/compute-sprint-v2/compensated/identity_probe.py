# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""All finite BF16 maxima: frozen correction(x-x) must pack to exact BF16 one."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import torch
import ttnn

import benchmark


def sources():
    result = benchmark.source_hashes()
    for path in (Path(__file__).resolve(), benchmark.HERE / "identity_probe/compute.cpp"):
        result[str(path.relative_to(benchmark.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity", choices=("LoFi", "HiFi2"), default="LoFi")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(8)
    raw = torch.arange(65536, dtype=torch.int32)
    raw = raw[(raw & 0x7f80) != 0x7f80]
    values = raw.to(torch.uint16).view(torch.bfloat16)
    assert values.numel() == 65280
    host = values[:, None].expand(-1, 32).clone().reshape(1, 1, -1, 32)
    original_hash = benchmark.tensor_hash(host)
    before = sources()
    device = ttnn.open_device(device_id=0, trace_region_size=8388608)
    try:
        src = ttnn.from_torch(host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = ttnn.allocate_tensor_on_device(host.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        count = host.numel() // 1024
        cbs = [ttnn.CBDescriptor(total_size=4096, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)])
            for i in (0, 16)]
        read, write, compute = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        read[0][0], write[0][0], compute[0][0] = [src.buffer_address(), 0, count], [out.buffer_address(), 0, count], [count]
        _, _, defines, _ = benchmark.CANONICAL.recipe("E")
        base = "experiments/sdpa-l2/bfp4-lofi-v2/preprocess/"
        descriptor = ttnn.ProgramDescriptor(cbs=cbs, semaphores=[], kernels=[
            ttnn.KernelDescriptor(kernel_source=base+"reader.cpp", core_ranges=grid,
                compile_time_args=[1]+ttnn.TensorAccessorArgs(src).get_compile_time_args(), runtime_args=read,
                config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=base+"writer.cpp", core_ranges=grid,
                compile_time_args=[1]+ttnn.TensorAccessorArgs(out).get_compile_time_args(), runtime_args=write,
                config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=benchmark.PREFIX+"identity_probe/compute.cpp", core_ranges=grid,
                compile_time_args=[struct.unpack("I", struct.pack("f", 1/math.sqrt(128)))[0]], runtime_args=compute,
                defines=list(defines.items()), config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=getattr(ttnn.MathFidelity, args.fidelity), fp32_dest_acc_en=False,
                    dst_full_sync_en=False, math_approx_mode=True)),
        ])
        invoke = lambda: ttnn.generic_op([src, out], descriptor)
        invoke()
        actual = ttnn.to_torch(out)
        columns = actual[..., 0].reshape(-1)
        mismatches = int((columns.view(torch.uint16).to(torch.int32) != 0x3f80).sum())
        assert mismatches == 0, f"Nonidentity correction for {mismatches} finite BF16 maxima"
        benchmark.replay(device, invoke, out, actual, 0, 0)
        assert benchmark.tensor_hash(host) == original_hash
        assert benchmark.tensor_hash(ttnn.to_torch(src)) == original_hash
        assert sources() == before
        result = dict(arguments=vars(args), finite_bf16_patterns=65280,
            includes_both_signed_zeros=True, includes_all_bf16_subnormals=True,
            first_column_output_bits="0x3f80", mismatches=0, mandatory_trace_replays=2,
            output_sha256=benchmark.tensor_hash(actual), original_input_sha256=original_hash,
            original_host_immutable=True, original_device_immutable=True,
            source_sha256=before, source_stable=True,
            scope="Frozen BF16-DST correction on every finite equal BF16 maximum, scale1/sqrt128; not a speed test")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str)+"\n")
        print(json.dumps(result, default=str), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

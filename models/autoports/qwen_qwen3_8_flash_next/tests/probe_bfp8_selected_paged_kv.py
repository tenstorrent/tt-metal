# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""BFP8 selected-row decompression proof and four-rank changed-index trace probe.

Synthetic values test exact unpack/addressing, not model precision acceptance.
--cpu-only imports neither TTNN nor torch. Parent owns serialized hardware jobs.
"""
import argparse
import json
import math
import os
import statistics
import struct
import time
from pathlib import Path


def cpu_proof():
    count = 0
    for exponent in range(7, 254):
        for byte in range(256):
            sign, mantissa = (byte & 128) << 8, byte & 127
            if mantissa:
                shift = 6 - (mantissa.bit_length() - 1)
                bits = sign | ((exponent - shift) << 7) | (((mantissa << shift) << 1) & 127)
            else:
                bits = sign
            value = math.ldexp(float(byte & 127), exponent - 127 - 6)
            if byte & 128:
                value = -value
            expected = struct.unpack("<I", struct.pack("<f", value))[0] >> 16
            assert bits == expected, (byte, exponent, bits, expected)
            count += 1
    offsets = set()
    for row in range(32):
        for col in range(32):
            offset = (((row >> 4) * 2 + (col >> 4)) * 256) + (row & 15) * 16 + (col & 15)
            exponent = offset // 16
            mantissa = offset + 64
            assert 0 <= exponent < 64
            assert 64 <= (mantissa & ~63) <= 1024
            assert (mantissa & ~63) + 64 <= 1088
            offsets.add(offset)
    assert offsets == set(range(1024))
    assert 36928 + 2048 + 1024 <= 40960
    return dict(passed=True, scalar_cases=count, tile_offsets=len(offsets), scratch_end=40000, scratch_bytes=40960)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--queries", default="1,2,31,32")
    parser.add_argument("--replays", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        passed=False, cpu_proof=cpu_proof(), rows=[], mesh=[1, 4], cache_dtype="bfloat8_b", output_dtype="bfloat16"
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    if args.cpu_only:
        report.update(passed=True, device_evidence=False)
        save()
        print(json.dumps(report))
        return
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("TT_METAL_TRACE_ALLOC_TRACKING=1 is required")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
        parser.error("Program-cache allocations must remain tracked")
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.bfp8_selected_paged_kv import LocalSelectedPagedKVBFP8

    torch.manual_seed(815)
    torch.set_num_threads(8)
    mesh, traces, cases = None, [], []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16 * 1024 * 1024)
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        composer = ttnn.ConcatMeshToTensor(mesh, dim=0)

        def tensor(host, dtype, layout=ttnn.TILE_LAYOUT, device=True):
            return ttnn.from_torch(
                host.contiguous(), device=mesh if device else None, dtype=dtype, layout=layout, mesh_mapper=mapper
            )

        def read(value):
            return ttnn.to_torch(value, mesh_composer=composer)

        # Native BFP8 packing establishes the reference; the reader must match
        # its unpacked BF16 exactly, independent of quantization relative to BF16.
        k = tensor(torch.randn(16, 1, 32, 256), ttnn.bfloat8_b)
        v = tensor(torch.randn(16, 1, 32, 256), ttnn.bfloat8_b)
        kh, vh = read(k).reshape(4, 4, 32, 256), read(v).reshape(4, 4, 32, 256)
        pages_host = torch.tensor([[3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1], [0, 2, 1, 3]], dtype=torch.int32)
        pages = tensor(pages_host, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        for q in map(int, args.queries.split(",")):
            if not 1 <= q <= 32:
                raise ValueError("Queries must be in 1..32")
            variants, expected = [], []
            for changed in range(2):
                ids = torch.randint(0, 128, (4, q, 2080), dtype=torch.int64)
                ids[:, :, ::7] = 0xFFFFFFFF
                ids[:, :, :8] = torch.tensor([0, 15, 16, 31, 32, 63, 64, 127])
                variants.append(tensor(ids.reshape(4 * q, 2080), ttnn.uint32, device=False))
                outputs = []
                for cache in (kh, vh):
                    rows = []
                    for r in range(4):
                        safe = ids[r].clamp(max=127)
                        selected = cache[r][pages_host[r][safe // 32].long(), safe % 32]
                        selected[ids[r] == 0xFFFFFFFF] = 0
                        rows.append(selected.reshape(q, 1, 2080, 256))
                    outputs.append(torch.cat(rows).to(torch.bfloat16))
                expected.append(outputs)
            ids = ttnn.to_device(variants[0], mesh)
            outshape = (4 * q, 1, 2080, 256)
            ko, vo = tensor(torch.zeros(outshape), ttnn.bfloat16), tensor(torch.zeros(outshape), ttnn.bfloat16)
            op = LocalSelectedPagedKVBFP8(k_cache=k, v_cache=v, page_table=pages, selected_ids=ids, keys=ko, values=vo)
            op.prepare()
            op.forward()
            cases.append((q, op, ids, variants, expected, ko, vo))
        ttnn.synchronize_device(mesh)
        for q, op, ids, variants, expected, ko, vo in cases:
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            op.forward()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces.append(trace)
            samples, checks = [], []
            for replay in range(args.replays):
                changed = replay % 2
                ttnn.copy_host_to_device_tensor(variants[changed], ids)
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                samples.append((time.perf_counter() - start) * 1000)
                checks.append(
                    all(torch.equal(read(out).to(torch.bfloat16), ref) for out, ref in zip((ko, vo), expected[changed]))
                )
            row = dict(queries=q, changed_input_checks=checks, trace_ms=statistics.median(samples))
            report["rows"].append(row)
            save()
            assert all(checks), row
            print(json.dumps(row), flush=True)
        report.update(passed=True, device_evidence=True, trace_tracking=True, program_cache_tracked=True)
    finally:
        if mesh is not None:
            for trace in traces:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)
        save()


if __name__ == "__main__":
    main()

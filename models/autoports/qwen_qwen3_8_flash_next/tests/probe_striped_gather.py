# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct NoC striped gather probe; --cpu-only never imports TTNN.

Default device tests exercise exact UINT32/BF16/FLOAT32 payloads on small tables.
--cache-dir uploads complete packed PLE word/scale memmaps and probes actual
large column addresses; that mode supports UINT32 and BF16, not FLOAT32.
All device launch commands must be serialized and bounded by the parent.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from models.autoports.qwen_qwen3_8_flash_next.tt.striped_gather import _ADDRESS_SOURCE


def cpu_proof():
    rng = np.random.default_rng(395)
    columns = np.unique(
        np.concatenate(
            (
                np.array([0, 1, 15, 16, 31, 32, 63, 64, 16777215, 16777216, 16777217, 200000958, 200000959]),
                rng.integers(0, 200000960, 512),
            )
        )
    )
    pairs = [(stripe, int(column)) for stripe in range(32) for column in columns]
    source = (
        "#include <cstdint>\n#include <iostream>\n"
        + _ADDRESS_SOURCE
        + """
int main() {
    uint32_t stripe, column;
    while (std::cin >> stripe >> column) std::cout << striped_tile_element(stripe,column) << "\\n";
}
"""
    )
    with tempfile.TemporaryDirectory(prefix="striped-gather-proof-") as directory:
        directory = Path(directory)
        cpp, binary = directory / "proof.cpp", directory / "proof"
        cpp.write_text(source)
        subprocess.run(
            ["g++", "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True, capture_output=True, text=True
        )
        answer = subprocess.run(
            [str(binary)], input="".join(f"{r} {c}\n" for r, c in pairs), check=True, capture_output=True, text=True
        )
    offsets = [int(value) for value in answer.stdout.split()]
    assert len(offsets) == len(pairs)
    # Independent NumPy face tilization of a tile tagged with its logical index.
    matrix = np.arange(1024, dtype=np.uint32).reshape(32, 32)
    tiled = matrix.reshape(2, 16, 2, 16).transpose(0, 2, 1, 3).reshape(-1)
    for (stripe, column), offset in zip(pairs, offsets):
        assert 0 <= offset < 1024
        assert tiled[offset] == stripe * 32 + column % 32
        for element_bytes in (2, 4):
            byte_offset = offset * element_bytes
            line = byte_offset // 64 * 64
            assert line % 64 == 0 and line + 64 <= 1024 * element_bytes
            assert (byte_offset - line) % element_bytes == 0
        assert column // 32 < 200000960 // 32
    return {
        "passed": True,
        "compiled_shared_cpp_helper": True,
        "comparisons": len(pairs),
        "maximum_column": int(columns.max()),
        "reference": "independent NumPy 16x16-face tilization",
        "aligned_source_read_bytes": 64,
        "tested_element_bytes": [2, 4],
        "limitation": "proves tile offset/index width and read bounds, not device/DRAM bank addressing",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--columns", type=int, default=8192)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dtypes", default="uint32,bfloat16,float32")
    parser.add_argument("--shapes", default="16x20,17x20,496x20,512x20,32x1")
    args = parser.parse_args()
    report = {
        "cpu_address_proof": cpu_proof(),
        "device_cases": [],
        "full_cache": str(args.cache_dir) if args.cache_dir else None,
    }
    if args.cpu_only:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.striped_gather import StripedGather, striped_gather

    torch.set_num_threads(8)
    torch.manual_seed(851)
    shapes = [tuple(int(value) for value in shape.split("x")) for shape in args.shapes.split(",")]
    kinds = args.dtypes.split(",")
    if args.cache_dir:
        kinds = [kind for kind in kinds if kind != "float32"]
    dtype_map = {"uint32": ttnn.uint32, "bfloat16": ttnn.bfloat16, "float32": ttnn.float32}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        for kind in kinds:
            if args.cache_dir:
                manifest = json.loads((args.cache_dir / "manifest.json").read_text())
                name, npdtype = ("words", np.uint32) if kind == "uint32" else ("scales", np.uint16)
                columns_count = manifest[f"{name}_columns"]
                mapping = np.memmap(args.cache_dir / f"{name}.bin", dtype=npdtype, mode="c", shape=(32, columns_count))
                table_host = torch.from_numpy(mapping)
                if kind == "bfloat16":
                    table_host = table_host.view(torch.bfloat16)
            else:
                columns_count = args.columns
                if columns_count < 64 or columns_count % 32:
                    raise ValueError("--columns must be >=64 and divisible by32")
                table_host = (
                    torch.randint(0, 2**32, (32, columns_count), dtype=torch.int64).to(torch.uint32)
                    if kind == "uint32"
                    else torch.randn(32, columns_count).to(torch.bfloat16 if kind == "bfloat16" else torch.float32)
                )
            table = ttnn.from_torch(table_host, dtype=dtype_map[kind], layout=ttnn.TILE_LAYOUT, device=mesh)
            for shape in shapes:
                columns = torch.randint(0, columns_count, shape, dtype=torch.int64)
                stripes = torch.randint(0, 32, shape, dtype=torch.int64)
                boundary = [0, 1, 15, 16, 31, 32, columns_count - 2, columns_count - 1]
                boundary_count = min(columns.numel(), len(boundary))
                columns.flatten()[:boundary_count] = torch.tensor(boundary[:boundary_count])
                stripes.flatten()[:32] = torch.arange(min(32, stripes.numel()))

                def golden(c, s):
                    # UINT32 advanced indexing is unsupported in some torch
                    # versions; NumPy selects only requested rows/cells first.
                    if kind == "uint32":
                        array = table_host.numpy()[s.numpy(), c.numpy()]
                        return torch.from_numpy(array.copy()).to(torch.int64)
                    return table_host[s, c].clone()

                expected = golden(columns, stripes)
                c = ttnn.from_torch(columns, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh)
                s = ttnn.from_torch(stripes, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh)
                eager = striped_gather(table, c, s)
                actual = ttnn.to_torch(eager)
                if kind == "uint32":
                    actual = actual.to(torch.int64)
                entry = {
                    "dtype": kind,
                    "table_columns": columns_count,
                    "shape": list(shape),
                    "eager_equal": torch.equal(actual, expected),
                }
                ttnn.deallocate(eager)
                # Real arithmetic produces new address buffers on each call;
                # test that cached generic programs pick up their new addresses.
                rebound_columns = torch.flip(columns, dims=(0,))
                rebound_stripes = (stripes + 9) % 32
                rebound_c = ttnn.from_torch(rebound_columns, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh)
                rebound_s = ttnn.from_torch(rebound_stripes, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh)
                rebound = striped_gather(table, rebound_c, rebound_s)
                rebound_actual = ttnn.to_torch(rebound)
                if kind == "uint32":
                    rebound_actual = rebound_actual.to(torch.int64)
                entry["rebound_inputs_equal"] = torch.equal(rebound_actual, golden(rebound_columns, rebound_stripes))
                ttnn.deallocate(rebound)
                ttnn.deallocate(rebound_c)
                ttnn.deallocate(rebound_s)
                # Persistent output pre-exists capture and needs no exemption.
                output = ttnn.empty(
                    shape,
                    dtype=dtype_map[kind],
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                operation = StripedGather(table=table, columns=c, stripes=s, output=output)
                operation.prepare()
                operation.forward()
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                operation.forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = ttnn.to_torch(output)
                if kind == "uint32":
                    actual = actual.to(torch.int64)
                entry["trace_equal"] = torch.equal(actual, expected)
                changed_columns = torch.flip(columns, dims=(0,))
                changed_stripes = (stripes + 17) % 32
                changed_expected = golden(changed_columns, changed_stripes)
                for value, target in ((changed_columns, c), (changed_stripes, s)):
                    host = ttnn.from_torch(value, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
                    ttnn.copy_host_to_device_tensor(host, target)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                changed = ttnn.to_torch(output)
                if kind == "uint32":
                    changed = changed.to(torch.int64)
                entry["changed_indices_equal"] = torch.equal(changed, changed_expected)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                again = ttnn.to_torch(output)
                if kind == "uint32":
                    again = again.to(torch.int64)
                entry["repeat_equal"] = torch.equal(changed, again)
                entry["selected_source_bytes_per_pass"] = int(np.prod(shape)) * 64
                entry["source_table_bytes"] = 32 * columns_count * (2 if kind == "bfloat16" else 4)
                ttnn.release_trace(mesh, trace)
                trace = None
                report["device_cases"].append(entry)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(entry), flush=True)
                assert all(
                    entry[name]
                    for name in (
                        "eager_equal",
                        "rebound_inputs_equal",
                        "trace_equal",
                        "changed_indices_equal",
                        "repeat_equal",
                    )
                )
                ttnn.deallocate(c)
                ttnn.deallocate(s)
                ttnn.deallocate(output)
                del operation
            ttnn.deallocate(table)
            del table_host
    except Exception as exc:
        report["error"] = str(exc).split("backtrace:")[0]
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

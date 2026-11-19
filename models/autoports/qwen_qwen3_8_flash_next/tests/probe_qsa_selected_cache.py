# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct selected paged K/V reader: CPU address proof and optional device probe.

--cpu-only imports no TTNN. The device probe uses synthetic BF16 cache payloads,
permuted page ownership, sentinel rows, changing selected IDs/page tables and
repeated trace replay. It is an operator test, not full decoder accuracy evidence.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from models.autoports.qwen_qwen3_8_flash_next.tt.qsa_selected_cache import _ADDRESS_SOURCE


def cpu_proof():
    rng = np.random.default_rng(945)
    cases = []
    for page_size in (32, 64, 256):
        for _ in range(1024):
            cases.append(
                (
                    int(rng.integers(0, 8192)),
                    int(rng.integers(0, 2)),
                    page_size,
                    int(rng.integers(0, page_size)),
                    int(rng.integers(0, 8)),
                    int(rng.integers(0, 32)),
                )
            )
        for token in (0, 1, 15, 16, 31, page_size - 1):
            for head in range(2):
                for col in (0, 15, 16, 31):
                    cases.append((8191, head, page_size, token, 7, col))
    source = (
        "#include <stdint.h>\n#include <iostream>\n"
        + _ADDRESS_SOURCE
        + r"""
int main() {
    uint32_t p,h,s,t,c,w;
    while (std::cin >> p >> h >> s >> t >> c >> w)
        std::cout << qsa_cache_tile(p,h,s,t,c) << " " << qsa_face_element(t%32,w) << "\n";
}
"""
    )
    with tempfile.TemporaryDirectory(prefix="qsa-cache-proof-") as temporary:
        root = Path(temporary)
        cpp, binary = root / "proof.cpp", root / "proof"
        cpp.write_text(source)
        subprocess.run(
            ["g++", "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True, capture_output=True, text=True
        )
        result = subprocess.run(
            [str(binary)],
            input="".join(" ".join(map(str, case)) + "\n" for case in cases),
            check=True,
            capture_output=True,
            text=True,
        )
    lines = result.stdout.splitlines()
    assert len(lines) == len(cases)
    # Independent tiled-face reference. No shared C++ expression is duplicated.
    matrix = np.arange(1024).reshape(32, 32)
    tiled = matrix.reshape(2, 16, 2, 16).transpose(0, 2, 1, 3).reshape(-1)
    for (page, head, page_size, token, channel, column), line in zip(cases, lines):
        tile, offset = map(int, line.split())
        outer, row_tile, column_tile = np.unravel_index(tile, (8192 * 2, page_size // 32, 8))
        assert (outer // 2, outer % 2, row_tile, column_tile) == (page, head, token // 32, channel)
        assert tiled[offset] == (token % 32) * 32 + column
        byte = offset * 2
        assert (byte // 64) * 64 + 64 <= 2048
    # Output group order must match [Q,2,2080,256] tile order.
    for query in range(32):
        for head in range(2):
            for slot_tile in (0, 1, 63, 64):
                group = (query * 2 + head) * 65 + slot_tile
                assert np.unravel_index(group * 8, (32, 2, 65, 8)) == (query, head, slot_tile, 0)
    return {
        "passed": True,
        "compiled_shared_cpp_helper": True,
        "comparisons": len(cases),
        "page_sizes": [32, 64, 256],
        "max_physical_page": 8191,
        "reference": "NumPy unravel_index and independent 16x16-face tilization",
        "limitation": "address arithmetic proof only; no device/bank-routing claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--queries", default="1,3,32")
    parser.add_argument("--page-sizes", default="32,256")
    parser.add_argument("--pages", type=int, default=16, help="Use4 to test a short page-table stick on Blackhole")
    args = parser.parse_args()
    report = {
        "scope": "selected-KV operator diagnostic; not decoder acceptance",
        "cpu_address_proof": cpu_proof(),
        "device_cases": [],
    }
    if args.cpu_only:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa_selected_cache import SelectedPagedKV, selected_paged_kv

    torch.set_num_threads(8)
    torch.manual_seed(496)
    queries_values = [int(value) for value in args.queries.split(",")]
    page_sizes = [int(value) for value in args.page_sizes.split(",")]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:

        def device(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout, device=mesh)

        def host(value, dtype, layout):
            return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)

        for page_size in page_sizes:
            physical_pages = args.pages
            if physical_pages < 2:
                raise ValueError("Device probe requires at least two physical pages")
            k = torch.randn(physical_pages, 2, page_size, 256).bfloat16()
            v = torch.randn_like(k)
            kd, vd = device(k), device(v)
            for queries in queries_values:
                page_table = torch.randperm(physical_pages).reshape(1, -1).int()
                selected = torch.randint(0, physical_pages * page_size, (queries, 2080), dtype=torch.int64)
                # Mix page/tile/face boundaries, repeated IDs and invalid rows.
                selected[:, 0:8] = torch.tensor([0, 1, 15, 16, 31, 32, page_size - 1, page_size])
                selected[:, 8] = physical_pages * page_size - 1
                selected[:, 9] = selected[:, 0]
                selected[:, 2048:] = 0xFFFFFFFF
                selected[:, 73::97] = 0xFFFFFFFF
                if queries > 1:
                    selected[1, :] = 0xFFFFFFFF

                def golden(cache, ids, pages):
                    result = torch.zeros(queries, 2, 2080, 256, dtype=torch.bfloat16)
                    for query in range(queries):
                        valid = ids[query] != 0xFFFFFFFF
                        tokens = ids[query, valid]
                        physical = pages[0, tokens // page_size].long()
                        gathered = cache[physical, :, tokens % page_size, :].permute(1, 0, 2)
                        result[query, :, valid, :] = gathered
                    return result

                expected_k, expected_v = golden(k, selected, page_table), golden(v, selected, page_table)
                pd = device(page_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                ids = device(selected, ttnn.uint32)
                eager_k, eager_v = selected_paged_kv(kd, vd, pd, ids)
                entry = {
                    "queries": queries,
                    "page_size": page_size,
                    "eager_keys_equal": torch.equal(ttnn.to_torch(eager_k), expected_k),
                    "eager_values_equal": torch.equal(ttnn.to_torch(eager_v), expected_v),
                }
                ttnn.deallocate(eager_k)
                ttnn.deallocate(eager_v)
                out_shape = (queries, 2, 2080, 256)
                outputs = [
                    ttnn.empty(
                        out_shape,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(2)
                ]
                operation = SelectedPagedKV(
                    k_cache=kd, v_cache=vd, page_table=pd, selected_ids=ids, keys=outputs[0], values=outputs[1]
                )
                operation.prepare()
                operation.forward()
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                operation.forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                # Every buffer predates capture; no acknowledgments are needed.
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                entry["trace_keys_equal"] = torch.equal(ttnn.to_torch(outputs[0]), expected_k)
                entry["trace_values_equal"] = torch.equal(ttnn.to_torch(outputs[1]), expected_v)
                changed_ids = selected.flip(-1).contiguous()
                changed_pages = page_table.flip(-1).contiguous()
                ttnn.copy_host_to_device_tensor(host(changed_ids, ttnn.uint32, ttnn.TILE_LAYOUT), ids)
                ttnn.copy_host_to_device_tensor(host(changed_pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), pd)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                changed_k, changed_v = ttnn.to_torch(outputs[0]), ttnn.to_torch(outputs[1])
                entry["changed_keys_equal"] = torch.equal(changed_k, golden(k, changed_ids, changed_pages))
                entry["changed_values_equal"] = torch.equal(changed_v, golden(v, changed_ids, changed_pages))
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                entry["repeat_equal"] = torch.equal(changed_k, ttnn.to_torch(outputs[0])) and torch.equal(
                    changed_v, ttnn.to_torch(outputs[1])
                )
                ttnn.release_trace(mesh, trace)
                trace = None
                entry["passed"] = all(value for name, value in entry.items() if name.endswith("_equal"))
                report["device_cases"].append(entry)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(entry), flush=True)
                assert entry["passed"]
                for tensor in [pd, ids, *outputs]:
                    ttnn.deallocate(tensor)
                del operation
            ttnn.deallocate(kd)
            ttnn.deallocate(vd)
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

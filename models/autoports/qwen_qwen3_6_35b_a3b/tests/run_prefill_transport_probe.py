# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact BF16 prefill row/page transport, runtime offsets and trace reuse.

Run serialized, under an external timeout, with TT trace allocation tracking.
This is a small transport probe, not a model correctness/performance claim.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.prefill_transport import PrefillTransport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "doc/full_model/prefill_transport_probe",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    result = {
        "status": "RUNNING",
        "scope": "prefill transport only",
        "tracker": True,
        "skip_program_cache": False,
        "captures": 0,
        "releases": 0,
        "steps": [],
        "sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), root / "tt/prefill_transport.py", root / "tt/slot_state.py")
        },
    }
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=10_000_000)
    trace = None

    def upload(value, dtype):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def refresh(value, destination):
        host = ttnn.from_torch(
            value.contiguous(),
            dtype=destination.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.copy_host_to_device_tensor(host, destination)

    def equal(tensor, expected, label):
        for chip, local in enumerate(ttnn.get_device_tensors(tensor)):
            actual = ttnn.to_torch(local).contiguous()
            assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
            if expected.dtype == torch.bfloat16:
                assert torch.equal(actual.view(torch.int16), expected.contiguous().view(torch.int16)), (label, chip)
            else:
                assert torch.equal(actual.to(torch.int64), expected.to(torch.int64)), (label, chip)

    try:
        row_transport, page_transport = PrefillTransport(mesh), PrefillTransport(mesh)
        # Cover nonzero mantissas, signs, subnormals and many physical DRAM pages.
        codes = torch.arange(173 * 2048, dtype=torch.int32)
        codes = ((codes % 0x7F00) + 1) | ((codes % 2) * 0x8000)
        source_host = codes.to(torch.int16).view(torch.bfloat16).reshape(1, 173, 2048)
        initial = torch.ones((1, 113, 2048), dtype=torch.bfloat16)
        pages_host = (((torch.arange(64, dtype=torch.int32) * 7) % 64) + 1000).reshape(1, 64)
        source, destination = upload(source_host, ttnn.bfloat16), upload(initial, ttnn.bfloat16)
        pages = upload(pages_host, ttnn.int32)
        fill_pages = upload(torch.zeros((1, 8), dtype=torch.int32), ttnn.int32)
        row_transport.rows(source, destination, count=1)
        page_transport.page_window(pages, fill_pages, 0)
        ttnn.synchronize_device(mesh)
        prepared_entries = mesh.num_program_cache_entries()
        assert prepared_entries > 0, "Program cache must be enabled for this probe"
        result["prepared_program_entries"] = prepared_entries

        def expected_window(start):
            indices = torch.arange(start, start + 8).clamp_max(63)
            return pages_host[:, indices]

        def check_step(label, source_row, destination_row, count, extent, first_page, *, traced):
            refresh(initial, destination)
            if traced:
                row_transport._refresh(source_row, destination_row, count, extent)
                page_transport._refresh(first_page)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            else:
                row_transport.rows(
                    source,
                    destination,
                    source_row=source_row,
                    destination_row=destination_row,
                    count=count,
                    extent=extent,
                )
                page_transport.page_window(pages, fill_pages, first_page)
            expected = initial.clone()
            expected[:, destination_row : destination_row + count] = source_host[:, source_row : source_row + count]
            expected[:, destination_row + count : destination_row + extent] = 0
            equal(destination, expected, label + "/rows_exact_bits")
            equal(fill_pages, expected_window(first_page), label + "/page_ids")
            entries = mesh.num_program_cache_entries()
            assert entries == prepared_entries, ("late program", label, prepared_entries, entries)
            result["steps"].append(
                {
                    "label": label,
                    "traced": traced,
                    "source_row": source_row,
                    "destination_row": destination_row,
                    "valid_rows": count,
                    "extent": extent,
                    "first_page": first_page,
                    "program_entries": entries,
                }
            )
            print(f"PREFILL_TRANSPORT_PASS {label} programs={entries}", flush=True)

        cases = [
            (17, 3, 31, 32, 1),
            (63, 0, 65, 80, 7),
            (172, 112, 1, 1, 63),
            (0, 9, 0, 32, 57),
        ]
        for i, case in enumerate(cases):
            check_step(f"eager_{i}", *case, traced=False)
        # Restore known metadata before capture. Both transport metadata buffers
        # are setup allocations; capture has no host copies or new tensors.
        row_transport._refresh(0, 0, 1, 1)
        page_transport._refresh(0)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        row_transport.rows_device(source, destination)
        page_transport.page_window_device(pages, fill_pages)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        result["captures"] = 1
        result["trace_id"] = int(trace)
        assert mesh.num_program_cache_entries() == prepared_entries
        for i, case in enumerate(reversed(cases)):
            check_step(f"trace_{i}", *case, traced=True)
        # Exercise an eager request between retained replays with unseen offsets.
        check_step("eager_after_capture", 101, 41, 27, 33, 31, traced=False)
        check_step("final_trace_after_eager", 7, 6, 91, 96, 62, traced=True)
        result["status"] = "PASS"
    except BaseException as exc:
        result["status"] = "FAIL"
        result["error"] = repr(exc)
        raise
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
            result["releases"] += 1
        ttnn.close_mesh_device(mesh)
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"PREFILL_TRANSPORT_PROBE_PASS {args.output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()

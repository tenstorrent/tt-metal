# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded native page-table identity-gather diagnosis; no cache reads."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(123)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results = []
    try:
        for pages in (128, 1920, 1921, 2048, 8192):
            table = torch.randperm(pages).int()[None]
            pt = ttnn.from_torch(table, device=mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            indices = ttnn.from_torch(
                torch.arange(pages, dtype=torch.int32)[None],
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            with device_only():
                gathered = ttnn.gather(pt, dim=1, index=indices)
                direct = pt
            actual = ttnn.to_torch(gathered)
            mismatch = (actual != table).nonzero()
            first = int(mismatch[0, 1]) if mismatch.numel() else None
            direct_exact = torch.equal(ttnn.to_torch(direct), table)
            result = dict(
                pages=pages,
                native_exact=torch.equal(actual, table),
                direct_exact=direct_exact,
                first_mismatch=first,
                mismatch_count=int((actual != table).sum()),
                first_expected=int(table[0, first]) if first is not None else None,
                first_actual=int(actual[0, first]) if first is not None else None,
            )
            results.append(result)
            args.output.write_text(json.dumps(dict(results=results, runtime_audit="clean"), indent=2) + "\n")
            print(result, flush=True)
            assert direct_exact
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

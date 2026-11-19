# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe lossless device lookup of packed integer payloads in tiled storage.

Setup supplies word addresses to isolate lookup from n-gram hash correctness.
The layout stripes linear words over 32 rows; a tile-aligned table then needs
no per-embedding row padding and no whole-table transpose at runtime.
"""

import argparse
import json
from pathlib import Path

import torch

import ttnn


def lookup(table, columns, stripes):
    selected_columns = ttnn.gather(table, 1, columns)
    selected_columns = ttnn.transpose(selected_columns, 0, 1)
    return ttnn.gather(selected_columns, 1, stripes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(17)
    words = torch.randint(0, 2**32, (4096 * 20,), dtype=torch.int64)
    logical_addresses = torch.tensor([0, 1, 31, 32, 159, 160, 161, 81919] * 4)
    table = words.reshape(-1, 32).T.contiguous()
    columns = (logical_addresses // 32).expand(32, -1).contiguous()
    stripes = (logical_addresses % 32).reshape(-1, 1)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = {}
    try:

        def to_device(x):
            return ttnn.from_torch(x, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh)

        w, c, s = [to_device(x) for x in (table, columns, stripes)]
        result = lookup(w, c, s)
        actual = ttnn.to_torch(result).flatten().to(torch.int64)
        report["word_lookup_exact"] = torch.equal(actual, words[logical_addresses])
        unpacked = []
        for shift in range(0, 32, 4):
            nibbles = ttnn.bitwise_and(ttnn.logical_right_shift(result, shift), 15)
            unpacked.append(ttnn.to_torch(nibbles).flatten().to(torch.int64))
        expected = torch.stack([(words[logical_addresses] >> shift) & 15 for shift in range(0, 32, 4)])
        report["nibbles_exact"] = torch.equal(torch.stack(unpacked), expected)
    except RuntimeError as exc:
        report["error"] = str(exc).split("backtrace:")[0]
    finally:
        ttnn.close_mesh_device(mesh)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report.get("nibbles_exact"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

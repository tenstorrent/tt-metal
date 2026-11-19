# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only byte-exact ownership/padding checks of the actual setup repacker."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import _repack_ple_table


def check(root, meta, filename, columns, words_per_row, ids):
    result = _repack_ple_table(root, meta, filename, columns, words_per_row)
    source = np.memmap(root / filename, mode="r", dtype=np.uint32, shape=(32, columns))
    local_columns = result.shape[1] // 4
    assert local_columns % 32 == 0
    lanes = np.arange(words_per_row)[None, :]
    global_words = ids[:, None] * words_per_row + lanes
    local_words = (ids[:, None] // 4) * words_per_row + lanes
    expected = source[global_words % 32, global_words // 32]
    actual = result[local_words % 32, (ids[:, None] % 4) * local_columns + local_words // 32]
    assert np.array_equal(actual, expected)
    used_columns = meta["rows"] // 4 * words_per_row // 32
    for rank in range(4):
        assert not np.any(result[:, rank * local_columns + used_columns : (rank + 1) * local_columns])
    return dict(
        file=filename,
        rows_checked=len(ids),
        words_checked=actual.size,
        per_rank_columns=local_columns,
        padding_columns=local_columns - used_columns,
        exact=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"device_opened": False, "synthetic": [], "checkpoint": []}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for rows in (768, 1024, 2048):
            meta = {"rows": rows}
            (root / "manifest.json").write_text(json.dumps(meta))
            for words in (20, 4):
                filename = f"{rows}_{words}.bin"
                values = np.arange(rows * words, dtype=np.uint32).reshape(-1, 32).T.copy()
                values.tofile(root / filename)
                report["synthetic"].append(check(root, meta, filename, values.shape[1], words, np.arange(rows)))
    if args.cache_dir:
        root = args.cache_dir
        meta = json.loads((root / "manifest.json").read_text())
        ids = np.unique(
            np.concatenate(
                (
                    np.arange(128),
                    np.arange(meta["rows"] - 128, meta["rows"]),
                    np.random.default_rng(9201).integers(0, meta["rows"], size=8192),
                )
            )
        )
        for filename, key, words in (("words.bin", "words_columns", 20), ("scales.bin", "scales_columns", 4)):
            report["checkpoint"].append(check(root, meta, filename, meta[key], words, ids))
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

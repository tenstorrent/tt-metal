# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Expand the unchanged full-context harness to inspect every trailing row.

The wrapper leaves HF oracle generation and cache identity unchanged. Extra PCC
checks run only at the existing host readback boundary, never inside a TT pass.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

from models.autoports.qwen_qwen3_6_35b_a3b.tests import long_context


def main():
    doc = (
        Path(__file__).parents[1]
        / "doc"
        / ("fused_decoder" if os.environ.get("QWEN_DECODER") == "fused" else "functional_decoder")
    )
    output = doc / "full_long.json"
    original_pcc = long_context.pcc
    checks = {}

    def pcc_with_tail_check(expected, actual):
        score = original_pcc(expected, actual)
        if tuple(expected.shape) in ((1, 262143, 2048), (1, 262144, 2048)):
            length = expected.shape[1]
            rows = {str(i): original_pcc(expected[:, i], actual[:, i]) for i in range(length - 64, length)}
            assert min(rows.values()) >= 0.995, (length, rows)
            checks[str(length)] = rows
            print("ALL_TRAILING_ROWS_PASS", length, min(rows.values()), flush=True)
        return score

    long_context.pcc = pcc_with_tail_check
    long_context.run(
        SimpleNamespace(
            layer=3,
            length=262144,
            oracle_only=False,
            oracle_cache=doc / "oracle_cache/full_262144.pt",
            output=output,
        )
    )
    assert set(checks) == {"262143", "262144"}
    result = json.loads(output.read_text())
    result["all_trailing_64_rows_pcc"] = checks
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare retained FAST outputs against the frozen pre-optimization control."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def row(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1, path
    return records[0]


comparisons = []
for old_path in sorted(HERE.glob("old-[0-9]*.jsonl")):
    new_path = old_path.with_name(old_path.name.replace("old-", "new-", 1))
    old, new = row(old_path), row(new_path)
    same = old["sampled_output_sha256"] == new["sampled_output_sha256"]
    assert same, (old_path, new_path)
    comparisons.append(
        dict(
            length=old["kv_len"],
            seed=old["seed"],
            distribution=old["distribution"],
            bitwise_equal=same,
            l2_pct=new["l2_pct"],
            pcc=new["pcc"],
        )
    )
assert len(comparisons) == 32, len(comparisons)
print(json.dumps(dict(qualification_cases=len(comparisons), comparisons=comparisons), indent=2))

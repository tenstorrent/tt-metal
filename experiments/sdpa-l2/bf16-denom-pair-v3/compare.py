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
# The previous task's final v2 outputs are the frozen baseline, not its earlier
# pre-macro "old" outputs. Both versions have numerator and denominator compensation.
for old_path in sorted((HERE.parent / "bf16-sfpu-v2").glob("new-[0-9]*.jsonl")):
    new_path = HERE / old_path.name
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
causal = []
for old_path in sorted(HERE.glob("causal-old-*.jsonl")):
    new_path = HERE / old_path.name.replace("causal-old-", "causal-new-")
    old, new = row(old_path), row(new_path)
    assert old["full_output_sha256"] == new["full_output_sha256"], (old_path, new_path)
    causal.append(
        dict(
            length=old["kv_len"],
            distribution=old["distribution"],
            full_output_equal=True,
            l2_pct=new["l2_pct"],
            pcc=new["pcc"],
        )
    )
assert len(causal) == 4, len(causal)
print(
    json.dumps(
        dict(qualification_cases=len(comparisons) + len(causal), comparisons=comparisons, causal=causal), indent=2
    )
)

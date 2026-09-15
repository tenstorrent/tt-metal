# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-only cross-check of retained, QK-HiFi4, and restoration diagnostics."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return [json.loads(line) for line in (HERE / name).read_text().splitlines()]


def main():
    qualification = {r["id"]: r for r in read("../qualification-v1/accepted-results.jsonl") if r["mode"] == "accurate"}
    baseline = {r["id"]: r for name in ("32768.jsonl", "262144.jsonl") for r in read(name)}
    original = {
        r["id"]: r for name in ("qk-hifi4-original-32768.jsonl", "qk-hifi4-original-262144.jsonl") for r in read(name)
    }
    controls = read("qk-hifi4-retained-q.jsonl")
    assert len(baseline) == len(original) == len(controls) == 6
    assert baseline.keys() == original.keys() == {r["id"] for r in controls}
    for row in controls:
        assert row["full_output_sha256"] == qualification[row["id"]]["full_output_sha256"]
        assert row["experimental_qk_hifi4"] and not row["original_q"]
    for row in list(baseline.values()) + list(original.values()) + controls:
        ref = qualification[row["id"]]
        assert row["input_sha256"] == ref["input_sha256"]
        assert row["reference_positions"] == ref["positions"]
        if not row.get("original_q", False):
            expected = ref["per_head"][row["head"]]
            assert abs(row["metrics"]["device"]["l2_pct"] - expected["l2_pct"]) < 1e-9
            assert abs(row["metrics"]["device"]["row_max_pct"] - expected["row_max_pct"]) < 1e-9
    restored = read("restored-normal.jsonl") + read("restored-outliers.jsonl")
    assert len(restored) == 2
    for row in restored:
        assert row["full_output_sha256"] == qualification[row["id"]]["full_output_sha256"]
        assert not row["experimental_qk_hifi4"]
    print("PASS: 18 matched device diagnostics, six bit-identical retained-Q controls, two restoration probes")
    print(
        "N | distribution | head | retained L2 | Q-only model L2 | QK4 original-Q L2 | retained max row | QK4 max row"
    )
    for key, row in baseline.items():
        new = original[key]
        m, n = row["metrics"], new["metrics"]["device"]
        print(
            f"{row['kv_len']} | {row['distribution']} | {row['head']} | "
            f"{m['device']['l2_pct']:.6f} | {m['q_only']['l2_pct']:.6f} | "
            f"{n['l2_pct']:.6f} | {m['device']['row_max_pct']:.6f} | {n['row_max_pct']:.6f}"
        )
    if (HERE / "both-hifi4.jsonl").exists():
        both = {r["id"]: r for r in read("both-hifi4.jsonl")}
        assert both.keys() == baseline.keys()
        print("N | distribution | QK4/PV2 L2 | QK4/PV4 L2 | PCC | p99 row | max row")
        for key, row in both.items():
            ref = qualification[key]
            assert row["experimental_both_hifi4"] and row["original_q"] and not row["experimental_qk_hifi4"]
            assert row["input_sha256"] == ref["input_sha256"]
            assert row["reference_positions"] == ref["positions"]
            assert row["head"] == baseline[key]["head"]
            m = row["metrics"]["device"]
            print(
                f"{row['kv_len']} | {row['distribution']} | {original[key]['metrics']['device']['l2_pct']:.6f} | "
                f"{m['l2_pct']:.6f} | {m['pcc']:.9f} | {m['row_p99_pct']:.6f} | {m['row_max_pct']:.6f}"
            )
        restored_both = read("both-restored.jsonl")
        assert len(restored_both) == 2
        for row in restored_both:
            assert not row["experimental_both_hifi4"] and not row["experimental_qk_hifi4"]
            assert row["full_output_sha256"] == qualification[row["id"]]["full_output_sha256"]
        print("PASS: six both-HiFi4 diagnostics and two post-HiFi4 full-output restoration checks")


if __name__ == "__main__":
    main()

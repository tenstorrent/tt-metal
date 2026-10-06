# SPDX-License-Identifier: Apache-2.0
"""Resolve the serving-regime comparison without changing the declared TF ranking."""

import json
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"
    selected = json.loads((root / "selected_precision_config.json").read_text())
    candidates = [json.loads(path.read_text()) for path in (root / "candidates").glob("*/result.json")]
    leader = max(
        (row for row in candidates if row["status"] == "pass" and "token_out" in row),
        key=lambda row: row["token_out"]["decode_t_s_u"],
    )
    rows = []
    names = ["selected_default"]
    if leader["config_id"] != selected["config_id"]:
        names.append("token_out_rank_control")
    for name in names:
        result = json.loads((root / name / "result.json").read_text())
        command = json.loads((root / name / "run.command.json").read_text())
        assert command["exit_code"] == 0 and result["status"] == "pass"
        assert len(result["runtime_summary"]["layers"]) == 50
        assert result["runtime_summary"]["capacity"] == 1048576
        assert command["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "0"
        rates = []
        for sample in result["token_out_repetitions"]:
            assert sample["prompt_tokens"] == sample["generated_tokens"] == 128
            assert sample["counters"]["decode_replays"] == sample["counters"]["split_replays"] == 128
            for key in (
                "token_refreshes",
                "position_refreshes",
                "rope_refreshes",
                "page_table_refreshes",
                "token_readbacks",
                "logit_readbacks",
                "synchronizations",
            ):
                assert sample["counters"].get(key, 0) == 0
            rates.append(sample["decode_t_s_u"])
        assert len(rates) == 5
        rows.append(
            dict(
                config_id=result["config_id"],
                evidence=name + "/result.json",
                rates=rates,
                median=statistics.median(rates),
                minimum=min(rates),
                maximum=max(rates),
                construction="fresh public build_generator; all 50 layers, full 1M cache",
            )
        )
    assert rows[0]["config_id"] == selected["config_id"]
    overlap = None
    if len(rows) == 2:
        assert rows[1]["config_id"] == leader["config_id"]
        overlap = max(row["minimum"] for row in rows) <= min(row["maximum"] for row in rows)
    report = dict(
        status="measured",
        selection_rule="User requires the fastest passing config by the declared median traced teacher-forcing matrix; token-out measurements are separately reported.",
        initial_single_loop_token_out_leader=leader["config_id"],
        fresh_results=rows,
        observed_ranges_overlap=overlap,
        interpretation=(
            "The two observed five-loop ranges overlap; this experiment does not resolve a token-out speed winner."
            if overlap
            else (
                "Observed ranges do not overlap; report their regime-specific medians without substituting token-out for the required teacher-forcing selection."
                if overlap is False
                else "The same policy leads both initial metrics; five fresh default-path token-out repetitions recorded."
            )
        ),
        baseline_evidence="candidates/baseline_bfp4_lofi/result.json",
        limitation="Five short repeated loops in one fresh construction per policy; observed ranges are not confidence intervals or long-run stability guarantees.",
    )
    (root / "token_out_ranking.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

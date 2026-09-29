"""CPU-only shared-suite comparison and mechanical-output measurements.

Human review still owns semantic quality; a mechanical pass is not sufficient.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("models/autoports/ifm_k2_horizon_7b/doc")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    checker = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/check_degenerate_output.py"
    spec = importlib.util.spec_from_file_location("datatype_quality_checker", checker)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = module.Report()
    hf_path = ROOT / "full_model/hf_qualitative_extended.json"
    baseline_path = ROOT / "optimized_full_model/qualitative_selected.json"
    hf = {row["id"]: row for row in json.loads(hf_path.read_text())}
    baseline = {row["id"]: row for row in json.loads(baseline_path.read_text())}
    rows = json.loads(args.source.read_text())
    assert {row["id"] for row in rows} == set(hf) == set(baseline)
    comparisons = []
    for row in rows:
        key = row["id"]
        assert row["prompt_token_ids"] == hf[key]["prompt_token_ids"] == baseline[key]["prompt_token_ids"]
        text = row["tt_text_through_eos"]
        module.check_completion(report, artifact=args.source, label=key, text=text)

        def visible(value):
            return value.partition("</ifm|think>")[2] if "</ifm|think>" in value else None

        comparisons.append(
            {
                "id": key,
                "prompt_token_ids_match_controls": True,
                "tt_visible_text": visible(text),
                "hf_visible_text": visible(hf[key]["hf_text"]),
                "baseline_visible_text": visible(baseline[key]["tt_text_through_eos"]),
                "tt_eos_index": row["first_eos_index"],
                "exact_baseline_tokens": row["tt_token_ids"] == baseline[key]["tt_token_ids"],
                "we_can_also_mention_count": text.count("We can also mention"),
            }
        )
    result = {
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "hf_control": str(hf_path),
        "baseline_control": str(baseline_path),
        "checker": str(checker),
        "checker_sha256": hashlib.sha256(checker.read_bytes()).hexdigest(),
        "mechanical_exit_code": report.exit_code,
        "findings": report.findings,
        "measured": report.measured,
        "comparisons": comparisons,
        "semantic_verdict": "requires human review",
    }
    args.output.write_text(json.dumps(result, indent=2, default=lambda value: vars(value)) + "\n")
    print("MECHANICAL", report.exit_code, report.findings)


if __name__ == "__main__":
    main()

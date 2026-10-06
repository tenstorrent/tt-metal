"""Host-only checks that owner-authorized interval bounds cannot look measured."""

import hashlib
import importlib.util
from pathlib import Path

import benchmark_stage.roofline

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/benchmark_run.py"
spec = importlib.util.spec_from_file_location("kolibri_benchmark_run", MODULE_PATH)
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)


def test_interval_bounds_labeled_and_point_estimate_preserved(tmp_path, monkeypatch):
    authorization = tmp_path / "owner-source.md"
    authorization.write_text("Explicit owner authorization for bounded overlap accounting.\n")
    monkeypatch.setattr(reporting, "AUTHORIZATION", authorization)
    output = tmp_path / "run"
    output.mkdir()
    (output / "REPORT.md").write_text(
        "Status: completed.\n"
        "| Single user | 1 | 1 | 4096 / 128 | 1.00 | 2.00 | 500.00 | 500.00 | 12.50 | 22.50 |\n"
        "| 32 users | 32 | 32 | 4096 / 128 | 1.00 | 2.00 | 500.00 | 500.00 | 3.00 | 4.00 |\n"
    )
    rows = {
        "1": {
            "prefill": {"percent": 12.5, "accounting_status": "modeled", "timing_method": "Observed completion"},
            "decode": {"percent": 22.5, "accounting_status": "modeled", "timing_method": "Observed completion"},
        },
        "32": {
            "prefill": {
                "percent": 3,
                "accounting_status": "bounded",
                "percent_bound": "lower_bound",
                "timing_method": "Mixed interval",
            },
            "decode": {
                "percent": 4,
                "accounting_status": "bounded",
                "percent_bound": "lower_bound",
                "timing_method": "Mixed interval",
            },
        },
    }
    monkeypatch.setattr(benchmark_stage.roofline, "load_roofline", lambda *args, **kwargs: rows)
    config = {
        "tasks": ["ifeval"],
        "generation": {"ifeval": {"max_gen_toks": 256}},
        "report_protocol_notes": ["Generation cap reduced from 4096 to 256 to fit client budget."],
    }
    reporting.add_authorized_accounting_report(output, config, {"status": "completed"})
    report = (output / "REPORT.md").read_text()
    assert "12.50 (modeled) | 22.50 (modeled)" in report
    assert "≥3.00 (modeled lower bound) | ≥4.00 (modeled lower bound)" in report
    assert "| 3.00 | 4.00 |" not in report
    assert "Generation cap reduced from 4096 to 256" in report
    assert "Separate intermediate-prefill completion is not observable" in report
    assert hashlib.sha256(authorization.read_bytes()).hexdigest() in report
    assert (output / authorization.name).read_bytes() == authorization.read_bytes()

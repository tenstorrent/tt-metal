# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-independent unit tests for the prefill CI summary utility (rank gating, file emission,
table rendering). No device / ttnn dependency, so these run in a plain CPU job."""

import json
import os

import pytest

from models.demos.deepseek_v3_d_p.utils import prefill_summary_utils as psu

_RANK_VARS = ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK")


@pytest.fixture(autouse=True)
def _clean_rank_env(monkeypatch):
    """Start each test with no MPI rank vars so is_primary_rank is deterministic."""
    for var in _RANK_VARS:
        monkeypatch.delenv(var, raising=False)


def test_render_table_layout():
    lines = psu.render_table(["chunk", "median"], [["chunk 0", "1.239s"], ["chunk 10", "1.4s"]])
    # border, header, border, 2 rows, border
    assert len(lines) == 6
    assert lines[0] == lines[2] == lines[-1]  # identical separators
    assert set(lines[0]) == {"+", "-"}
    # columns widen to the longest cell ("chunk 10") and every rendered row shares one width
    assert lines[1] == "| chunk    | median |"
    assert lines[3] == "| chunk 0  | 1.239s |"
    assert len({len(line) for line in lines}) == 1


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, True),  # non-MPI run
        ({"OMPI_COMM_WORLD_RANK": "0"}, True),
        ({"OMPI_COMM_WORLD_RANK": "1"}, False),
        ({"PMIX_RANK": "3"}, False),
        ({"PMI_RANK": "0"}, True),
    ],
)
def test_is_primary_rank(monkeypatch, env, expected):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert psu.is_primary_rank() is expected


def test_emit_summary_primary_writes_and_prints(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    lines = psu.render_table(["chunk", "median"], [["chunk 0", "1.239s"]])
    path = psu.emit_summary("perf", "run_a", "Chunk timing", lines)

    # summary_dir resolves symlinks (e.g. macOS /var -> /private/var), so compare against the resolved path.
    assert path == tmp_path.resolve() / "perf" / "run_a.md"
    content = path.read_text()
    assert content.startswith("### Chunk timing\n\n```text\n")
    assert content.rstrip().endswith("```")
    assert "\n".join(lines) in content
    # also emitted to stdout (loguru is stderr-bound here)
    assert "Chunk timing" in capsys.readouterr().out


def test_emit_summary_non_primary_is_noop(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    result = psu.emit_summary("perf", "run_a", "Chunk timing", ["x"])

    assert result is None
    assert not (tmp_path / "perf").exists()
    assert capsys.readouterr().out == ""


def test_summary_dir_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    d = psu.summary_dir("pcc")
    assert d == tmp_path.resolve() / "pcc"
    assert d.is_dir()


def test_emit_perf_metrics_writes_json_and_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    path = psu.emit_perf_metrics("run_a", "Layer perf", {"median_ms": 1.5, "samples_ms": [1.4, 1.6]})

    assert path == tmp_path.resolve() / "perf_json" / "run_a.json"
    assert json.loads(path.read_text()) == {
        "name": "run_a",
        "kind": "test_perf",
        "median_ms": 1.5,
        "samples_ms": [1.4, 1.6],
    }
    summary = (tmp_path / "perf" / "run_a.md").read_text()
    assert "median_ms" in summary and "samples_ms" not in summary
    assert "Layer perf" in capsys.readouterr().out


def test_emit_perf_metrics_record_cannot_override_name_or_kind(monkeypatch, tmp_path):
    # scaling_summary.py skips kind="test_perf"; a record must not be able to opt back in.
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    path = psu.emit_perf_metrics("run_a", "Layer perf", {"name": "other", "kind": "disagg", "median_ms": 1.5})

    rec = json.loads(path.read_text())
    assert (rec["name"], rec["kind"]) == ("run_a", "test_perf")


def test_summaries_root_without_passwd_entry(monkeypatch):
    # Containers can run as a uid with no passwd entry and no USER/LOGNAME; getuser() then raises.
    monkeypatch.delenv("PREFILL_SUMMARIES", raising=False)

    def no_user():
        raise KeyError("getpwuid(): uid not found")

    monkeypatch.setattr(psu.getpass, "getuser", no_user)
    assert psu.summaries_root().name == f"prefill_summaries_{os.getuid()}"


def test_emit_perf_metrics_non_primary_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")

    assert psu.emit_perf_metrics("run_a", "Layer perf", {"median_ms": 1.5}) is None
    assert not (tmp_path / "perf_json").exists()


def test_scaling_summary_skips_test_perf_records(tmp_path):
    from models.demos.common.prefill.runners.ci.scaling_summary import _load

    (tmp_path / "disagg.json").write_text(json.dumps({"model": "kimi_k3", "config": "sc4"}))
    (tmp_path / "layer.json").write_text(json.dumps({"model": "kimi_k3", "config": "sc1", "kind": "test_perf"}))

    assert _load(tmp_path) == {"kimi_k3": {"sc4": {"model": "kimi_k3", "config": "sc4"}}}

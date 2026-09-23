# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-independent unit tests for the prefill CI summary utility (rank gating, file emission,
table rendering). No device / ttnn dependency, so these run in a plain CPU job."""

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


# GQA rows from both slots must report their per-layer minima, while MLA/index columns remain intact.
def test_ci_pcc_matrix_reports_gqa_and_preserves_mla_index(tmp_path, capsys):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _pcc_matrix

    log = tmp_path / "stderr"
    log.write_text("[1,0]<stderr>: layer 16: K=0.99980 V=0.99950\n" "[1,0]<stderr>: layer 16: K=0.99990 V=0.99924\n")
    _pcc_matrix(tmp_path)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split() == ["layer", "K", "V"]
    assert lines[2].split() == ["16", "0.99980", "0.99924"]
    assert lines[-1].endswith("0.999240")

    with log.open("a") as output:
        output.write("slot 0 layer 16 KV PCC: nope=0.98500 pe=0.97500\n")
        output.write("slot 0 layer 16 (index rank 0) index PCC: 0.96500\n")
    _pcc_matrix(tmp_path)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split() == ["layer", "kvpe.nope", "kvpe.pe", "K", "V", "index"]
    assert lines[2].split() == ["16", "0.98500", "0.97500", "0.99980", "0.99924", "0.96500"]
    assert lines[-1].endswith("0.965000")

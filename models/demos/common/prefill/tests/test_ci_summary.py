# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.


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


def _summarize(tmp_path, monkeypatch, chunks_per_rank, *, warmup=2, real=4, context=None, config_lines=()):
    """Run summarize_ci_run.main over two ranks that each timed `chunks_per_rank` chunks (warmup first), 1 s apart."""
    import json
    import sys

    from models.demos.common.prefill.runners.ci import summarize_ci_run

    ranklogs, timing = tmp_path / "ranklogs", tmp_path / "timing"
    (ranklogs / "runner/1/rank.0").mkdir(parents=True)
    (ranklogs / "runner/1/rank.0/stderr").write_text("".join(f"{line}\n" for line in config_lines))
    timing.mkdir()
    for rank in range(2):
        # Warmup chunks run slower (compile), so a probe that lands on one is visible.
        rows = [
            f"{rank},{c},{1000.0 + c + 0.5 * rank:.6f},{900.0 if c < warmup else 400.0:.3f}\n"
            for c in range(chunks_per_rank)
        ]
        (timing / f"rank{rank}.csv").write_text("".join(rows))
    argv = ["summarize_ci_run.py", "--ranklogs", str(ranklogs), "--timing-dir", str(timing)]
    argv += ["--real-chunks", str(real), "--warmup-chunks", str(warmup), "--chunk-size", "5120"]
    argv += ["--probe-chunks", "0,1,3", "--summary-name", "kimi_k3_sc4"]
    if context is not None:
        (tmp_path / "context.json").write_text(json.dumps(context))
        argv += ["--run-context", str(tmp_path / "context.json")]
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path / "summaries"))
    monkeypatch.setattr(sys, "argv", argv)
    summarize_ci_run.main()
    return json.loads((tmp_path / "summaries/perf_json/kimi_k3_sc4.json").read_text())


_CONFIG_LINES = (
    "[1,0]<stderr>:   PREFILL_NUM_LAYERS                  = 24",
    "  PREFILL_USE_TRACE                   = True (trace_region=256 MB)",
    "  PREFILL_NUM_USERS                   = 83",
)


def test_ci_summary_records_what_the_run_was(tmp_path, monkeypatch):
    hosts = [
        {"host": "glx-a", "hostname": "a", "tdp_limit_max": 150, "power_class": "14kw"},
        {"host": "glx-b", "hostname": "b", "tdp_limit_max": 75, "power_class": "8kw"},
    ]
    rec = _summarize(tmp_path, monkeypatch, 6, context={"commit": "abc123", "hosts": hosts}, config_lines=_CONFIG_LINES)

    assert (rec["commit"], rec["traced"], rec["num_users"], rec["num_layers"]) == ("abc123", True, 83, 24)
    assert rec["hosts"] == hosts
    assert set(rec["throughput_tok_s"]) == {"@5k", "@10k", "@20k"}
    summary = (tmp_path / "summaries/perf/kimi_k3_sc4.md").read_text()
    assert "commit=abc123 traced=True num_users=83 num_layers=24" in summary
    assert "host glx-b: TDP_LIMIT_MAX=75 -> 8kw" in summary


def test_ci_summary_partial_run_keeps_measured_chunks_on_their_probes(tmp_path, monkeypatch):
    # Died after 2 warmup + 2 of 4 measured chunks. Keeping the last 4 would label warmup chunk 0 as @5k and
    # lose @10k's identity; the measured chunks must stay at chunk index 0 and 1, and the unreached @20k absent.
    complete = _summarize(tmp_path / "complete", monkeypatch, 6)
    partial = _summarize(tmp_path / "partial", monkeypatch, 4)

    assert partial["chunk_index"] == {"@5k": 0, "@10k": 1}
    assert partial["chunk_time_ms"] == {label: complete["chunk_time_ms"][label] for label in ("@5k", "@10k")}
    assert "@20k" not in partial["throughput_tok_s"]


def test_ci_summary_without_timing_still_records_the_run(tmp_path, monkeypatch):
    rec = _summarize(tmp_path, monkeypatch, 0, context={"commit": "abc123", "hosts": []}, config_lines=_CONFIG_LINES)
    assert (rec["commit"], rec["traced"]) == ("abc123", True)

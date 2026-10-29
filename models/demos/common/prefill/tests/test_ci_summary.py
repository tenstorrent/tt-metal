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


def _retire_line(start, first, retire, warmup=0):
    return (
        f"[producer] CHUNK_RETIRE slot=0 start={start} end={start + 1024} first={first:.6f} "
        f"retire={retire:.6f} warmup={warmup}\n"
    )


def test_perf_metrics_time_chunks_from_first_ack_to_last(tmp_path):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _perf_metrics, _spans_from_logs

    lines = [_retire_line(0, 1.0, 1.5, warmup=1)]
    lines += [_retire_line(i * 1024, 100.0 + i, 100.0 + i + 3.0) for i in range(10)]
    (tmp_path / "stderr").write_text("".join(lines))
    spans = _spans_from_logs(tmp_path, real_chunks=10)
    out, rec = _perf_metrics(spans, chunk_size=1024, pipeline_ranks=4)
    assert rec["num_chunks"] == 10
    assert rec["pipeline_ranks"] == 4
    assert all(abs(v - 3000.0) < 1e-6 for v in rec["chunk_time_ms"].values())
    last = max(rec["ttft_s"], key=lambda k: rec["chunk_index"][k])
    assert abs(rec["ttft_s"][last] - 12.0) < 1e-6
    assert all(abs(v - 1024.0) < 1e-6 for v in rec["throughput_tok_s"].values())


def test_perf_metrics_absent_without_retire_lines(tmp_path):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _perf_metrics, _spans_from_logs

    (tmp_path / "stderr").write_text("[pp rank 0] CHUNK_START c=0 compute_start=1.0\n")
    spans = _spans_from_logs(tmp_path, real_chunks=0)
    assert spans == []
    assert _perf_metrics(spans, chunk_size=1024, pipeline_ranks=1) == ([], {})

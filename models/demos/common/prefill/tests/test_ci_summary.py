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


def test_metrics_sidecar_splits_the_config_suffix_with_its_layout(tmp_path, monkeypatch):
    import json

    from models.demos.common.prefill.runners.ci.summarize_ci_run import _publish_json

    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    for name, model, config in (
        ("kimi_k3_sc4_8x4", "kimi_k3", "sc4_8x4"),
        ("kimi_k3_sc4", "kimi_k3", "sc4"),
        ("glm53_mtp_kv_share_sc1", "glm53_mtp_kv_share", "sc1"),
    ):
        _publish_json({"num_chunks": 1}, name)
        rec = json.loads((tmp_path / "perf_json" / f"{name}.json").read_text())
        assert (rec["model"], rec["config"]) == (model, config)


def _retire_line(start, fwd, acks_ms):
    """A producer CHUNK_RETIRE line whose layer i is acked at acks_ms[i] (absolute ms)."""
    first = acks_ms[0] / 1000.0
    layers = ",".join(f"{i}-{i + 1}@{t - acks_ms[0]:.1f}" for i, t in enumerate(acks_ms))
    return (
        f"[1,0]<stderr>: [producer] CHUNK_RETIRE slot=0 start={start} end={start + 32} first={first:.6f} "
        f"retire={acks_ms[-1] / 1000.0:.6f} warmup=0 acks={len(acks_ms)} fwd={fwd} layers={layers}\n"
    )


def _two_stage_log(tmp_path, chunks):
    log = tmp_path / "stderr"
    log.write_text(
        "[1,0]<stdout>: [pp rank 0/2] mesh=(8, 4) layers=[0, 2) is_first=True\n"
        "[1,1]<stdout>: [pp rank 1/2] mesh=(8, 4) layers=[2, 4) is_first=False\n"
        + "".join(_retire_line(32 * k, fwd, acks) for k, (fwd, acks) in enumerate(chunks))
    )


# Stage 0 takes 10 ms a chunk and stage 1 30 ms, so every chunk after the first waits behind the one before it
# on stage 1 (20 ms more each time). The span counts that queue; chunk_time must not.
def test_chunk_time_excludes_queueing_between_stages(tmp_path):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import (
        _perf_metrics,
        _spans_from_logs,
        _stage_ends_from_logs,
    )

    chunks, free = [], 0.0
    for k in range(6):
        ready = 10.0 * k + 10.0
        start = max(ready, free)
        free = start + 30.0
        chunks.append((k, [10.0 * k + 5.0, ready, start + 15.0, free]))
    _two_stage_log(tmp_path, chunks)
    assert _stage_ends_from_logs(tmp_path) == [2, 4]
    _lines, rec = _perf_metrics(_spans_from_logs(tmp_path, 0), 32, 2, stage_ends=_stage_ends_from_logs(tmp_path))
    assert [round(x, 3) for x in rec["chunk_span_curve_ms"]] == [35.0, 55.0, 75.0, 95.0, 115.0, 135.0]
    assert [round(x, 3) for x in rec["queue_wait_curve_ms"]] == [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    assert [round(x, 3) for x in rec["chunk_time_curve_ms"]] == [35.0] * 6


# Two chunks packed in one forward finish stage 1 together: the second is not queued behind the first.
def test_chunk_time_does_not_queue_a_chunk_behind_its_own_forward(tmp_path):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import (
        _perf_metrics,
        _spans_from_logs,
        _stage_ends_from_logs,
    )

    _two_stage_log(tmp_path, [(0, [5.0, 10.0, 40.0, 50.0]), (0, [6.0, 12.0, 41.0, 52.0])])
    _lines, rec = _perf_metrics(_spans_from_logs(tmp_path, 0), 32, 2, stage_ends=_stage_ends_from_logs(tmp_path))
    assert rec["queue_wait_curve_ms"] == [0.0, 0.0]


# Without the runner's layer banner (one stage, or an older log) chunk_time stays the plain span.
def test_chunk_time_without_stage_layout_is_the_span(tmp_path):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _perf_metrics, _spans_from_logs

    (tmp_path / "stderr").write_text(
        _retire_line(0, 0, [0.0, 10.0, 20.0, 30.0]) + _retire_line(32, 1, [10.0, 20.0, 60.0, 90.0])
    )
    _lines, rec = _perf_metrics(_spans_from_logs(tmp_path, 0), 32, 1)
    assert [round(x, 3) for x in rec["chunk_time_curve_ms"]] == [30.0, 80.0]

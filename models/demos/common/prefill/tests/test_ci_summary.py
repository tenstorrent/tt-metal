# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
import pytest


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


def test_perf_summary_and_gate_name_what_ran(tmp_path, monkeypatch):
    import json
    import sys

    from models.demos.common.prefill.runners.ci import perf_gate
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _perf_metrics, _setup_from_logs

    # Each runner rank's start-up line, as mpirun tags it; the last stage is shorter.
    logs = tmp_path / "ranklogs"
    logs.mkdir()
    (logs / "stdout").write_text(
        "".join(
            f"[1,{r}]<stdout>: [pp rank {r}/8] mesh=(4, 4) layers=[{12 * r}, {min(12 * r + 12, 93)}) is_first={r == 0} "
            f"is_last={r == 7} chunk_size=2560 forward_chunks=4 max_seq_len=1064960 num_users=87 "
            "sp=4 tp=4 sp_axis=1 traced=1\n"
            for r in range(8)
        )
    )
    setup = _setup_from_logs(logs)
    assert setup == (
        "8 stages x [4,4] (SP4 on mesh axis 1, TP4), 93 layers as 12+12+12+12+12+12+12+9, 87 slots, "
        "2560-token chunks, 4 per forward (packed), traced"
    )
    lines, rec = _perf_metrics(_packed_spans(40, 4, 0.2, tail_s=0.2), 2560, 8, {39}, chunks_per_forward=4, setup=setup)
    assert lines[1] == f"run: {setup}"
    assert rec["setup"] == setup

    summaries = tmp_path / "summaries"
    monkeypatch.setenv("PREFILL_SUMMARIES", str(summaries))
    monkeypatch.delenv("CI_FAILURE_RECORDS", raising=False)
    perf_json = tmp_path / "kimi_k3_1_packed_sc4.json"
    perf_json.write_text(json.dumps(rec))
    label = next(iter(rec["throughput_tok_s"]))
    table = tmp_path / "baselines.json"
    table.write_text(
        json.dumps(
            {
                "margin": {"throughput_tok_s": 0.05},
                "baselines": {
                    "kimi_k3_1": {
                        "sc4_packed": {"probe": label, "8kw": {"throughput_tok_s": rec["throughput_tok_s"][label]}}
                    }
                },
            }
        )
    )
    args = [str(perf_json), "--table", str(table), "--model", "kimi_k3_1", "--config", "sc4_packed", "--power", "8kw"]
    monkeypatch.setattr(sys, "argv", ["perf_gate.py", *args])
    assert perf_gate.main() == 0
    gate = (summaries / "perf" / "kimi_k3_1_packed_sc4_gate.md").read_text().splitlines()
    assert gate[2] == f"run: {setup}; baseline kimi_k3_1/sc4_packed on the 8kw power class"


# Without a baseline the gate only reports, so it must check that the whole request retired: a run cut short reports a
# shallow rate that would otherwise pass. Runs the summarizer and the gate as CI does (--real-chunks 40).
@pytest.mark.parametrize("retired, rc", [(40, 0), (12, 1)])
def test_unbaselined_gate_fails_a_run_cut_short(tmp_path, monkeypatch, retired, rc):
    import json
    import sys

    from models.demos.common.prefill.runners.ci import perf_gate, summarize_ci_run

    monkeypatch.delenv("CI_FAILURE_RECORDS", raising=False)
    summaries = tmp_path / "summaries"
    monkeypatch.setenv("PREFILL_SUMMARIES", str(summaries))
    logs = tmp_path / "ranklogs"
    logs.mkdir()
    (logs / "stderr").write_text(
        "".join(
            f"[producer] CHUNK_RETIRE slot=0 start={c * 2560} end={(c + 1) * 2560} first={begin:.3f} "
            f"retire={end:.3f} warmup=0 acks=24\n"
            for c, (begin, end, _) in enumerate(_packed_spans(retired, 4, 0.2, tail_s=0.2))
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_ci_run.py", "--ranklogs", str(logs), "--real-chunks", "40", "--chunk-size", "2560"]
        + ["--pipeline-ranks", "8", "--chunks-per-forward", "4", "--summary-name", "run_sc4"],
    )
    summarize_ci_run.main()
    table = tmp_path / "baselines.json"
    table.write_text(json.dumps({"margin": {}, "baselines": {}}))
    perf_json = summaries / "perf_json" / "run_sc4.json"
    args = [str(perf_json), "--table", str(table), *"--model run --config sc4 --power 8kw --require 1".split()]
    monkeypatch.setattr(sys, "argv", ["perf_gate.py", *args])
    assert perf_gate.main() == rc


def _packed_spans(num_chunks, chunks_per_forward, forward_s, tail_s):
    # Every chunk of a packed forward retires at the forward's end. A short final forward is padded to a
    # full one, so it costs tail_s (near forward_s), not its share of forward_s.
    spans, t = [], 0.0
    for start in range(0, num_chunks, chunks_per_forward):
        size = min(chunks_per_forward, num_chunks - start)
        begin, t = t, t + (forward_s if size == chunks_per_forward else tail_s)
        spans += [(begin, t, 24)] * size
    return spans


# A packed forward's chunks retire together. The throughput window must span whole forwards: an edge
# inside a forward credits chunks that retired at the window's start against a later forward's time.
@pytest.mark.parametrize("cpf", [2, 4, 8])
def test_packed_throughput_windows_span_full_forwards(cpf):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _perf_metrics

    chunk, forward_s = 2560, 0.2
    rate = cpf * chunk / forward_s
    # 410 % 4 == 2 and 410 % 8 == 2: a short final forward, here costing its share of a full one.
    spans = _packed_spans(410, cpf, forward_s, tail_s=forward_s * (410 % cpf or cpf) / cpf)
    probes = {0, 19, 39, 100, 204, 409}

    _, rec = _perf_metrics(spans, chunk, 2, probes, chunks_per_forward=cpf)
    assert rec["chunks_per_forward"] == cpf
    assert list(rec["throughput_tok_s"]) == ["@2k", "@50k", "@100k", "@252k", "@512k", "@1M"]
    for label, tput in rec["throughput_tok_s"].items():
        assert abs(tput - rate) / rate < 1e-9, (label, tput)


def test_packed_throughput_without_forward_grouping_overcounts_the_tail():
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _perf_metrics

    # The pre-fix window at chunk 409 (405..409) credits 4 chunks to the 2-chunk final forward's time.
    spans = _packed_spans(410, 4, 0.2, tail_s=0.2)
    _, rec = _perf_metrics(spans, 2560, 2, {409})
    assert abs(rec["throughput_tok_s"]["@1M"] / (4 * 2560 / 0.2) - 1.0) < 1e-9
    _, rec = _perf_metrics(_packed_spans(410, 4, 0.2, tail_s=0.1), 2560, 2, {409})
    assert abs(rec["throughput_tok_s"]["@1M"] / (4 * 2560 / 0.2) - 2.0) < 1e-9


def test_packed_tail_probe_includes_the_short_final_forward():
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _perf_metrics

    # A padded 2-chunk final forward costing a full forward: @1M spans the last full forward and the tail,
    # 6 chunks over 2 forwards' time, so the padding shows up in the number.
    spans = _packed_spans(410, 4, 0.2, tail_s=0.2)
    _, rec = _perf_metrics(spans, 2560, 2, {409}, chunks_per_forward=4)
    assert abs(rec["throughput_tok_s"]["@1M"] - 6 * 2560 / 0.4) < 1e-6


def test_packed_request_of_one_forward_reports_no_rate():
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _perf_metrics

    for n in (3, 4):  # one short forward, one full forward
        _, rec = _perf_metrics(_packed_spans(n, 4, 0.2, tail_s=0.2), 2560, 1, {0, n - 1}, chunks_per_forward=4)
        assert rec["throughput_tok_s"] == {}, n


def test_one_chunk_per_forward_window_is_unchanged():
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _perf_metrics

    # Pipelined single-chunk forwards retiring unevenly: the window stays d +/- 4 chunks, capped at the ends.
    ends = [0.1 + 0.05 * i + 0.01 * (i % 3) for i in range(410)]
    spans = [(end - 0.1, end, 24) for end in ends]
    probes = [0, 19, 204, 409]
    _, rec = _perf_metrics(spans, 2560, 2, set(probes), chunks_per_forward=1)
    expected = []
    for d in probes:
        lo, hi = max(0, d - 4), min(409, d + 4)
        expected.append((hi - lo) * 2560 / (ends[hi] - ends[lo]))
    assert list(rec["throughput_tok_s"].values()) == expected


def _pipeline_logs(
    path,
    layer_ms,
    num_chunks=40,
    chunks_per_forward=1,
    batched_acks=False,
    split=(),
    handoff_ms=0.0,
    bounded=False,
    ack_every=1,
    drain_ms=0.0,
):
    """Runner and producer log lines of a pipeline whose stage k takes layer_ms[k][i] ms for its layer i.

    All forwards are queued at the start. A forward reaches stage k handoff_ms after stage k-1 finishes it; when
    bounded, stage k also waits for stage k+1 to take its previous forward. A stage acks every ack_every layers, or
    with batched_acks all at once, drain_ms apart. A forward in `split` acks the layers either side of
    the stage 0/1 boundary in one message."""
    sizes = [len(costs) for costs in layer_ms]
    bounds = [sum(sizes[:k]) for k in range(len(sizes) + 1)]
    lines = [
        f"[1,{k}]<stdout>: [pp rank {k}/{len(sizes)}] mesh=(4, 4) layers=[{bounds[k]}, {bounds[k + 1]}) "
        f"chunk_size=2560 forward_chunks={chunks_per_forward} max_seq_len=1064960 num_users=88\n"
        for k in range(len(sizes))
    ]
    done = [0.0] * len(sizes)
    started = [0.0] * len(sizes)
    for f in range(0, num_chunks, chunks_per_forward):
        arrived, acks = 1.0, []
        prev_started = list(started)
        for k, costs in enumerate(layer_ms):
            start = max(arrived, done[k], prev_started[k + 1] if bounded and k + 1 < len(sizes) else 0.0)
            started[k] = start
            t = start
            for i, ms in enumerate(costs):
                t += ms / 1000.0
                if not batched_acks and ((i + 1) % ack_every == 0 or i + 1 == len(costs)):
                    acks.append((bounds[k] + i + 1 - ((i % ack_every) + 1), bounds[k] + i + 1, t))
            if batched_acks:
                acks += [(bounds[k] + i, bounds[k] + i + 1, t + i * drain_ms / 1000.0) for i in range(len(costs))]
            done[k] = t
            arrived = t + handoff_ms / 1000.0
        if f in split:
            b = bounds[1]
            acks = [a for a in acks if a[0] not in (b - 1, b)] + [(b - 1, b + 1, next(t for a, _, t in acks if a == b))]
            acks.sort(key=lambda a: a[2])
        first = acks[0][2]
        layer_field = ",".join(f"{a}-{z}@{(t - first) * 1000.0:.1f}" for a, z, t in acks)
        for c in range(f, min(f + chunks_per_forward, num_chunks)):
            lines.append(
                f"[1,0]<stderr>: [producer] CHUNK_RETIRE slot=0 start={c * 2560} end={(c + 1) * 2560} "
                f"first={first:.6f} retire={done[-1]:.6f} warmup=0 acks={len(acks)} layers={layer_field}\n"
            )
    path.mkdir()
    (path / "log").write_text("".join(lines))
    return path


def _uniform(work_ms, layers):
    return [[ms / n] * n for ms, n in zip(work_ms, layers)]


def _measure(logs, num_chunks=40, chunks_per_forward=1):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _perf_metrics, _spans_from_logs, _stage_ranges

    _, rec = _perf_metrics(
        _spans_from_logs(logs, num_chunks),
        2560,
        len(_stage_ranges(logs)),
        {num_chunks - 1},
        chunks_per_forward,
        stages=_stage_ranges(logs),
    )
    return rec


# The queue forms in front of the slowest stage; latency counts it only when that stage is past the first.
@pytest.mark.parametrize("slowest", [0, 1, 2])
def test_chunk_time_excludes_the_queue_wherever_it_forms(tmp_path, slowest):
    work = [100.0, 100.0, 100.0]
    work[slowest] = 110.0
    rec = _measure(_pipeline_logs(tmp_path / "logs", _uniform(work, [12, 12, 9])))
    assert rec["chunk_time_ms"]["@100k"] == pytest.approx(310.0, abs=0.5)
    assert rec["stage_ms"]["@100k"] == pytest.approx(work, abs=0.5)
    if slowest:
        assert rec["latency_ms"]["@100k"] > 400.0 and rec["queued_ms"]["@100k"] > 100.0
    else:
        assert rec["latency_ms"]["@100k"] < 310.0


def test_stages_that_ack_together_have_no_chunk_time(tmp_path, monkeypatch):
    import json

    from models.demos.common.prefill.runners.ci.summarize_ci_run import (
        _perf_metrics,
        _publish_json,
        _spans_from_logs,
        _stage_ranges,
    )

    logs = _pipeline_logs(
        tmp_path / "logs", _uniform([120.0, 100.0], [12, 12]), chunks_per_forward=4, batched_acks=True, drain_ms=0.2
    )
    lines, rec = _perf_metrics(_spans_from_logs(logs, 40), 2560, 2, {0, 19, 39}, 4, stages=_stage_ranges(logs))
    assert len(rec["chunk_time_curve_ms"]) == 40 and rec["latency_ms"]["@100k"] > 0
    assert [line for line in lines if "chunk_time" in line] == [
        "chunk_time: n/a (every forward has a stage whose acks do not show when it started: acked as a batch or "
        "in one ack)"
    ]
    # n/a is null in the metrics JSON.
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    _publish_json(rec, "kimi_k3_1_packed_sc1")
    sidecar = json.loads((tmp_path / "perf_json" / "kimi_k3_1_packed_sc1.json").read_text(), parse_constant=pytest.fail)
    assert sidecar["chunk_time_ms"]["@100k"] is None
    assert set(sidecar["chunk_time_curve_ms"]) == {None}


# Packed stage times are per chunk; a short final forward is divided by its own chunk count.
def test_packed_stage_times_are_per_chunk(tmp_path):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _spans_from_logs, _stage_ranges, _stage_times

    logs = _pipeline_logs(tmp_path / "logs", _uniform([120.0, 100.0], [12, 12]), num_chunks=38, chunks_per_forward=4)
    rec = _measure(logs, 38, 4)
    assert rec["stage_ms"]["@95k"] == pytest.approx([30.0, 25.0], abs=0.2)
    assert rec["chunk_time_ms"]["@95k"] == pytest.approx(55.0, abs=0.2)
    per_forward = _stage_times(_spans_from_logs(logs, 38), _stage_ranges(logs), 4)
    assert per_forward[37][0] == pytest.approx([60.0, 50.0], abs=0.2)  # chunks 36 and 37


def test_a_forward_that_cannot_be_split_by_stage_does_not_lose_the_probe(tmp_path):
    rec = _measure(_pipeline_logs(tmp_path / "logs", _uniform([100.0, 110.0], [12, 12]), split={38}))
    assert rec["chunk_time_ms"]["@100k"] == pytest.approx(210.0, abs=0.5)
    # A window mostly unsplit is n/a.
    rec = _measure(_pipeline_logs(tmp_path / "most", _uniform([100.0, 110.0], [12, 12]), split=set(range(33, 40))))
    assert rec["chunk_time_ms"]["@100k"] != rec["chunk_time_ms"]["@100k"]


# Back-pressure (a one-forward buffer between stages) and hand-offs fall outside a stage's own first and last ack.
def test_chunk_time_excludes_back_pressure_and_hand_offs(tmp_path):
    work = [100.0, 60.0, 150.0]
    rec = _measure(_pipeline_logs(tmp_path / "logs", _uniform(work, [12, 12, 9]), bounded=True, handoff_ms=5.0))
    assert rec["chunk_time_ms"]["@100k"] == pytest.approx(310.0, abs=0.5)
    assert rec["stage_ms"]["@100k"] == pytest.approx(work, abs=0.5)
    assert rec["latency_ms"]["@100k"] > 400.0
    assert rec["throughput_tok_s"]["@100k"] == pytest.approx(2560 / 0.150, rel=1e-3)


def test_a_stage_acking_every_two_layers_is_timed(tmp_path):
    rec = _measure(_pipeline_logs(tmp_path / "logs", _uniform([120.0, 100.0], [12, 12]), ack_every=2))
    assert rec["stage_ms"]["@100k"] == pytest.approx([120.0, 100.0], abs=0.5)


# Scaling counts the first ack's layers at the stage average, so a stage starting on a cheap layer reads high.
def test_scaling_counts_the_first_layer_at_the_stage_average(tmp_path):
    rec = _measure(_pipeline_logs(tmp_path / "logs", [[10.0, 30.0, 30.0, 30.0], [25.0] * 4]))
    assert rec["stage_ms"]["@100k"] == pytest.approx([120.0, 100.0], abs=0.5)  # stage 0 does 100 ms of work


def test_scaling_report_shows_an_unmeasured_probe_as_missing_and_drops_an_unmeasured_metric():
    from models.demos.common.prefill.runners.ci.scaling_summary import _ref_rows, _rows

    rows = _ref_rows({"chunk_time_ms": {"@100k": None, "@1M": 796.4}, "throughput_tok_s": {"@1M": 22788.0}})
    assert ["chunk_time @100k", "-"] in rows
    assert ["chunk_time @1M", "796 ms"] in rows
    rows = _ref_rows({"chunk_time_ms": {"@100k": None, "@1M": None}, "throughput_tok_s": {"@1M": 22788.0}})
    assert [row[0] for row in rows] == ["throughput @1M"]
    unmeasured = {"chunk_time_ms": {"@1M": None}, "throughput_tok_s": {"@1M": 22788.0}}
    assert [row[0] for row in _rows(unmeasured, unmeasured)] == ["throughput @1M"]

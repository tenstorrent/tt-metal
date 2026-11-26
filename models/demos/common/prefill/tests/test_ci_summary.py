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

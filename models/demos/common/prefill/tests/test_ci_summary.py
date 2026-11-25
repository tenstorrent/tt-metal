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


def _packed_spans(num_chunks, chunks_per_forward, forward_s, first_s=0.0):
    # Every chunk of a packed forward retires at the forward's end; a short final forward costs its share.
    spans, t = [], first_s
    for start in range(0, num_chunks, chunks_per_forward):
        size = min(chunks_per_forward, num_chunks - start)
        begin, t = t, t + forward_s * size / chunks_per_forward
        spans += [(begin, t, 24)] * size
    return spans


# A packed forward's chunks retire together. The throughput window must span whole forwards, or a tail
# probe credits chunks that retired at the window's start against a short final forward's time.
def test_packed_throughput_windows_span_whole_forwards():
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _perf_metrics

    chunk, cpf, forward_s = 2560, 4, 0.2
    rate = cpf * chunk / forward_s
    spans = _packed_spans(410, cpf, forward_s)  # 102 full forwards and a 2-chunk tail
    probes = {0, 19, 39, 100, 204, 409}

    _, rec = _perf_metrics(spans, chunk, 2, probes, chunks_per_forward=cpf)
    assert rec["chunks_per_forward"] == cpf
    for label, tput in rec["throughput_tok_s"].items():
        assert abs(tput - rate) / rate < 1e-9, (label, tput)

    # One chunk per forward assumed: the tail credits 4 chunks to the 2-chunk final forward, twice the rate.
    _, rec = _perf_metrics(spans, chunk, 2, probes)
    assert abs(rec["throughput_tok_s"]["@1M"] / rate - 2.0) < 1e-9


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

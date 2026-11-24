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


def _d2d_line(rank, c, start, post, recv, free):
    where = "slot=- start=-" if start is None else f"slot=0 start={start}"
    return (
        f"[1,{rank}]<stderr>: 2026-11-24 | INFO | [pp rank {rank}] D2D_TIMING c={c} {where} "
        f"post={post:.6f} recv={recv:.6f} free={free:.6f}\n"
    )


def _write_stage_logs(root, offsets, warmup=2, chunks=3, chunk_size=10):
    """offsets[rank] = (post, recv, free) seconds past each chunk's whole second; one warmup request of `warmup`
    chunks first, then the measured request of `chunks`, then the shutdown record."""
    for rank, (post, recv, free) in offsets.items():
        log = root / "runner" / "1" / f"rank.{rank}" / "stderr"
        log.parent.mkdir(parents=True)
        lines = [f"[pp rank {rank}/{len(offsets)}] mesh=(8, 4) layers=[{rank}, {rank + 1}) is_first=x\n"]
        starts = [i * chunk_size for i in range(warmup)] + [i * chunk_size for i in range(chunks)] + [None]
        for c, start in enumerate(starts):
            t = float(c)
            lines.append(_d2d_line(rank, c, start, t + post, t + recv, t + free))
        log.write_text("".join(lines))


def test_stage_metrics_reads_each_rank_clock_and_takes_the_smaller_side_of_a_boundary(tmp_path):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _stage_metrics, _stage_records

    # Per chunk (one second apart): rank 0 hands off at +0.0 and its link is free at +0.2; rank 1 receives 50 ms after
    # it posts at +0.5; rank 2 receives 200 ms after it posts at +0.7. So:
    #   rank 0: fwd = 1.0 - 0.2 = 800 ms, send = 200 ms;  rank 1: recv = 50 ms, send = 0.6 - 0.5 = 100 ms
    #   rank 2: recv = 200 ms;  boundary 0->1 = min(200, 50) = 50 ms, 1->2 = min(100, 200) = 100 ms
    _write_stage_logs(tmp_path, {0: (0.0, 0.1, 0.2), 1: (0.5, 0.55, 0.6), 2: (0.7, 0.9, 0.95)})
    records, stages, notes = _stage_records(tmp_path, real_chunks=3, chunk_size=10)
    assert notes == [] and sorted(records) == [0, 1, 2]
    assert stages[1] == "layers=[1, 2)"
    lines, rec = _stage_metrics(records, stages, notes, chunk_size=10, probe_chunks={0, 2})
    cell = rec["stage_ms"]["0"]["@10"]
    assert round(cell["fwd"], 3) == 800.0 and round(cell["cycle"], 3) == 1000.0 and round(cell["send"], 3) == 200.0
    assert "recv" not in cell
    assert round(rec["stage_ms"]["2"]["@30"]["recv"], 3) == 200.0 and "send" not in rec["stage_ms"]["2"]["@30"]
    assert round(rec["boundary_ms"]["0->1"]["@30"]["transfer"], 3) == 50.0
    assert round(rec["boundary_ms"]["1->2"]["@10"]["transfer"], 3) == 100.0
    assert any(line.startswith("  0->1") for line in lines)


def test_stage_records_leave_out_a_rank_without_an_end_or_with_foreign_chunks(tmp_path):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _stage_records

    _write_stage_logs(tmp_path, {0: (0.0, 0.1, 0.2), 1: (0.5, 0.55, 0.6)})
    log = tmp_path / "runner" / "1" / "rank.1" / "stderr"
    log.write_text("".join(line for line in log.read_text().splitlines(True) if "slot=-" not in line))
    records, _, notes = _stage_records(tmp_path, real_chunks=3, chunk_size=10)
    assert sorted(records) == [0] and "rank 1: no shutdown record" in notes[0]
    # Asking for more chunks than the request holds reaches into the warmup: not the request's chunks in order.
    records, _, notes = _stage_records(tmp_path, real_chunks=4, chunk_size=10)
    assert records == {} and "rank 0: its last 4 chunks" in notes[0]

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


# The bh_sc1 pool mixes 8 kW and 14 kW hosts: the scaling table must name each leg's power class and warn on a mix.
def test_scaling_summary_flags_mixed_power_classes(tmp_path, capsys, monkeypatch):
    import json
    import sys

    from models.demos.common.prefill.runners.ci import scaling_summary

    common = {"model": "glm53", "chunk_size": 5120, "num_chunks": 204, "max_seq": 1044480}
    probes = {"@5k": 1.0}
    legs = {"sc1": ("14kw", 1, 11730.0), "sc4": ("8kw", 4, 44818.0)}
    for config, (power, ranks, tps) in legs.items():
        rec = dict(common, config=config, power_class=power, pipeline_ranks=ranks)
        rec.update(chunk_time_ms=probes, ttft_s=probes, throughput_tok_s={"@5k": tps})
        (tmp_path / f"glm53_{config}.json").write_text(json.dumps(rec))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(sys, "argv", ["scaling_summary.py", "--metrics-dir", str(tmp_path)])
    scaling_summary.main()
    out = capsys.readouterr().out
    assert "power: sc1 14kw | sc4 8kw" in out
    assert "::warning::glm53: sc1 ran on a 14kw host and sc4 on 8kw" in out

    for config in legs:
        path = tmp_path / f"glm53_{config}.json"
        rec = json.loads(path.read_text())
        rec["power_class"] = "8kw"
        path.write_text(json.dumps(rec))
    scaling_summary.main()
    out = capsys.readouterr().out
    assert "power: sc1 8kw | sc4 8kw" in out
    assert "::warning::" not in out

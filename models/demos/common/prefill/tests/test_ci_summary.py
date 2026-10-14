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


# A batched step's lanes are chunks c, c + 1, ..., each with the step's start and duration.
def test_ci_timing_from_ranklogs_expands_batched_steps(tmp_path):
    from models.demos.common.prefill.runners.ci.summarize_ci_run import _timing_from_ranklogs

    (tmp_path / "rank0.log").write_text(
        "[pp rank 0] CHUNK_START c=0 compute_start=10.000000 slot=0 [0,2048) provided=0\n"
        "[pp rank 0] CHUNK_COMPUTE c=0 compute_ms=50.000\n"
        "[pp rank 0] STEP_START c=1 compute_start=11.000000 lanes=[(0, 2048, 4096), (1, 0, 2048)]\n"
        "[pp rank 0] STEP_COMPUTE c=1 lanes=2 compute_ms=80.000\n"
    )
    assert _timing_from_ranklogs(tmp_path) == {0: {0: [10.0, 50.0], 1: [11.0, 80.0], 2: [11.0, 80.0]}}

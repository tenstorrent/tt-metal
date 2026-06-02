import numpy as np
import torch
from models.experimental.opt_transfer.references.seamless_m4t_v2 import SeamlessBlock
from models.experimental.opt_transfer.verify import (
    drift_metrics,
    golden_outputs,
    pcc,
    perf_gain_pct,
    perf_gate_pass,
)


def test_pcc_identity_is_one():
    a = torch.randn(4, 4)
    assert pcc(a, a.clone()) > 0.999


def test_golden_outputs_runs_reference():
    blk = SeamlessBlock(1024, 16)
    x = torch.randn(1, 8, 1024)
    out = golden_outputs(blk, (x,))
    assert out.shape == (1, 8, 1024)


def test_drift_detects_late_divergence():
    T, V = 100, 50
    g = np.random.randn(T, V)
    f = g.copy()
    f[80:] = np.random.randn(20, V)
    m = drift_metrics(g, f)
    assert 70 <= m["first_divergence_step"] <= 85
    assert m["token_match_rate"] < 1.0


def test_drift_identical_trajectories():
    g = np.random.randn(40, 30)
    m = drift_metrics(g, g.copy())
    assert m["token_match_rate"] == 1.0
    assert m["first_divergence_step"] == 40


def test_perf_gain_pct():
    assert abs(perf_gain_pct(naive_ms=100.0, fused_ms=60.0) - 40.0) < 1e-6


def test_perf_gate_rejects_regression():
    assert perf_gate_pass(naive_ms=100.0, fused_ms=99.5, min_gain_pct=2.0) is False
    assert perf_gate_pass(naive_ms=100.0, fused_ms=80.0, min_gain_pct=2.0) is True


import ast

from models.experimental.opt_transfer.verify import emit_shape_test
from models.experimental.opt_transfer.schema import KBEntry, PatternKind, FusionProposal


def test_emit_shape_test_writes_runnable_artifact(tmp_path):
    e = KBEntry(
        "nlp_create_qkv_heads",
        "ttnn.experimental.nlp_create_qkv_heads",
        "attention.qkv",
        PatternKind.HORIZONTAL_MERGE,
        ["linear"],
        {},
        {},
        "concat_qkv",
        "s",
    )
    p = FusionProposal(
        "nlp_create_qkv_heads",
        "ttnn.experimental.nlp_create_qkv_heads",
        ["q", "k", "v"],
        {"num_heads": 16},
        "concat_qkv",
        "",
        "s",
    )
    path = emit_shape_test(
        e,
        p,
        {"q": {"weight": torch.randn(4, 4), "bias": None}},
        {"H": 16},
        torch.randn(2, 2),
        torch.randn(2, 2),
        tmp_path,
    )
    assert path.exists()
    ast.parse(path.read_text())
    assert list((tmp_path / "fixtures").glob("*.pt"))

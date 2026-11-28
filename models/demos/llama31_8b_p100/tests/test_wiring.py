# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ast
import pathlib
import sys
import types

import pytest

import models.common.sampling.tt_sampling as tt_sampling
import models.ttt_compat.tt.attention as tt_attention
import models.ttt_compat.tt.decoder as tt_decoder
import models.ttt_compat.tt.lm_head as tt_lm_head
import models.ttt_compat.tt.mlp as tt_mlp
import models.ttt_compat.tt.model as tt_model
from models.demos.llama31_8b_p100.tt import p100, wiring
from models.demos.llama31_8b_p100.tt.attention import Attention
from models.demos.llama31_8b_p100.tt.decoder import TransformerBlock, fuse_norms
from models.demos.llama31_8b_p100.tt.lm_head import LMHead
from models.demos.llama31_8b_p100.tt.mlp import MLP
from models.demos.llama31_8b_p100.tt.model import Transformer

PKG = "models.demos.llama31_8b_p100.tt"
TT_DIR = pathlib.Path(__file__).resolve().parents[1] / "tt"
OPS = sorted(p.stem[len("ops_") :] for p in TT_DIR.glob("ops_*.py"))
HANDLES = {
    "matmul_bh": [("attention", "_matmul_bh"), ("lm_head", "_matmul_bh"), ("mlp", "_matmul_bh")],
    "rotary_bf8": [("attention", "_rotary_bf8")],
    "sdpa_bh": [("attention", "_sdpa_bh")],
    "qkv_heads_decode": [("attention", "_qkv_heads_decode")],
    "concat_heads_decode": [("attention", "_concat_heads_decode")],
    "rms_norm_residual": [("decoder", "_rms_norm_residual")],
    "rms_norm_residual_decode": [("decoder", "_rms_norm_residual_decode")],
    "decode_cross_norm": [("decoder", "_decode_cross_norm")],
    "greedy_argmax": [],
}
NOT_P100 = {
    "P150": dict(device_name="P150", banks=8),
    "P100 with 8 DRAM banks": dict(banks=8),
    "N150": dict(device_name="N150", banks=12),
    "P150x4": dict(device_name="P150x4", num_devices=4, banks=8),
    "P100 mesh of 2": dict(num_devices=2),
    "galaxy": dict(device_name="TG", num_devices=32, galaxy=True, banks=12),
    "other model": dict(model="Llama-3.2-3B"),
}


def fake_args(device_name="P100", num_devices=1, banks=7, model="Llama-3.1-8B", galaxy=False):
    return types.SimpleNamespace(
        base_model_name=model,
        device_name=device_name,
        num_devices=num_devices,
        is_galaxy=galaxy,
        dram_grid_size=types.SimpleNamespace(x=banks, y=1),
    )


@pytest.fixture
def build(monkeypatch):
    monkeypatch.setattr(p100, "has_ops", lambda *names: True)
    seen = {}

    def parent_init(self, *a, **k):
        self.args = seen["args"]
        seen.setdefault("kwargs", k)

    def model_init(self, *a, **k):
        seen["model_kwargs"] = k
        self.args = seen["args"]
        self.sampling = types.SimpleNamespace(tt_sampling=object.__new__(tt_sampling.TTSampling))

    for cls in (tt_attention.Attention, tt_lm_head.LMHead, tt_decoder.TransformerBlock, tt_mlp.MLP):
        monkeypatch.setattr(cls, "__init__", parent_init)
    monkeypatch.setattr(tt_model.Transformer, "__init__", model_init)

    def run(args, hidden=()):
        monkeypatch.setattr(wiring, "_mods", {})
        for op in OPS:
            monkeypatch.delitem(sys.modules, f"{PKG}.ops_{op}", raising=False)
        for op in hidden:
            monkeypatch.setitem(sys.modules, f"{PKG}.ops_{op}", None)
        seen.clear()
        seen["args"] = args
        wired = wiring.on("matmul_bh", args) is not None
        built = {
            "attention": Attention(),
            "lm_head": LMHead(),
            "decoder": TransformerBlock(),
            "mlp": MLP(None, None, args, {}, None, 0, None, None, prefetcher=object() if wired else None),
            "model": Transformer(),
        }
        return built, seen

    return run


def test_every_op_has_a_wiring_entry():
    assert OPS and sorted(HANDLES) == OPS


def test_p100_wires_every_op(build):
    built, seen = build(fake_args())
    for op, places in HANDLES.items():
        for layer, attr in places:
            assert getattr(built[layer], attr) is wiring.op(op), (op, layer)
    assert seen["model_kwargs"]["attention_class"] is Attention
    assert seen["model_kwargs"]["block_class"] is TransformerBlock
    assert seen["model_kwargs"]["lm_head_cls"] is LMHead
    assert type(built["model"].sampling.tt_sampling).__name__ == "Llama31_8BBhTTSampling"
    assert fuse_norms(built["model"].args, "prefill") and fuse_norms(built["model"].args, "decode")


@pytest.mark.parametrize("op", OPS)
def test_each_op_removable(build, op):
    built, _ = build(fake_args(), hidden=[op])
    assert wiring.op(op) is None
    for other, places in HANDLES.items():
        for layer, attr in places:
            handle = getattr(built[layer], attr)
            assert (handle is None) if other == op else (handle is not None), (other, layer)
    greedy = type(built["model"].sampling.tt_sampling).__name__ == "Llama31_8BBhTTSampling"
    assert greedy == (op != "greedy_argmax")
    assert built["mlp"].w13 is None or op != "matmul_bh"
    assert fuse_norms(built["model"].args, "prefill") == (op != "rms_norm_residual")
    assert fuse_norms(built["model"].args, "decode") == (op not in ("rms_norm_residual_decode", "decode_cross_norm"))


@pytest.mark.parametrize("where", sorted(NOT_P100))
def test_not_p100_runs_tt_path(build, where):
    args = fake_args(**NOT_P100[where])
    assert not p100.enabled(args)
    built, _ = build(args)
    for op, places in HANDLES.items():
        assert wiring.on(op, args) is None, op
        for layer, attr in places:
            assert getattr(built[layer], attr) is None, (op, layer)
    assert built["mlp"].w13 is None
    assert type(built["model"].sampling.tt_sampling) is tt_sampling.TTSampling
    assert not fuse_norms(args, "prefill") and not fuse_norms(args, "decode")


def test_ops_not_built_runs_tt_path(build, monkeypatch):
    monkeypatch.setattr(p100, "has_ops", lambda *names: False)
    built, _ = build(fake_args())
    for op, places in HANDLES.items():
        for layer, attr in places:
            assert getattr(built[layer], attr) is None, (op, layer)


def test_op_modules_are_only_reached_through_wiring():
    for path in TT_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            assert not any(n.rsplit(".", 1)[-1].startswith("ops_") for n in names), (path.name, names)

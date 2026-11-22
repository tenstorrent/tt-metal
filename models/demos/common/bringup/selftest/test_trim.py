# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F47: a layer subset's checkpoint is trimmed after the HF sanity and parity passed."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.intake import check_hf_sanity, trim_checkpoint
from models.demos.common.bringup.intake.trim_checkpoint import FULL_INDEX, INDEX, MARKER, classify, kept
from models.demos.common.bringup.plan.ledger_gen import generate
from models.demos.common.bringup.plan.memory import checkpoint_tensors
from models.demos.common.bringup.selftest.conftest import got
from models.demos.common.bringup.selftest.fixture_model import Reference

SANITY = {"revision_ok": 1, "text_top1_acc": 0.7, "smoke_ok": 1}


def _t(seed, *shape):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def make_ckpt(hf):
    """Shard a: layer 0 only (kept); b: layer 2 only (dropped); c: layers 0-2 + embed + mtp (rewritten)."""
    hf.mkdir(parents=True)
    shards = {
        "a.safetensors": {"model.layers.0.attn.w": _t(1, 4, 8)},
        "b.safetensors": {"model.layers.2.attn.w": _t(2, 4, 8)},
        "c.safetensors": {
            "model.layers.0.mlp.experts.0.w": _t(3, 8, 8),
            "model.layers.1.mlp.experts.0.w": _t(4, 8, 8),
            "model.layers.2.mlp.experts.0.w": _t(5, 8, 8),
            "model.embed_tokens.weight": _t(6, 16, 8),
            "model.mtp_layers.0.w": _t(7, 8, 8),
            "model.layers.0.sink": torch.tensor([1.0, float("nan")]),
        },
    }
    wm = {}
    for f, ts in shards.items():
        save_file(ts, str(hf / f))
        wm.update({n: f for n in ts})
    (hf / INDEX).write_text(json.dumps({"metadata": {"tp_size": 4}, "weight_map": wm}))
    meta = hf / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    for f in shards:
        (meta / f"{f}.metadata").write_text("0" * 40 + "\n")
    return {n: t for ts in shards.values() for n, t in ts.items()}


@pytest.fixture
def subset(fx, tmp_path):
    return lambda **o: Spec.load(
        fx(
            layers="0",
            paths={"art": str(tmp_path / "art"), "repo": str(tmp_path / "repo"), "hf": str(tmp_path / "hf")},
            intake={"smoke": {"prompt": "capital?", "expect": "Paris"}},
            **o,
        )
    )


def test_ledger_adds_r4_only_for_an_owned_subset(fx, subset, tmp_path):
    r4 = [t for t in generate(subset(), Reference())["tasks"] if t["id"] == "R.4"]
    assert len(r4) == 1 and r4[0]["deps"] == ["R.2"] and "trim_checkpoint" in r4[0]["gate"]["cmd"]
    assert r4[0]["gate"]["metrics"]["trim_verify_errors"] == "== 0" and not r4[0].get("role")
    full = Spec.load(fx())
    assert not [t for t in generate(full, Reference())["tasks"] if t["id"] == "R.4"]
    off = subset(checkpoint={"trim": False})
    assert not [t for t in generate(off, Reference())["tasks"] if t["id"] == "R.4"]
    assert not trim_checkpoint.applies(subset(prior="models/demos/other"))
    assert trim_checkpoint.applies(Spec.load(fx(checkpoint={"trim_drop": ["model.mtp_layers.*"]})))
    assert trim_checkpoint.keep_layers(subset(hf={"parity_layers": 2})) == 2


def test_classify_and_kept():
    assert kept("model.language_model.layers.3.x", 4, []) and not kept("model.layers.4.x", 4, [])
    assert kept("model.mtp_layers.0.x", 1, []) and not kept("model.mtp_layers.0.x", 1, ["model.mtp_layers.*"])
    wm = {"model.layers.0.a": "s1", "model.layers.1.a": "s2", "model.layers.0.b": "s3", "model.layers.1.b": "s3"}
    assert classify(wm, 1, []) == {"s1": "keep", "s2": "drop", "s3": "rewrite"}


def test_trim_keeps_bytes_and_the_full_map(subset, tmp_path):
    s = subset(checkpoint={"trim_drop": ["model.mtp_layers.*"]})
    hf = tmp_path / "hf"
    orig = make_ckpt(hf)
    full_map = checkpoint_tensors(hf)
    m = trim_checkpoint.trim(s, hf, dict(SANITY))
    assert m["status"] == "done" and m["verify_errors"] == 0 and m["keep_layers"] == 1
    assert sorted(p.name for p in hf.glob("*.safetensors")) == ["a.safetensors", "subset-c.safetensors"]
    idx = json.loads((hf / INDEX).read_text())
    assert idx["metadata"]["tp_size"] == 4 and (hf / FULL_INDEX).exists()
    want = {
        "model.layers.0.attn.w",
        "model.layers.0.mlp.experts.0.w",
        "model.embed_tokens.weight",
        "model.layers.0.sink",
    }
    assert set(idx["weight_map"]) == want and m["kept_tensors"] == 4 and m["dropped_tensors"] == 4
    for n, f in idx["weight_map"].items():
        with safe_open(str(hf / f), framework="pt") as h:
            x = h.get_tensor(n)
        assert torch.equal(x.view(torch.uint8), orig[n].view(torch.uint8)), n  # NaN included
    assert checkpoint_tensors(hf) == full_map  # the plan and checkpoint gates still see the whole model
    meta = hf / ".cache" / "huggingface" / "download"
    assert sorted(p.name for p in meta.iterdir()) == ["a.safetensors.metadata"]
    assert trim_checkpoint.trim(s, hf)["t_done"] == m["t_done"]  # idempotent


def test_interrupted_trim_resumes(subset, tmp_path):
    s = subset()
    hf = tmp_path / "hf"
    make_ckpt(hf)
    (hf / MARKER).write_text(
        json.dumps(
            {"status": "trimming", "keep_layers": 1, "tensors": checkpoint_tensors(hf), "sanity": SANITY, "t": "x"}
        )
    )
    (hf / "subset-c.safetensors").write_bytes(b"garbage")  # a crash mid-rewrite
    m = trim_checkpoint.trim(s, hf)
    assert m["status"] == "done" and "mtp" in " ".join(json.loads((hf / INDEX).read_text())["weight_map"])


def test_main_refuses_without_passing_sanity(subset, tmp_path, monkeypatch):
    s = subset()
    make_ckpt(tmp_path / "hf")
    monkeypatch.setenv("BRINGUP_SPEC", str(s.path))
    monkeypatch.setenv(M.TASK_ENV, "R.1")
    M.record("revision_ok", 1)
    M.record("text_top1_acc", 0.1)  # below the floor
    M.record("smoke_ok", 1)
    monkeypatch.setenv(M.TASK_ENV, "T")
    with pytest.raises(SystemExit, match="do not pass"):
        trim_checkpoint.main([])
    assert not (tmp_path / "hf" / MARKER).exists() and (tmp_path / "hf" / "b.safetensors").exists()
    monkeypatch.setenv(M.TASK_ENV, "R.1")
    M.record("text_top1_acc", 0.7)
    monkeypatch.setenv(M.TASK_ENV, "T")
    trim_checkpoint.main([])
    g = got()
    assert g["trim_done"] == 1 and g["trim_verify_errors"] == 0 and g["trim_dropped_tensors"] == 3
    assert json.loads((tmp_path / "hf" / MARKER).read_text())["sanity"]["text_top1_acc"] == 0.7


def test_sanity_replays_after_the_trim(subset, tmp_path, monkeypatch):
    s = subset()
    hf = tmp_path / "hf"
    make_ckpt(hf)
    trim_checkpoint.trim(s, hf, dict(SANITY))
    monkeypatch.setenv("BRINGUP_SPEC", str(s.path))
    monkeypatch.setattr(check_hf_sanity, "load_model", lambda spec: pytest.fail("the whole model is gone"))
    check_hf_sanity.main([])
    g = got()
    assert g["text_top1_acc"] == 0.7 and g["smoke_ok"] == 1 and "revision_ok" in g
    assert M.load("T")["smoke_ok"]["replayed"] is True

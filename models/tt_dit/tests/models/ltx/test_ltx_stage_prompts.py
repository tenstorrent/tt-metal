# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the distilled pipeline's per-stage prompt choice. A traced stage persists the prompt in
the shared StateTensors; with reuse_prompt a later traced stage of the same gen must read those buffers
and upload nothing.

The method is AST-extracted from the pipeline source and exec'd, like ``test_ltx_seeded_noise``, so the
test needs no device build. Run: python -m pytest <file> -v -p no:cacheprovider
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest
import torch

DISTILLED_PATH = Path(__file__).resolve().parents[3] / "pipelines" / "ltx" / "pipeline_ltx_distilled.py"


class _State:
    """StateTensor's update semantics: first traced write binds the buffer, later ones copy into it."""

    def __init__(self):
        self.value = None

    def update(self, v, traced):
        if self.value is None or not traced:
            self.value = v
        else:
            self.value.copy_(v)


METHODS = ("_stage_prompts", "_prompt_host_bf16", "_write_prompts_in_place", "_log_latent_stats")


def _source(names):
    src = DISTILLED_PATH.read_text()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LTXDistilledPipeline")
    env_on = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_env_on")

    def text(node):
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        return "".join(lines[start - 1 : node.end_lineno])

    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    return text(env_on), "class P:\n" + "".join(text(n) for n in methods)


class _FakeTtnn:
    """Host writes land in the destination tensor, like copy_host_to_device_tensor on a replicated buffer."""

    bfloat16 = TILE_LAYOUT = None

    def __init__(self):
        self.writes = []

    @staticmethod
    def from_torch(t, dtype=None, layout=None):
        return t

    def copy_host_to_device_tensor(self, host, dst):
        assert host.dtype == torch.bfloat16 and host.shape == dst.shape
        dst.copy_(host)
        self.writes.append(dst)


def _pipeline():
    env_src, cls_src = _source(METHODS)
    uploads = []

    def bf16_tensor(t, device=None):
        uploads.append("a")
        return t.to(torch.bfloat16)

    fake_ttnn = _FakeTtnn()
    ns = {"torch": torch, "bf16_tensor": bf16_tensor, "ttnn": fake_ttnn, "os": __import__("os")}
    ns["logger"] = types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None)
    exec(env_src, ns)
    exec(cls_src, ns)
    p = ns["P"]()
    p.writes = fake_ttnn.writes
    p.cross_attention_dim = 6
    p._device_prompt_handoff = False
    p.mesh_device = None
    p._prompt_v, p._prompt_a = _State(), _State()

    def prepare(v):
        uploads.append("v")
        return torch.nn.functional.pad(v.unsqueeze(0), (0, p.cross_attention_dim - v.shape[-1])).to(torch.bfloat16)

    p._prepare_prompt = prepare
    return p, uploads


def test_reuse_skips_second_upload_and_returns_persisted_buffers():
    p, uploads = _pipeline()
    v, a = torch.randn(8, 4), torch.randn(8, 2)
    s1_v, s1_a = p._stage_prompts(v, a, True, False, False)
    assert uploads == ["v", "a"]
    s2_v, s2_a = p._stage_prompts(v, a, True, False, True)
    assert uploads == ["v", "a"]
    assert s2_v is s1_v is p._prompt_v.value and s2_a is s1_a is p._prompt_a.value
    assert torch.equal(s2_v[0, :, :4], v.bfloat16()) and torch.equal(s2_a[0], a.bfloat16())


def test_without_reuse_second_stage_uploads_into_same_buffer():
    p, uploads = _pipeline()
    v, a = torch.randn(8, 4), torch.randn(8, 2)
    s1_v, _ = p._stage_prompts(v, a, True, False, False)
    s2_v, _ = p._stage_prompts(v, a, True, False, False)
    assert uploads == ["v", "a", "v", "a"] and s2_v is s1_v


def test_reuse_requires_traced_persisted_prompt(expect_error):
    p, _ = _pipeline()
    v, a = torch.randn(8, 4), torch.randn(8, 2)
    with expect_error(AssertionError, ""):
        p._stage_prompts(v, a, True, False, True)
    p._stage_prompts(v, a, False, False, False)
    assert p._prompt_v.value is None
    with expect_error(AssertionError, ""):
        p._stage_prompts(v, a, False, False, True)


def _bf16_embeds(seq=8):
    # The encoder output is bf16; the pipeline hands it over as an fp32 copy.
    return torch.randn(1, seq, 4).bfloat16().float(), torch.randn(1, seq, 2).bfloat16().float()


def test_host_copy_writes_in_place_with_default_values(monkeypatch):
    monkeypatch.setenv("LTX_PROMPT_HOST_COPY", "1")
    p, uploads = _pipeline()
    v0, a0 = _bf16_embeds()
    s1_v, s1_a = p._stage_prompts(v0, a0, True, False, False)
    assert uploads == ["v", "a"] and p.writes == []  # capture gen: buffers do not exist yet
    v1, a1 = _bf16_embeds()
    ref_v = p._prepare_prompt(v1)
    uploads.clear()
    s2_v, s2_a = p._stage_prompts(v1, a1, True, False, False)
    assert uploads == [] and p.writes == [s1_v, s1_a]
    assert s2_v is s1_v and s2_a is s1_a
    assert torch.equal(s2_v, ref_v) and torch.equal(s2_a, a1.unsqueeze(0).bfloat16())


def test_host_copy_falls_back_on_shape_change_and_untraced(monkeypatch):
    monkeypatch.setenv("LTX_PROMPT_HOST_COPY", "1")
    p, uploads = _pipeline()
    p._stage_prompts(*_bf16_embeds(8), True, False, False)
    assert p._write_prompts_in_place(*_bf16_embeds(16)) is False
    uploads.clear()
    p._stage_prompts(*_bf16_embeds(8), False, False, False)
    assert uploads == ["v", "a"] and p.writes == []


@pytest.mark.parametrize(("value", "logged"), [(None, True), ("1", True), ("0", False)])
def test_latent_stats_opt_out(monkeypatch, value, logged):
    if value is None:
        monkeypatch.delenv("LTX_LATENT_STATS", raising=False)
    else:
        monkeypatch.setenv("LTX_LATENT_STATS", value)
    p, _ = _pipeline()
    calls = []

    def stats(t):
        calls.append(t)
        return {"mean": 0.0, "std": 1.0, "whiteness": 0.0, "zeros": 0.0, "nonfinite": 0}

    p._latent_stats = stats
    p._log_latent_stats("s1", torch.zeros(2, 3), torch.zeros(2, 3))
    assert len(calls) == (2 if logged else 0)

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
from pathlib import Path

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


def _pipeline():
    src = DISTILLED_PATH.read_text()
    lines = src.splitlines(keepends=True)
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "LTXDistilledPipeline")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_stage_prompts")
    uploads = []

    def bf16_tensor(t, device=None):
        uploads.append("a")
        return t.clone()

    ns = {"torch": torch, "bf16_tensor": bf16_tensor}
    exec("class P:\n" + "".join(lines[node.lineno - 1 : node.end_lineno]), ns)
    p = ns["P"]()
    p.mesh_device = None
    p._prompt_v, p._prompt_a = _State(), _State()

    def prepare(v):
        uploads.append("v")
        return v.unsqueeze(0).clone()

    p._prepare_prompt = prepare
    return p, uploads


def test_reuse_skips_second_upload_and_returns_persisted_buffers():
    p, uploads = _pipeline()
    v, a = torch.randn(8, 4), torch.randn(8, 2)
    s1_v, s1_a = p._stage_prompts(v, a, True, False)
    assert uploads == ["v", "a"]
    s2_v, s2_a = p._stage_prompts(v, a, True, True)
    assert uploads == ["v", "a"]
    assert s2_v is s1_v is p._prompt_v.value and s2_a is s1_a is p._prompt_a.value
    assert torch.equal(s2_v[0], v) and torch.equal(s2_a[0], a)


def test_without_reuse_second_stage_uploads_into_same_buffer():
    p, uploads = _pipeline()
    v, a = torch.randn(8, 4), torch.randn(8, 2)
    s1_v, _ = p._stage_prompts(v, a, True, False)
    s2_v, _ = p._stage_prompts(v, a, True, False)
    assert uploads == ["v", "a", "v", "a"] and s2_v is s1_v


def test_reuse_requires_traced_persisted_prompt(expect_error):
    p, _ = _pipeline()
    v, a = torch.randn(8, 4), torch.randn(8, 2)
    with expect_error(AssertionError, ""):
        p._stage_prompts(v, a, True, True)
    p._stage_prompts(v, a, False, False)
    assert p._prompt_v.value is None
    with expect_error(AssertionError, ""):
        p._stage_prompts(v, a, False, True)

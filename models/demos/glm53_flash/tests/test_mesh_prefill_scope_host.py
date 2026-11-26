# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``GLMMesh.prefill_scope``: the gather-mode TP all-reduce (``GLM53_TP_ALLREDUCE=gather``) is decode-only; inside a
prefill scope ``tp_allreduce_gather`` is False whatever the shape (2026-10-26: 16-row prefill slabs passed the
shape gate and cost 48 ms of 128-token TTFT on the 8x4)."""

from __future__ import annotations

import pytest

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.tt.mesh import GLMMesh  # noqa: E402


def test_prefill_scope_disables_gather(monkeypatch):
    mesh = GLMMesh(device=None, rows=8, cols=4)
    monkeypatch.setenv("GLM53_TP_ALLREDUCE", "gather")
    assert mesh.tp_allreduce_gather and not mesh.in_prefill
    with mesh.prefill_scope():
        assert mesh.in_prefill and not mesh.tp_allreduce_gather
        with mesh.prefill_scope():  # nested (a layer inside a model prefill)
            assert not mesh.tp_allreduce_gather
        assert mesh.in_prefill
    assert not mesh.in_prefill and mesh.tp_allreduce_gather
    monkeypatch.setenv("GLM53_TP_ALLREDUCE", "ccl")
    assert not mesh.tp_allreduce_gather


def test_prefill_scope_survives_exceptions():
    mesh = GLMMesh(device=None, rows=2, cols=4)
    raised = False
    try:
        with mesh.prefill_scope():
            raise RuntimeError("inside")
    except RuntimeError:
        raised = True
    assert raised and not mesh.in_prefill

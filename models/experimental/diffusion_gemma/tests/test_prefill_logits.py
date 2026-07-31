# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for skipping the prefill lm_head whose logits DG discards.

Host-only: ``discard_prefill_logits`` sets one attribute the shared backbone
already honours, so the contract worth pinning is the flag lifecycle (set,
restore, restore-on-raise, no-op when disabled) plus the fact that the two DG
prefill call sites actually enter it. The device consequence -- no
``create_global_semaphore`` on the prefill path, so no command-queue drain -- is
covered by the prefill-ramp harness, not here.
"""

from __future__ import annotations

import inspect

from models.experimental.diffusion_gemma.tt import chunked_prefill, generate
from models.experimental.diffusion_gemma.tt.prefill_logits import discard_prefill_logits


class _Model:
    """Stand-in for Gemma4Model: only the flag matters."""


class _ClassFlagModel:
    _prefill_trace_mode = False


def test_sets_flag_inside_and_removes_it_after():
    model = _Model()
    assert not hasattr(model, "_prefill_trace_mode")
    with discard_prefill_logits(model):
        assert model._prefill_trace_mode is True
    assert not hasattr(model, "_prefill_trace_mode")


def test_restores_a_preexisting_value():
    model = _Model()
    model._prefill_trace_mode = False
    with discard_prefill_logits(model):
        assert model._prefill_trace_mode is True
    assert model._prefill_trace_mode is False


def test_restores_a_preexisting_true_value():
    # gemma4's traced-prefill generator sets the same flag; nesting must not
    # clear it on the way out.
    model = _Model()
    model._prefill_trace_mode = True
    with discard_prefill_logits(model):
        assert model._prefill_trace_mode is True
    assert model._prefill_trace_mode is True


def test_restores_on_exception(expect_error):
    model = _Model()
    with expect_error(RuntimeError, "prefill blew up"):
        with discard_prefill_logits(model):
            assert model._prefill_trace_mode is True
            raise RuntimeError("prefill blew up")
    assert not hasattr(model, "_prefill_trace_mode")


def test_class_level_flag_is_not_left_shadowed_as_true():
    # delattr on an instance that never had its own copy raises; the fallback
    # must not leave the instance pinned to True.
    model = _ClassFlagModel()
    with discard_prefill_logits(model):
        assert model._prefill_trace_mode is True
    assert model._prefill_trace_mode is False


def test_disabled_is_a_passthrough():
    model = _Model()
    with discard_prefill_logits(model, enabled=False):
        assert not hasattr(model, "_prefill_trace_mode")
    assert not hasattr(model, "_prefill_trace_mode")


def test_prefill_prompt_tokens_discards_logits():
    src = inspect.getsource(generate.prefill_prompt_tokens)
    assert "discard_prefill_logits(tt_model)" in src, (
        "DG prefill must not end in the shared lm_head: its all-gather calls "
        "create_global_semaphore, which drains the command queue once per prefill"
    )


def test_chunked_prefill_keeps_logits_only_for_the_final_chunk():
    src = inspect.getsource(chunked_prefill)
    assert "discard_prefill_logits(model, enabled=not want_logits)" in src, (
        "non-final chunks discard their output, so they must skip the lm_head; "
        "the final chunk's logits are returned and must not be skipped"
    )


class _GatherSpyModel:
    mesh_device = object()
    hidden_size = 16
    embedding_weight = "w"
    embed_scale = 2.0

    class mesh_config:  # noqa: N801 - stands in for the model's mesh_config object
        tp = 4
        tp_axis = 1

    ccl_manager = "ccl-manager"


def test_embed_routes_the_tp_gather_through_dg_not_the_shared_allgather(monkeypatch):
    """The embed all-gather is the other per-prefill plain ``ttnn.all_gather``.

    ``embed_host_tokens`` is called by prefill AND by every block commit, so if this
    reverts to ``Gemma4Model.embed_tokens`` both paths go back to the semaphore-creating
    factory. Verified bit-identical on QB2 (max_abs 0 at [1,1,32,262144] and
    [1,1,8192,10240]); this test only pins the routing.
    """
    seen = {}

    class _T:
        def __init__(self, name):
            self.name = name

        def deallocate(self, force):
            seen[f"dealloc_{self.name}"] = force

    class _FakeTtnn:
        bfloat16 = "bfloat16"

        @staticmethod
        def embedding(tokens, weight, dtype=None):
            return _T("embeds")

        @staticmethod
        def mul(value, scale):
            return _T("scaled")

        @staticmethod
        def unsqueeze_to_4D(value):
            seen["unsqueezed"] = value.name
            return _T("4d")

        @staticmethod
        def all_gather(*args, **kwargs):  # pragma: no cover - must never be reached
            raise AssertionError("DG must not call the shared plain ttnn.all_gather")

    import models.experimental.diffusion_gemma.tt.ccl as dg_ccl

    monkeypatch.setattr(generate, "ttnn", _FakeTtnn)
    monkeypatch.setattr(dg_ccl, "ccl_allgather", lambda t, cfg, mgr, **kw: seen.setdefault("dg_gather", (t.name, mgr)))

    out = generate._embed_tokens_dg(_GatherSpyModel(), _T("tokens"))

    assert seen["unsqueezed"] == "scaled"
    assert seen["dg_gather"] == ("4d", "ccl-manager")
    assert out == seen["dg_gather"]

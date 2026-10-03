# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The base factory hands an adapter deployment to the Turbo pipeline. Host-only: the Turbo factory is
replaced by a recorder, so nothing opens a mesh. What this pins is the forwarded call itself, since
a stray name in it reaches `MiniMaxH3Pipeline.__init__` as an unexpected keyword."""

from __future__ import annotations

import inspect

from ....pipelines.minimax_h3 import pipeline_minimax_h3 as base
from ....pipelines.minimax_h3 import pipeline_minimax_h3_turbo as turbo
from ....pipelines.minimax_h3.weights_minimax_h3 import LORA_PATH_ENV


def test_env_adapter_dispatches_to_the_turbo_factory_with_the_base_arguments(monkeypatch):
    monkeypatch.setenv(LORA_PATH_ENV, "/adapters/turbo.safetensors")
    seen = {}

    def recorder(cls, **kwargs):
        seen.update(kwargs)
        return "turbo"

    monkeypatch.setattr(turbo.MiniMaxH3TurboPipeline, "create_pipeline", classmethod(recorder))

    assert (
        base.MiniMaxH3Pipeline.create_pipeline(mesh_device="mesh", weights_dir="/w", task="fl2va", lora_strength=0.5)
        == "turbo"
    )

    base_params = set(inspect.signature(base.MiniMaxH3Pipeline.create_pipeline).parameters) - {"subclass_kwargs"}
    assert set(seen) == base_params | {"lora_strength"}
    assert (seen["mesh_device"], seen["weights_dir"], seen["task"], seen["lora_strength"]) == (
        "mesh",
        "/w",
        "fl2va",
        0.5,
    )


def test_without_an_adapter_the_base_factory_is_not_redirected(monkeypatch):
    monkeypatch.delenv(LORA_PATH_ENV, raising=False)
    monkeypatch.setattr(
        turbo.MiniMaxH3TurboPipeline,
        "create_pipeline",
        classmethod(lambda cls, **kw: (_ for _ in ()).throw(AssertionError)),
    )
    # No mesh here, so the base path fails on the weights directory -- after the dispatch decision.
    monkeypatch.setattr(base, "resolve_weights_dir", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("weights")))
    try:
        base.MiniMaxH3Pipeline.create_pipeline(mesh_device="mesh", weights_dir="/w")
    except RuntimeError as err:
        assert str(err) == "weights"
    else:
        raise AssertionError("expected the base path to run")

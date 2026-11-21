# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A FLUX.2 pipeline that traces and offloads weights is called like any other. Needs no device."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ...pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from ...pipelines.pipeline_api import Pipeline


class _Resolved(Exception):
    """Stops construction once the base has the resolved values."""


def _handed_to_base(monkeypatch: pytest.MonkeyPatch, construct: Callable[[], object]) -> tuple[bool, bool]:
    def init(self: Pipeline, *, offload_weights: bool, trace: bool) -> None:
        raise _Resolved(offload_weights, trace)

    monkeypatch.setattr(Pipeline, "__init__", init)
    try:
        construct()
    except _Resolved as resolved:
        return resolved.args
    msg = "construction did not reach the base"
    raise AssertionError(msg)


def test_construction_hands_the_flags_to_the_base(monkeypatch: pytest.MonkeyPatch) -> None:
    def construct() -> Flux2Pipeline:
        return Flux2Pipeline(
            mesh_device=None,
            parallel_config=None,
            encoder_parallel_config=None,
            vae_parallel=None,
            topology=None,
            num_links=1,
            offload_weights=True,
            trace=True,
        )

    assert _handed_to_base(monkeypatch, construct) == (True, True)


def test_a_call_reaches_run_on_an_offloading_traced_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Flux2Pipeline, "_run", lambda self, **kwargs: kwargs)
    pipeline = object.__new__(Flux2Pipeline)
    pipeline.offload_weights = True
    pipeline.trace = True
    assert pipeline(prompts=["a"], num_inference_steps=1) == {"prompts": ["a"], "num_inference_steps": 1}


def test_calls_go_through_the_base() -> None:
    assert Flux2Pipeline.__call__ is Pipeline.__call__

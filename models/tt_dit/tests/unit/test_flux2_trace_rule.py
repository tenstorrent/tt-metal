# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 refuses to trace while offloading weights, before it loads anything. Needs no device."""

from __future__ import annotations

import pytest

from ...pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from ...pipelines.pipeline_api import Pipeline

_RULE = "^TDM-130: trace=True is unsupported with offload_weights=True"


def test_construction_raises_before_any_device_use() -> None:
    with pytest.raises(ValueError, match=_RULE):
        Flux2Pipeline(
            mesh_device=None,
            parallel_config=None,
            encoder_parallel_config=None,
            vae_parallel=None,
            topology=None,
            num_links=1,
            offload_weights=True,
            trace=True,
        )


def test_call_raises_on_an_offloading_traced_pipeline() -> None:
    pipeline = object.__new__(Flux2Pipeline)
    pipeline.offload_weights = True
    pipeline.trace = True
    with pytest.raises(ValueError, match=_RULE):
        pipeline(prompts=["a"], num_inference_steps=1)


def test_calls_go_through_the_base() -> None:
    assert Flux2Pipeline.__call__ is Pipeline.__call__
    with pytest.raises(TypeError, match="takes no `traced` argument"):
        object.__new__(Flux2Pipeline)(prompts=["a"], num_inference_steps=1, traced=True)

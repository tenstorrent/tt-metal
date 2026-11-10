# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The LTX pipelines refuse to trace while offloading weights, before they build anything. Needs no device."""

from __future__ import annotations

import pytest

from ...encoders.gemma.encoder_pair import GemmaTokenizerEncoderPair
from ...pipelines.ltx.pipeline_ltx import LTXPipeline
from ...pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from ...pipelines.ltx.pipeline_ltx_one_stage import LTXOneStagePipeline
from ...pipelines.ltx.pipeline_ltx_two_stages import LTXTwoStagesPipeline
from ...pipelines.pipeline_api import Pipeline

_RULE = "^TDM-130: trace=True is unsupported with offload_weights=True"
_PIPELINES = [LTXPipeline, LTXDistilledPipeline, LTXOneStagePipeline, LTXTwoStagesPipeline]


@pytest.mark.parametrize("pipeline_class", _PIPELINES)
def test_construction_raises_before_any_device_use(pipeline_class: type[LTXPipeline]) -> None:
    with pytest.raises(ValueError, match=_RULE):
        pipeline_class(None, None, None, offload_weights=True, trace=True)


@pytest.mark.parametrize("pipeline_class", _PIPELINES)
def test_generate_goes_through_the_base(pipeline_class: type[LTXPipeline]) -> None:
    assert pipeline_class.__call__ is Pipeline.__call__
    pipeline = object.__new__(pipeline_class)
    pipeline.offload_weights = True
    pipeline.trace = True
    with pytest.raises(ValueError, match=_RULE):
        pipeline.generate("a prompt")


@pytest.mark.parametrize("pipeline_class", _PIPELINES)
def test_has_no_per_call_eager_run(pipeline_class: type[LTXPipeline]) -> None:
    with pytest.raises(NotImplementedError, match="no per-call eager run"):
        object.__new__(pipeline_class).run_untraced("a prompt")


class _Module:
    def __init__(self) -> None:
        self.exclusions: list[_Module] = []

    def register_coresident_exclusions(self, *modules: _Module) -> None:
        self.exclusions.extend(modules)


def test_encoder_pair_registers_through_its_pipeline() -> None:
    calls: list[tuple[_Module, tuple[_Module, ...]]] = []
    pair = object.__new__(GemmaTokenizerEncoderPair)
    pair.offload_weights = True
    pair._register_exclusions_with = lambda module, *others: calls.append((module, others))
    encoder, dit, vae = _Module(), _Module(), _Module()
    pair.register_coresident_peers([dit, vae])
    pair._register_exclusions(encoder)
    assert calls == [(encoder, (dit, vae)), (dit, (encoder,)), (vae, (encoder,))]
    assert encoder.exclusions == []


@pytest.mark.parametrize(
    ("trace", "offload_weights", "encoder_trace"),
    [(True, False, True), (False, False, False), (False, True, False), (True, True, False)],
)
def test_encoder_pair_traces_only_with_a_traced_resident_pipeline(
    trace: bool, offload_weights: bool, encoder_trace: bool
) -> None:
    pair = GemmaTokenizerEncoderPair(
        None, mesh_device=None, ccl_manager=None, parallel_config=None, trace=trace, offload_weights=offload_weights
    )
    assert pair._encoder_trace is encoder_trace

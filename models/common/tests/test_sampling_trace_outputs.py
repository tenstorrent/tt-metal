# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import ttnn
from models.common.sampling import generator as sampling_module


@pytest.fixture
def make_sampler(monkeypatch):
    monkeypatch.setattr(sampling_module, "TTPenalties", Mock())
    monkeypatch.setattr(sampling_module, "SeedManager", Mock())
    monkeypatch.setattr(ttnn, "Shape", tuple)
    allocate = Mock(side_effect=lambda shape, *args: SimpleNamespace(shape=shape))
    monkeypatch.setattr(ttnn, "allocate_tensor_on_device", allocate)
    for name in ("begin_trace_capture", "end_trace_capture", "synchronize_device", "release_trace"):
        monkeypatch.setattr(ttnn, name, Mock())

    def create(allow_argmax):
        state = SimpleNamespace(
            max_batch_size=32, _sampling_dp=1, _allow_force_argmax_sampling=allow_argmax, force_argmax_sampling=False
        )
        monkeypatch.setattr(sampling_module, "TTSampling", Mock(return_value=state))
        sampler = sampling_module.SamplingGenerator(args=SimpleNamespace(), mesh_device=object(), tt_ccl=None)
        sampler._run_sampling = Mock(side_effect=lambda *args, tt_out_tok, **kwargs: (tt_out_tok, None))
        return sampler

    return create


@pytest.mark.parametrize("allow_argmax", [False, True])
def test_sampling_outputs_allocated_once_and_reused_through_capture_and_reset(make_sampler, allow_argmax):
    sampler = make_sampler(allow_argmax)
    outputs = dict(sampler._trace_token_outputs)
    assert outputs[False].shape == (1, 1, 1, 32)
    assert len(outputs) == 1 + int(allow_argmax)
    if allow_argmax:
        assert outputs[True].shape == (1, 1, 32)
    allocations = ttnn.allocate_tensor_on_device.call_count
    logits = SimpleNamespace(shape=(1, 1, 32, 128))
    for _ in range(2):
        for mode in outputs:
            sampler.tt_sampling.force_argmax_sampling = mode
            sampler.precompile(logits)
            assert sampler._run_sampling.call_args.kwargs["tt_out_tok"] is outputs[mode]
            result = sampler.capture_trace(logits, skip_precompile=True)
            assert result[0] is outputs[mode]
            assert sampler._run_sampling.call_args.kwargs["tt_out_tok"] is outputs[mode]
        sampler.reset_trace()
        assert not sampler._trace_states
    assert ttnn.allocate_tensor_on_device.call_count == allocations
    assert all(sampler._trace_token_outputs[mode] is output for mode, output in outputs.items())


@pytest.mark.parametrize("method", ["precompile", "capture_trace"])
def test_unsupported_argmax_has_clear_error_before_capture(make_sampler, method, expect_error):
    sampler = make_sampler(False)
    sampler.tt_sampling.force_argmax_sampling = True
    with expect_error(ValueError, "force_argmax_sampling=True was not enabled"):
        getattr(sampler, method)(SimpleNamespace(shape=(1, 1, 32, 128)))
    sampler._run_sampling.assert_not_called()
    ttnn.begin_trace_capture.assert_not_called()
    assert ttnn.allocate_tensor_on_device.call_count == 1


@pytest.mark.parametrize("shape", [(1, 1, 1, 128), (1, 32, 128), (2, 1, 32, 128), (1, 1, 32, 128, 1)])
def test_sampling_trace_rejects_unpadded_or_wrong_rank_logits(make_sampler, shape, expect_error):
    sampler = make_sampler(True)
    with expect_error(ValueError, "Pad logits to the configured sampling batch"):
        sampler.precompile(SimpleNamespace(shape=shape))
    sampler._run_sampling.assert_not_called()
    ttnn.begin_trace_capture.assert_not_called()
    assert ttnn.allocate_tensor_on_device.call_count == 2


def test_explicit_output_remains_owned_by_caller(make_sampler):
    sampler = make_sampler(True)
    # An explicit output can serve another supported model-specific layout.
    logits = SimpleNamespace(shape=(1, 32, 128))
    output = object()
    sampler.precompile(logits, tt_out_tok=output)
    assert sampler._run_sampling.call_args.kwargs["tt_out_tok"] is output
    assert sampler.capture_trace(logits, tt_out_tok=output, skip_precompile=True)[0] is output
    assert ttnn.allocate_tensor_on_device.call_count == 2

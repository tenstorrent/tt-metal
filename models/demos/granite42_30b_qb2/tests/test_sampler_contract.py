# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for Granite's common-sampler integration boundaries."""

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def generator_method(name, **namespace):
    source = Path(__file__).parents[1] / "tt/generator.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "GraniteGenerator"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {"torch": torch, **namespace}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


def test_parameter_refresh_preserves_inverse_temperature_and_seed_contract():
    sampler = Mock()
    state = SimpleNamespace(seeds=object())
    gen = SimpleNamespace(
        samplers={False: sampler, True: Mock()}, states={1: state}, refresh=Mock(), counters=Counter()
    )
    method = generator_method("set_sampling_params")
    method(gen, top_k=[1, 32] * 16, top_p=0.9, temperature=2.0, seed=42)
    k, p, inverse_temp = sampler.reset_params.call_args.args
    assert k.tolist() == [1, 32] * 16
    assert p.shape == (32,)
    torch.testing.assert_close(inverse_temp, torch.full((32,), 0.5, dtype=torch.bfloat16))
    assert gen.refresh.call_args.args[0] is state.seeds
    assert gen.refresh.call_args.args[1].tolist() == [43] * 32
    gen.samplers[True].reset_params.assert_not_called()


@pytest.mark.parametrize("batch", [1, 8, 16])
@pytest.mark.parametrize("argmax", [False, True])
def test_sample_step_uses_device_seeds_and_copies_only_live_rows(batch, argmax):
    tt = Mock()
    tt.uint32 = object()
    state = SimpleNamespace(
        batch=batch,
        seeds=object(),
        logits=SimpleNamespace(shape=[1, 1, batch, 49152]),
        tokens=object(),
        history=object(),
        history_cursor=object(),
    )
    gen = SimpleNamespace(samplers={False: Mock(), True: Mock()}, sampling_output=object())
    append = Mock()
    generator_method("_sample_step", ttnn=tt, append_tokens=append)(gen, state, argmax)
    tt.typecast.assert_called_once_with(state.seeds, tt.uint32)
    assert gen.samplers[argmax].seeds_tt_tensor is tt.typecast.return_value
    gen.samplers[argmax].assert_called_once_with(tt.pad.return_value, tt_out_tok=gen.sampling_output)
    gen.samplers[not argmax].assert_not_called()
    tt.pad.assert_called_once_with(state.logits, [(0, 0), (0, 0), (0, 32 - batch), (0, 0)], value=0)
    tt.slice.assert_called_once_with(gen.sampling_output, [0, 0, 0, 0], [1, 1, 1, batch])
    tt.copy.assert_called_once_with(tt.slice.return_value, state.tokens)
    tt.plus_one.assert_called_once_with(state.seeds, skip_negative_entries=True)
    append.assert_called_once_with(state.tokens, state.history, state.history_cursor)
    tt.from_torch.assert_not_called()
    tt.to_torch.assert_not_called()

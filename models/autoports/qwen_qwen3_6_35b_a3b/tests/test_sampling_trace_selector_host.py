# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of canonical sampling trace selection; no TTNN import or devices."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def owner():
    source = Path(__file__).resolve().parents[1] / "tt/generator.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "QwenGenerator"
    )
    cls.bases = []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in ("sampling_trace_key", "replay_sampling")
    ]
    calls = []
    namespace = dict(ttnn=SimpleNamespace(execute_trace=lambda *args, **kwargs: calls.append((args, kwargs))))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    result = namespace["QwenGenerator"]()
    result.mesh = object()
    result.traces = {"sampling": object(), "sampling_64": object()}
    result.trace_replays = {"model_fixed": 7}
    result.sampling_trace_replays = {}
    result.counters = {"sampling_replays": 0}
    result.active_slots = (0,)
    result.sampling_values = dict(temperature=[1.0] * 32, top_k=[20] * 32)
    result.sampler = SimpleNamespace(tt_sampling=SimpleNamespace(_sampling_draw_width=64))
    return result, calls


@pytest.mark.parametrize(
    "active,wide_slot,temperature,top_k,expected",
    [
        ((0,), 0, 1.0, 32, "sampling"),
        ((0,), 0, 0.7, 40, "sampling_64"),
        ((0,), 0, 0.0, 64, "sampling"),
        ((0,), 17, 0.7, 40, "sampling_64"),
        ((0,), 31, 0.7, 64, "sampling_64"),
        ((0,), 31, 0.0, 64, "sampling"),
        ((0, 17), 17, 0.7, 40, "sampling_64"),
        ((17,), 17, 0.7, 64, "sampling_64"),
        ((), 0, 0.7, 40, "sampling_64"),
    ],
)
def test_all_retained_physical_parameters_select_native_width(active, wide_slot, temperature, top_k, expected):
    gen, _ = owner()
    gen.active_slots = active
    gen.sampling_values["temperature"][wide_slot] = temperature
    gen.sampling_values["top_k"][wide_slot] = top_k
    assert gen.sampling_trace_key() == expected


def test_width_switches_reuse_existing_traces_and_leave_model_accounting_untouched():
    gen, calls = owner()
    traces = dict(gen.traces)
    sampler = gen.sampler
    for k in (20, 40, 20):
        gen.sampling_values["top_k"][0] = k
        gen.replay_sampling()
    assert [args[1] for args, _ in calls] == [traces["sampling"], traces["sampling_64"], traces["sampling"]]
    assert all(kwargs == dict(cq_id=0, blocking=False) for _, kwargs in calls)
    assert gen.traces == traces and gen.sampler is sampler
    assert gen.sampling_trace_replays == {"sampling": 2, "sampling_64": 1}
    assert gen.counters == {"sampling_replays": 3}
    assert gen.trace_replays == {"model_fixed": 7}
    assert gen.sampler.tt_sampling._sampling_draw_width == 32


def test_missing_wide_trace_fails_before_submission():
    gen, calls = owner()
    gen.sampling_values["top_k"][0] = 40
    del gen.traces["sampling_64"]
    with pytest.raises(RuntimeError, match="sampling_64"):  # allow-pytest.raises: CPU only
        gen.replay_sampling()
    assert calls == [] and gen.counters["sampling_replays"] == 0


def test_dormant_wide_owner_keeps_safe_trace_until_its_parameters_are_replaced():
    gen, calls = owner()
    gen.sampling_values["top_k"][31] = 40
    gen.active_slots = (0, 31)
    gen.replay_sampling()
    gen.active_slots = (0,)
    gen.replay_sampling()
    assert gen.sampling_values["top_k"][31] == 40
    assert gen.sampler.tt_sampling._sampling_draw_width == 64
    # Model the host metadata after an authoritative update of that owner;
    # selecting a trace must never alter it or clamp a dormant request itself.
    gen.sampling_values["top_k"][31] = 20
    gen.replay_sampling()
    assert [args[1] for args, _ in calls] == [
        gen.traces["sampling_64"],
        gen.traces["sampling_64"],
        gen.traces["sampling"],
    ]
    assert gen.sampling_trace_replays == {"sampling_64": 2, "sampling": 1}

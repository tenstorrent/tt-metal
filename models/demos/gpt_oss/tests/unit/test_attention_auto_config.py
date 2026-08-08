# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.demos.gpt_oss.tt.attention import operations


class FakeTensor:
    def __init__(self, name: str):
        self.name = name
        self.deallocate_calls = []

    def deallocate(self, force: bool) -> None:
        self.deallocate_calls.append(force)


def test_apply_output_projection_reports_reduce_scatter_winner(monkeypatch):
    input_tensor = FakeTensor("input")
    cast_tensor = FakeTensor("cast")
    weights = SimpleNamespace(o_proj="weight", o_proj_bias="bias")
    linear_calls = []

    monkeypatch.setattr(operations.ttnn, "typecast", lambda tensor, dtype: cast_tensor)
    monkeypatch.setattr(
        operations.ttnn.experimental.auto_config,
        "explain_matmul",
        lambda *args, **kwargs: {"winner": {"kind": "linear_then_reduce_scatter"}},
    )

    def fake_linear(lhs, rhs, **kwargs):
        linear_calls.append((lhs, rhs, kwargs))
        return "linear-out"

    monkeypatch.setattr(operations.ttnn, "linear", fake_linear)

    out, used_reduce_scatter = operations.apply_output_projection(input_tensor, weights, activation_dtype="bf16")

    assert out == "linear-out"
    assert used_reduce_scatter is True
    assert linear_calls == [
        (cast_tensor, weights.o_proj, {"bias": weights.o_proj_bias, "dtype": "bf16", "auto_config": True})
    ]
    assert cast_tensor.deallocate_calls == [True]


def test_apply_output_projection_reports_non_reduce_scatter_winner(monkeypatch):
    input_tensor = FakeTensor("input")
    cast_tensor = FakeTensor("cast")
    weights = SimpleNamespace(o_proj="weight", o_proj_bias="bias")

    monkeypatch.setattr(operations.ttnn, "typecast", lambda tensor, dtype: cast_tensor)
    monkeypatch.setattr(
        operations.ttnn.experimental.auto_config,
        "explain_matmul",
        lambda *args, **kwargs: {"winner": {"kind": "default_linear"}},
    )
    monkeypatch.setattr(operations.ttnn, "linear", lambda *args, **kwargs: "linear-out")

    out, used_reduce_scatter = operations.apply_output_projection(input_tensor, weights, activation_dtype="bf16")

    assert out == "linear-out"
    assert used_reduce_scatter is False
    assert cast_tensor.deallocate_calls == [True]

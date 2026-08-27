# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only lifetime tests for the sequential gated-delta schedule."""


def test_masked_gram_releases_kk_before_allocating_sum(monkeypatch):
    from models.experimental.gated_attention_gated_deltanet.tt import ttnn_delta_rule_seq as seq

    events = []
    scaled = object()
    result = object()

    class FakeTTNN:
        @staticmethod
        def multiply(lhs, rhs, memory_config=None):
            events.append(("multiply", lhs, rhs, memory_config))
            return scaled

        @staticmethod
        def deallocate(tensor):
            events.append(("deallocate", tensor))

        @staticmethod
        def add(lhs, rhs, memory_config=None):
            events.append(("add", lhs, rhs, memory_config))
            return result

    monkeypatch.setattr(seq, "ttnn", FakeTTNN)
    kk, mask, eye, memory_config = object(), object(), object(), object()

    actual = seq._masked_gram_plus_identity(
        kk,
        mask,
        eye,
        memory_config=memory_config,
    )

    assert actual is result
    assert events == [
        ("multiply", kk, mask, memory_config),
        ("deallocate", kk),
        ("add", eye, scaled, memory_config),
        ("deallocate", scaled),
    ]


def test_normalize_chunk_matrix_reuses_full_matrix_storage(monkeypatch):
    from models.experimental.gated_attention_gated_deltanet.tt import ttnn_delta_rule_seq as seq

    events = []
    diagonal, diagonal_sum, inverse, inverse_row = object(), object(), object(), object()

    class FakeTTNN:
        @staticmethod
        def multiply(lhs, rhs, memory_config=None, output_tensor=None):
            events.append(("multiply", lhs, rhs, memory_config, output_tensor))
            return output_tensor if output_tensor is not None else diagonal

        @staticmethod
        def sum(tensor, dim=None, memory_config=None):
            events.append(("sum", tensor, dim, memory_config))
            return diagonal_sum

        @staticmethod
        def reciprocal(tensor, memory_config=None):
            events.append(("reciprocal", tensor, memory_config))
            return inverse

        @staticmethod
        def deallocate(tensor):
            events.append(("deallocate", tensor))

        @staticmethod
        def reshape(tensor, shape, memory_config=None):
            events.append(("reshape", tensor, shape, memory_config))
            return inverse_row

        @staticmethod
        def subtract(lhs, rhs, memory_config=None, output_tensor=None):
            events.append(("subtract", lhs, rhs, memory_config, output_tensor))
            return output_tensor

        @staticmethod
        def add(lhs, rhs, memory_config=None, output_tensor=None):
            events.append(("add", lhs, rhs, memory_config, output_tensor))
            return output_tensor

    monkeypatch.setattr(seq, "ttnn", FakeTTNN)
    monkeypatch.setattr(seq, "_ck", lambda *_: None)
    matrix, eye, memory_config = object(), object(), object()

    unit, row = seq._normalize_chunk_matrix(
        matrix,
        eye,
        batch=7,
        chunk_size=128,
        memory_config=memory_config,
    )

    assert unit is matrix
    assert row is inverse_row
    assert events == [
        ("multiply", matrix, eye, memory_config, None),
        ("sum", diagonal, -1, memory_config),
        ("reciprocal", diagonal_sum, memory_config),
        ("deallocate", diagonal_sum),
        ("reshape", inverse, [7, 128, 1], memory_config),
        ("subtract", matrix, diagonal, memory_config, matrix),
        ("deallocate", diagonal),
        ("multiply", inverse_row, matrix, memory_config, matrix),
        ("add", eye, matrix, memory_config, matrix),
    ]


def test_multiply_into_dead_rhs_uses_rhs_as_output(monkeypatch):
    from models.experimental.gated_attention_gated_deltanet.tt import ttnn_delta_rule_seq as seq

    calls = []

    class FakeTTNN:
        @staticmethod
        def multiply(lhs, rhs, memory_config=None, output_tensor=None):
            calls.append((lhs, rhs, memory_config, output_tensor))
            return output_tensor

    monkeypatch.setattr(seq, "ttnn", FakeTTNN)
    lhs, rhs, memory_config = object(), object(), object()

    result = seq._multiply_into_dead_rhs(lhs, rhs, memory_config=memory_config)

    assert result is rhs
    assert calls == [(lhs, rhs, memory_config, rhs)]

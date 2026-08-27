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

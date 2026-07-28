# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.experimental.diffusion_gemma.tt import expert_operations


def test_diffusion_gemma_gelu_uses_the_checkpoint_tanh_variant(monkeypatch):
    calls = []
    monkeypatch.setattr(
        expert_operations.ttnn,
        "gelu",
        lambda value, **kwargs: calls.append((value, kwargs)) or "activated",
    )

    assert expert_operations.apply_gelu("gate") == "activated"
    assert calls == [("gate", {"variant": expert_operations.ttnn.GeluVariant.Tanh})]


def test_dense_expert_dispatch_is_context_local_and_resets(monkeypatch):
    monkeypatch.setattr(expert_operations, "_original_decode_geglu", lambda gate, up: ("legacy", gate, up))
    monkeypatch.setattr(expert_operations, "apply_geglu", lambda gate, up: ("tanh", gate, up))

    assert expert_operations._contextual_geglu("g", "u") == ("legacy", "g", "u")
    with expert_operations.use_tanh_expert_activations(True):
        assert expert_operations._contextual_geglu("g", "u") == ("tanh", "g", "u")
    assert expert_operations._contextual_geglu("g", "u") == ("legacy", "g", "u")

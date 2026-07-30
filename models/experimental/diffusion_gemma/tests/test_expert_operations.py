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


def test_legacy_geglu_releases_the_activation_without_editing_shared_gemma4(monkeypatch):
    """The no-context fallback must free its own temporary.

    DiffusionGemma used to get this free by adding ``activated.deallocate(True)`` to
    ``models/demos/gemma4/tt/experts/operations.py``. That shared edit was reverted on
    2026-07-30; this test is what keeps the release from going with it.
    """
    released = []

    class _Activated:
        def deallocate(self, force=False):
            released.append(force)

    activated = _Activated()
    monkeypatch.setattr(expert_operations.ttnn, "gelu", lambda value, **kwargs: activated)
    monkeypatch.setattr(expert_operations.ttnn, "mul", lambda a, b: "down_input")

    assert expert_operations._legacy_geglu_with_release("gate", "up") == "down_input"
    assert released == [True], "the fallback leaked the GELU activation"


def test_dense_expert_dispatch_is_context_local_and_resets(monkeypatch):
    monkeypatch.setattr(expert_operations, "_legacy_geglu_with_release", lambda gate, up: ("legacy", gate, up))
    monkeypatch.setattr(expert_operations, "apply_geglu", lambda gate, up: ("tanh", gate, up))

    assert expert_operations._contextual_geglu("g", "u") == ("legacy", "g", "u")
    with expert_operations.use_tanh_expert_activations(True):
        assert expert_operations._contextual_geglu("g", "u") == ("tanh", "g", "u")
    assert expert_operations._contextual_geglu("g", "u") == ("legacy", "g", "u")

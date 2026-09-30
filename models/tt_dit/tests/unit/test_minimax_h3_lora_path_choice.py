# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for WHICH of the two LoRA merge paths a MiniMax-H3 pipeline takes.

This matters more than it looks. The two paths are not equivalent implementations of one operation:
on the 4-chip mesh the on-device bind was measured to be a complete no-op -- the served config with
no adapter produced bit-identical video and audio to the served config with one -- so the choice
decides whether a distillation adapter is applied at all. The DiT dtype and the merge path are also
collinear in the shipped rule (a quant profile implies the host fuse), which is why the override
exists and why it is pinned here: without it no experiment can separate a dtype defect from an
adapter defect, and it would be easy to "simplify" the env lookup away.
"""
from types import SimpleNamespace

import pytest

from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

FORCE = "MINIMAX_H3_FORCE_HOST_FUSE"


def _fuses_on_host(lora_path, dit_quant_profile):
    """Call the real predicate against the two attributes it reads, with no device and no weights."""
    stub = SimpleNamespace(lora_path=lora_path, dit_quant_profile=dit_quant_profile)
    return MiniMaxH3Pipeline._lora_fuses_on_host(stub)


@pytest.fixture(autouse=True)
def _no_inherited_override(monkeypatch):
    monkeypatch.delenv(FORCE, raising=False)


def test_no_adapter_never_fuses_on_host():
    assert _fuses_on_host(None, None) is False
    assert _fuses_on_host(None, object()) is False


def test_the_override_cannot_invent_an_adapter(monkeypatch):
    """It forces a PATH, not an adapter: with nothing to merge there is still nothing to merge."""
    monkeypatch.setenv(FORCE, "1")
    assert _fuses_on_host(None, None) is False


def test_quantized_weights_fuse_on_host():
    """The shipped rule: bfloat8_b's per-tile-row exponent swallows a rank-128 delta, so the
    adapter has to be merged into the checkpoint before quantization or it does nothing."""
    assert _fuses_on_host("adapter.safetensors", object()) is True


def test_unquantized_weights_bind_on_device_by_default():
    """The default for bf16, and the path proven ineffective on the 1x4 -- so this assertion is the
    record of current behaviour, NOT an endorsement of it. When the bind is repaired this test
    stays; when the DEFAULT is changed, this is the test that should fail and be updated."""
    assert _fuses_on_host("adapter.safetensors", None) is False


def test_the_override_forces_the_host_fuse_on_unquantized_weights(monkeypatch):
    """The whole point of the knob: bf16 with the adapter genuinely applied, so that dtype and
    merge path can be varied independently."""
    monkeypatch.setenv(FORCE, "1")
    assert _fuses_on_host("adapter.safetensors", None) is True


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "2"])
def test_only_the_literal_1_enables_the_override(monkeypatch, value):
    """An experiment knob that triggered on any truthy-looking string would change serving
    behaviour for anyone who exported it loosely."""
    monkeypatch.setenv(FORCE, value)
    assert _fuses_on_host("adapter.safetensors", None) is False


def test_the_override_does_not_disturb_the_quantized_path(monkeypatch):
    monkeypatch.setenv(FORCE, "1")
    assert _fuses_on_host("adapter.safetensors", object()) is True

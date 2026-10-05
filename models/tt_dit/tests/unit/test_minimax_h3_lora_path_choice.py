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


# ---------------------------------------------------------------------------------------------
# Which adapter targets the H3 loader is willing to bind, and what it does with the rest.
#
# `precomputed_adaln: True` -- which the (1, 1) preset serves and which the (1, 4) preset was
# measured at and did NOT take -- evaluates `time_embedder`, every `adaln_proj` and
# `norm_out.linear` into host tables instead of keeping them on device. An adapter targeting a
# module that is not on the device could be dropped without anything saying so: the same shape of
# silent no-op stage 06b existed to fix, where a profile advertises an adapter it is not applying.
#
# Block-level adaLN leaves are not dropped silently: `_parse_target` returns None for any leaf
# outside `_QKV_SUBS`/`_SINGLETONS`, the caller collects those into `unmapped`, and
# `load_h3_adapter_into` RAISES rather than binding a partial adapter. That guard does not depend
# on the adaLN setting -- it refuses the adapter on either preset, which is what makes the setting
# safe to change.
#
# `time_embedder` is NOT covered by that path and is called out here so the gap is not mistaken for
# coverage: it is handled by the `_GLOBALS` branch above `_parse_target`, so with adaLN precomputed
# (where the transformer sets `time_embedder = None`) such an adapter dies on an AttributeError
# rather than through the audited "no H3 destination" message. Loud either way, so not the
# silent-drop hazard -- but a different failure path, and untested.
#
# The published Turbo adapters are unaffected: they carry no adaLN and no `time_embedder` keys at
# all (312 A/B pairs, all attention and ff). That is a property of today's files, not of the
# format, so the guard is asserted rather than assumed.
@pytest.mark.parametrize(
    "base",
    [
        "transformer_blocks.0.adaln_proj",
        "transformer_blocks.49.norm1.linear",
        "token_refiner.refiner_blocks.1.adaln_proj",
        "transformer_blocks.3.attn.to_out.1",
    ],
)
def test_h3_loader_does_not_map_targets_it_cannot_bind(base):
    """A leaf the loader has no destination for must come back unmapped, not silently skipped."""
    from models.tt_dit.experimental.lora.h3_adapter_loader import _parse_target

    assert _parse_target(base) is None


@pytest.mark.parametrize(
    "base",
    [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.49.attn.to_out.0",
        "transformer_blocks.49.ff.net.0.proj",
        "transformer_blocks.49.ff.net.2",
        "token_refiner.refiner_blocks.1.ff.net.2",
    ],
)
def test_h3_loader_maps_every_target_the_turbo_adapters_use(base):
    """The six leaves the published Turbo files target, on both block stacks, stay mappable."""
    from models.tt_dit.experimental.lora.h3_adapter_loader import _parse_target

    assert _parse_target(base) is not None


def test_h3_loader_refuses_a_partially_bindable_adapter(expect_error):
    """`unmapped` must end the load. Binding what matched and dropping the rest would hand back a
    model that silently disagrees with the adapter it is named after -- the 06b defect's shape.

    The raise lands after `promote_to_lora` has walked the model but before a single
    `bind_active`, so no delta is merged into any weight. The model is not untouched, though: the
    loop calls `register_lora`, so on the raise it is left promoted with registered-but-unbound
    banks. That is recoverable state, not a half-applied adapter, and the distinction is the whole
    point of raising here rather than after the bind pass. Promotion is stubbed because walking a
    real transformer needs weights and a device, and neither is what this test is about.
    """
    import torch

    from models.tt_dit.experimental.lora import h3_adapter_loader as loader

    pairs = {"transformer_blocks.0.adaln_proj": {"A": torch.zeros(8, 4), "B": torch.zeros(4, 8)}}
    saved = (loader._collect_pairs, loader._read, loader.promote_to_lora)
    loader._collect_pairs = lambda _raw: (pairs, {})
    loader._read = lambda _path: ({}, {})
    loader.promote_to_lora = lambda _transformer: 0
    try:
        with expect_error(RuntimeError, "no H3 destination"):
            loader.load_h3_adapter_into(object(), "fake.safetensors")
    finally:
        loader._collect_pairs, loader._read, loader.promote_to_lora = saved

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1 row 1: the real checkpoint's dequantization, against a reference dequantization. Host-only.

The recipe's reference is ``gpt_oss_d_p/tests/unit/test_mxfp4_loader.py`` — "dequantized expert
weights against a reference dequantization of the packed blocks + scales". This checkpoint is fp8
with **per-tensor scalar** scales rather than MXFP4 blocks, so the reference dequantization is one
multiply instead of an unpack, and there are no experts; everything else about the row transfers.

The bar is **bit-exact**, not PCC. ``w.to(f32) * scale`` has one rounding, done the same way on
both sides, so any difference at all means the loader is reading a different tensor, a different
scale, or applying the scale in the wrong direction — all of which a PCC threshold would hide
(a scale error of a few percent still PCCs at 1.0, since PCC is scale-invariant; that is exactly
the failure mode this row exists to catch).

Three claims, in decreasing cost:

* :func:`test_dequantized_matches_manual` reads tensors twice and compares — a handful of tensors,
  because each one is up to 705 MB.
* :func:`test_state_dict_view_matches_layer_weights` checks the lazy view P1 loads through against
  the eager :meth:`~...reference.checkpoint.CheckpointLoader.layer_weights` the M1 ground truth
  uses, so the device and the host reference cannot be reading different weights.
* :func:`test_quantization_is_where_the_loader_thinks` and
  :func:`test_prefix_filter_excludes_the_vision_tower` are index-only and cover all 88 layers, so
  the global statements cost nothing.

Skipped rather than failed when the checkpoint is absent: this file is in the host suite, which
must stay runnable on a machine with neither hardware nor ``/mnt/models``.
"""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.checkpoint import (
    QUANTIZED_SUFFIXES,
    CheckpointLoader,
    CheckpointStateDict,
)
from models.demos.mistral_medium_3_5_128b.reference.modeling import REF_DTYPE, LayerWeights

#: Layers whose tensors are actually read. Layer 0 and one from deep in the stack: the shards are
#: laid out in layer order, so this exercises two different safetensors files.
PROBE_LAYERS = (0, 87)


@pytest.fixture(scope="module")
def loader(cfg):
    try:
        return CheckpointLoader.from_env(cfg)
    except FileNotFoundError as e:
        pytest.skip(f"real checkpoint not available: {e}")


def _manual_dequant(loader, name):
    """The reference dequantization: read the packed tensor and its scale, multiply in fp32.

    Independent of :meth:`CheckpointLoader.dequantized` down to the ``raw`` read, which is the
    part under test — it is the scale lookup and the multiply that can be wrong, not safetensors.
    """
    w = loader.raw(name)
    scale_key = f"{name[: -len('.weight')]}.weight_scale_inv"
    if not loader.has(scale_key):
        return None
    return (w.to(torch.float32) * loader.raw(scale_key).to(torch.float32)).to(REF_DTYPE)


@pytest.mark.parametrize("layer_idx", PROBE_LAYERS)
def test_dequantized_matches_manual(loader, layer_idx):
    """Every quantized projection in a layer, dequantized two ways, bit for bit."""
    p = loader.layer_prefix(layer_idx)
    for suffix in QUANTIZED_SUFFIXES:
        name = f"{p}{suffix}.weight"
        expected = _manual_dequant(loader, name)
        assert expected is not None, f"{name} has no weight_scale_inv but is in QUANTIZED_SUFFIXES"
        got = loader.dequantized(name)
        torch.testing.assert_close(got, expected, rtol=0, atol=0, msg=f"{name} dequantized differently")
        del expected, got


def test_the_scale_is_actually_applied(loader):
    """The scale is not 1.0, so the multiply is observable.

    Without this, :func:`test_dequantized_matches_manual` would pass just as happily against a
    loader that ignored ``weight_scale_inv`` entirely — both sides would be reading the raw fp8.
    """
    name = f"{loader.layer_prefix(0)}self_attn.q_proj.weight"
    scale = loader.raw(f"{name[: -len('.weight')]}.weight_scale_inv").to(torch.float32)
    assert scale.numel() == 1, f"expected a per-tensor scalar scale, got shape {tuple(scale.shape)}"
    assert scale.item() != 1.0, "the scale is 1.0, so this file cannot tell a scaled load from an unscaled one"
    raw_max = loader.raw(name).to(torch.float32).abs().max().item()
    got_max = loader.dequantized(name).float().abs().max().item()
    assert got_max == pytest.approx(raw_max * scale.item(), rel=1e-2), (
        f"dequantized magnitude {got_max:g} is not raw {raw_max:g} times the scale {scale.item():g} — "
        f"the scale may be applied in the wrong direction"
    )


def test_unquantized_tensors_pass_through(loader):
    """``embed_tokens``, the norms and ``lm_head`` are stored bf16 and must not be rescaled.

    Multiplying one of these by a scale that is not there is the mirror-image error of failing to
    dequantize one that is, and the loader decides between them from the index alone.
    """
    names = [
        f"{loader.prefix}embed_tokens.weight",
        f"{loader.prefix}norm.weight",
        f"{loader.layer_prefix(0)}input_layernorm.weight",
        f"{loader.layer_prefix(0)}post_attention_layernorm.weight",
        "lm_head.weight",
    ]
    for name in names:
        assert not loader.has(f"{name[: -len('.weight')]}.weight_scale_inv"), f"{name} unexpectedly has a scale"
        raw = loader.raw(name)
        assert raw.dtype == REF_DTYPE, f"{name} is stored {raw.dtype}, expected {REF_DTYPE}"
        torch.testing.assert_close(loader.dequantized(name), raw, rtol=0, atol=0)
        del raw


def test_quantization_is_where_the_loader_thinks(loader, cfg):
    """Over all 88 layers: exactly the :data:`QUANTIZED_SUFFIXES` carry a scale, and nothing else.

    Index-only, so the whole model is covered for the price of a dict scan. ``QUANTIZED_SUFFIXES``
    is a hand-written constant that the loader's assertions lean on; this is what keeps it honest
    against the checkpoint rather than against its author's memory.
    """
    scaled = {k for k in loader.weight_map if k.endswith(".weight_scale_inv")}
    expected = {
        f"{loader.layer_prefix(i)}{s}.weight_scale_inv"
        for i in range(cfg.num_hidden_layers)
        for s in QUANTIZED_SUFFIXES
    }
    in_text_model = {k for k in scaled if k.startswith(loader.prefix)}
    assert in_text_model == expected, (
        f"text-model scale keys differ from QUANTIZED_SUFFIXES x 88 layers: "
        f"{len(in_text_model - expected)} unexpected, {len(expected - in_text_model)} missing"
    )


def test_prefix_filter_excludes_the_vision_tower(loader):
    """The checkpoint carries a ``model.vision_tower.*`` stack this bring-up must not load.

    ``weight_prefix`` is the only thing keeping it out, and it is a config string — so assert both
    that the tower is really there (or this test proves nothing) and that no name the loader builds
    can reach it.
    """
    tower = [k for k in loader.weight_map if k.startswith("model.vision_tower.")]
    assert tower, "no vision_tower tensors in the index; this checkpoint is not what the loader assumes"
    assert not any(k.startswith(loader.prefix) for k in tower), "the language-model prefix overlaps the vision tower"
    assert loader.prefix == "model.language_model.", loader.prefix


@pytest.mark.parametrize("layer_idx", [0])
def test_state_dict_view_matches_layer_weights(loader, layer_idx):
    """The lazy view P1 builds the device model from equals the eager container M1 measures against.

    Two loading paths for the same weights is exactly how a device run and its host reference come
    to disagree for reasons that look like a numerics bug. They share
    :meth:`~...reference.checkpoint.CheckpointLoader.dequantized`; this pins the naming and the
    layer indexing on top of it.
    """
    view = CheckpointStateDict(loader)
    sub = view.substate(f"layers.{layer_idx}")
    eager = loader.layer_weights(layer_idx)

    assert set(sub) == {f"{s}.weight" for s in LayerWeights._SUFFIXES}, sorted(sub)
    for suffix in LayerWeights._SUFFIXES:
        field = suffix.split(".")[-1]
        torch.testing.assert_close(sub[f"{suffix}.weight"], getattr(eager, field), rtol=0, atol=0, msg=suffix)


def test_state_dict_view_tails_and_unknown_keys(loader):
    """The three tail sub-trees resolve, and an unknown key raises instead of loading nothing.

    An empty dict is this package's "load from the tensor cache instead" signal (see
    ``tt/embedding.py``, ``tt/attention/weights.py``), so a mistyped sub-tree that returned ``{}``
    would build a model on uninitialized weights and report a PCC for it.
    """
    view = CheckpointStateDict(loader)
    assert set(view.substate("norm")) == {"weight"}
    assert view.substate("norm")["weight"].shape == (loader.config.hidden_size,)
    with pytest.raises(KeyError):  # allow-pytest.raises: host-side
        view.substate("layer.0")  # singular: the typo that must not silently succeed
    with pytest.raises(KeyError):  # allow-pytest.raises: host-side
        view.substate("mlp")

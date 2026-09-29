# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Torch-side conversion of the MiniMax-H3 audio VAE checkpoint. No ``ttnn`` here.

``tt_dit`` has no weight-norm support anywhere -- ``git grep weight_g`` over
``models/tt_dit`` returns nothing -- because the contract is that weight norm is removed
before a state dict reaches it. The H3 audio checkpoint stores ``weight_g``/``weight_v``
for all ~172 of its convolutions, so that fusion has to happen somewhere, and a single
torch-side conversion boundary is the right place: it is orthogonal to every conv class,
it would otherwise be duplicated across four of them, and it is testable with **zero
device time**.

The fusion rule is uniform because ``torch.nn.utils.weight_norm`` is used at its default
``dim=0`` everywhere::

    weight = weight_g * weight_v / ||weight_v||   (L2 over all axes but axis 0)

The trap is that for ``ConvTranspose1d`` axis 0 is **``in_channels``**, not
``out_channels``, because its weight is stored ``(in, out, k)``. Fusing over the wrong
axis yields a well-formed decoder that sounds subtly wrong, which is the kind of bug that
gets misattributed to precision for days -- hence
:func:`assert_weight_norm_axes_consistent`.

Also folded here, so the device side stays simple:

* ``attn.q_bias`` / ``attn.zero_k_bias`` / ``attn.v_bias`` -> one ``attn.qkv.bias``. The
  reference passes ``cat(q_bias, zero_k_bias, v_bias)`` to a single ``F.linear``; the k
  third is a frozen zero, which is asserted before being discarded.
* ``activations.{2i}`` -> ``acts1.{i}`` and ``activations.{2i+1}`` -> ``acts2.{i}``, which
  is the layout ``vocoder_ltx.AMPBlock1`` already expects.
* ``ups.{i}.0`` -> ``ups.{i}`` and ``activation_post`` -> ``act_post``.
"""

from __future__ import annotations

import re

import torch


def fuse_weight_norm(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """``weight_g * weight_v / ||weight_v||``, L2-reduced over every axis but axis 0.

    Matches ``torch.nn.utils.weight_norm`` at its default ``dim=0`` for both ``Conv1d``
    (axis 0 = out_channels) and ``ConvTranspose1d`` (axis 0 = **in_channels**). The result
    keeps the checkpoint's own layout; transposing is the conv class's job, not ours.
    """
    norm = weight_v.flatten(1).norm(dim=1).view(-1, *([1] * (weight_v.dim() - 1)))
    return weight_g * weight_v / norm


def assert_weight_norm_axes_consistent(state: dict[str, torch.Tensor]) -> None:
    """Guard the axis-0 assumption on every weight-normed conv in the checkpoint.

    For each ``weight_g``/``weight_v`` pair, ``weight_g`` must be a per-axis-0 scalar --
    shape ``(axis0, 1, 1)`` -- and any sibling ``bias`` must be length ``out_channels``.
    For a ``Conv1d`` that is ``weight_v.shape[0]``; for a ``ConvTranspose1d`` it is
    ``weight_v.shape[1]``. A bias that matches neither means the layout is not what this
    converter assumes.
    """
    for key in sorted(k for k in state if k.endswith(".weight_g")):
        prefix = key[: -len(".weight_g")]
        weight_g = state[key]
        weight_v = state[f"{prefix}.weight_v"]
        assert (
            weight_g.shape[0] == weight_v.shape[0]
        ), f"{prefix}: weight_g axis 0 is {weight_g.shape[0]} but weight_v axis 0 is {weight_v.shape[0]}"
        assert all(
            size == 1 for size in weight_g.shape[1:]
        ), f"{prefix}: weight_g {tuple(weight_g.shape)} is not a per-axis-0 scalar, so dim=0 is the wrong axis"
        bias = state.get(f"{prefix}.bias")
        if bias is not None:
            assert bias.shape[0] in (weight_v.shape[0], weight_v.shape[1]), (
                f"{prefix}: bias {bias.shape[0]} matches neither weight_v axis "
                f"({weight_v.shape[0]}, {weight_v.shape[1]})"
            )


def fuse_all_weight_norms(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace every ``weight_g``/``weight_v`` pair with a single fused ``weight``."""
    assert_weight_norm_axes_consistent(state)
    fused = {}
    for key, value in state.items():
        if key.endswith(".weight_v"):
            prefix = key[: -len(".weight_v")]
            fused[f"{prefix}.weight"] = fuse_weight_norm(state[f"{prefix}.weight_g"], value)
        elif key.endswith(".weight_g"):
            continue
        else:
            fused[key] = value
    return fused


def fuse_attention_biases(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold ``q_bias`` / ``zero_k_bias`` / ``v_bias`` into one ``qkv.bias``.

    The reference builds ``cat(q_bias, zero_k_bias, v_bias)`` for a single fused
    projection. ``zero_k_bias`` is a frozen zero buffer; that is asserted rather than
    assumed, so a future checkpoint that unfreezes it fails loudly instead of silently
    losing a bias term.
    """
    out = dict(state)
    prefixes = {k[: -len(".q_bias")] for k in state if k.endswith(".q_bias")}
    for prefix in sorted(prefixes):
        q_bias = out.pop(f"{prefix}.q_bias")
        k_bias = out.pop(f"{prefix}.zero_k_bias", None)
        v_bias = out.pop(f"{prefix}.v_bias")
        if k_bias is None:
            k_bias = torch.zeros_like(q_bias)
        else:
            assert (
                torch.count_nonzero(k_bias) == 0
            ), f"{prefix}.zero_k_bias is not all zero -- it is a real parameter and cannot be dropped"
        out[f"{prefix}.qkv.bias"] = torch.cat([q_bias, k_bias, v_bias], dim=0)
    return out


_ACTIVATION_RE = re.compile(r"^(?P<prefix>.*?)activations\.(?P<index>\d+)\.(?P<rest>.*)$")


def remap_amp_activations(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``activations.{2i}`` -> ``acts1.{i}``, ``activations.{2i+1}`` -> ``acts2.{i}``.

    H3 stores one flat list of six activations per AMP block; ``vocoder_ltx.AMPBlock1``
    holds them as two lists of three, interleaved. Getting this backwards produces a
    plausible-looking decoder, so it is worth doing in one obvious place.
    """
    out = {}
    for key, value in state.items():
        match = _ACTIVATION_RE.match(key)
        if match is None:
            out[key] = value
            continue
        index = int(match["index"])
        target = "acts1" if index % 2 == 0 else "acts2"
        out[f"{match['prefix']}{target}.{index // 2}.{match['rest']}"] = value
    return out


def remap_decoder_names(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``ups.{i}.0`` -> ``ups.{i}``, ``activation_post`` -> ``act_post``."""
    out = {}
    for key, value in state.items():
        key = re.sub(r"(^|\.)ups\.(\d+)\.0\.", r"\1ups.\2.", key)
        key = key.replace("activation_post.", "act_post.")
        out[key] = value
    return out


def convert_minimax_h3_audio_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The whole conversion, in the order the steps depend on each other.

    Weight norm is fused first so the later renames only ever move plain ``weight`` keys.
    """
    converted = fuse_all_weight_norms(state)
    converted = fuse_attention_biases(converted)
    converted = remap_amp_activations(converted)
    return remap_decoder_names(converted)

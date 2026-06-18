# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the LoRA fuser (no device/ttnn needed).

Covers the conventions ``fuse_loras_into`` normalizes: diffusers/PEFT
``lora_A/B`` (with adapter-name infix), kohya ``lora_down/up``, ``.alpha`` rank
scaling, full-weight ``.diff`` / bias ``.diff_b`` deltas, prefix stripping, and
strict-mode keyspace validation.
"""

import os

import pytest
import torch
from safetensors.torch import save_file

from models.tt_dit.utils.fuse_loras import LoraSpec, fuse_loras_into


def _base():
    return {
        "blk.to_q.weight": torch.zeros(8, 8),
        "blk.to_q.bias": torch.zeros(8),
        "blk.skip.weight": torch.ones(4, 4),
    }


def _save(tmp_path, name, sd):
    p = os.path.join(tmp_path, name)
    save_file(sd, p)
    return p


def test_diffusers_ab_with_strength_and_prefix(tmp_path):
    a, b = torch.randn(2, 8), torch.randn(8, 2)
    p = _save(
        tmp_path,
        "ab.safetensors",
        {
            "model.diffusion_model.blk.to_q.lora_A.weight": a,
            "model.diffusion_model.blk.to_q.lora_B.weight": b,
        },
    )
    out = fuse_loras_into(_base(), [LoraSpec(p, 2.0)])
    assert torch.allclose(out["blk.to_q.weight"], (b @ a) * 2.0, atol=1e-5)
    # Untouched key passes through.
    assert torch.allclose(out["blk.skip.weight"], torch.ones(4, 4))


def test_kohya_down_up_with_alpha(tmp_path):
    a, b = torch.randn(2, 8), torch.randn(8, 2)
    p = _save(
        tmp_path,
        "kohya.safetensors",
        {
            "transformer.blk.to_q.lora_down.weight": a,
            "transformer.blk.to_q.lora_up.weight": b,
            "transformer.blk.to_q.alpha": torch.tensor(4.0),
        },
    )
    out = fuse_loras_into(_base(), [LoraSpec(p, 1.0)])
    # scale = strength * alpha / rank = 1 * 4 / 2
    assert torch.allclose(out["blk.to_q.weight"], (b @ a) * 2.0, atol=1e-5)


def test_peft_adapter_infix_and_diff_b(tmp_path):
    a, b = torch.randn(2, 8), torch.randn(8, 2)
    p = _save(
        tmp_path,
        "peft.safetensors",
        {
            "diffusion_model.blk.to_q.lora_A.default.weight": a,
            "diffusion_model.blk.to_q.lora_B.default.weight": b,
            "diffusion_model.blk.to_q.diff_b": torch.full((8,), 3.0),
        },
    )
    out = fuse_loras_into(_base(), [LoraSpec(p, 1.0)])
    assert torch.allclose(out["blk.to_q.weight"], b @ a, atol=1e-5)
    assert torch.allclose(out["blk.to_q.bias"], torch.full((8,), 3.0), atol=1e-5)


def test_full_weight_diff(tmp_path):
    p = _save(tmp_path, "diff.safetensors", {"blk.skip.diff": torch.full((4, 4), 0.5)})
    out = fuse_loras_into(_base(), [LoraSpec(p, 2.0)])
    assert torch.allclose(out["blk.skip.weight"], torch.ones(4, 4) + 1.0, atol=1e-5)


def test_strict_raises_on_zero_fuse(tmp_path):
    p = _save(tmp_path, "miss.safetensors", {"no.such.module.lora_A.weight": torch.randn(2, 8)})
    # Non-strict: warns + passes through unchanged.
    out = fuse_loras_into(_base(), [LoraSpec(p, 1.0)])
    assert torch.allclose(out["blk.to_q.weight"], torch.zeros(8, 8))
    # Strict: keyspace mismatch is an error.
    with pytest.raises(ValueError):
        fuse_loras_into(_base(), [LoraSpec(p, 1.0)], strict=True)


def test_empty_loras_is_passthrough():
    base = _base()
    assert fuse_loras_into(base, []) is base

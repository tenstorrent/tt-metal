# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1: dequantized weights against a reference dequantization of the packed values + scales.
Pattern: ``gpt_oss_d_p/tests/unit/test_mxfp4_loader.py``. Host only, no device.

Mistral ships PER-TENSOR fp8 (``weight_block_size: null``), which no donor in the repo handles — every
fp8 donor is DeepSeek blockwise — so :mod:`...tt.fp8_dequant` is the one piece of the weight path
written from scratch, and this is its test. Four properties, each a way the loader could be silently
wrong rather than loud:

  1. the round trip is accurate to fp8's actual precision, and no better (a test that passed with a
     dropped scale would be worthless, so the tolerance is checked from both sides);
  2. an fp8 tensor with NO matching scale is a hard failure, never a pass-through;
  3. the scale keys (``weight_scale``, ``input_scale``) never reach the model, and ``input_scale`` in
     particular is an ACTIVATION scale — applying it to a weight would scale that weight twice;
  4. ``modules_to_not_convert`` pass through untouched.

Plus the end-to-end path on a real on-disk checkpoint: shards -> dequant -> prefix mapping -> Meta
swizzle, with the swizzle checked against ``reverse_permute`` directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.tt.fp8_dequant import (
    dequantize_state_dict,
    dequantize_weight_tensor,
    is_scale_key,
    is_unquantized_module,
    weight_scale_key,
)

E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def quantize(weight: torch.Tensor):
    """The reference quantization: per-tensor scale = amax / e4m3_max (compressed-tensors / vLLM)."""
    scale = (weight.abs().max().float() / E4M3_MAX).clamp(min=1e-12)
    return (weight.float() / scale).to(torch.float8_e4m3fn), scale.reshape(())


def test_dequantize_matches_a_reference_dequantization():
    """``w_fp8.float() * scale`` against an independently computed dequantization, and the accuracy
    is bounded from BOTH sides so a no-op or a wrong scale cannot pass."""
    torch.manual_seed(0)
    weight = torch.randn(512, 256) * 0.02
    packed, scale = quantize(weight)

    got = dequantize_weight_tensor(packed, scale)
    want = (packed.float() * scale.float()).to(torch.bfloat16)
    assert torch.equal(got.float(), want.float()), "dequantization differs from the reference expression"

    rel = ((got.float() - weight).abs() / weight.abs().clamp(min=1e-8)).median().item()
    logger.info(f"fp8 per-tensor round trip: median relative error {rel:.5f}")
    # e4m3 keeps 3 mantissa bits, so ~2-6% median relative error is the expected band. Above it, a
    # scale is wrong; far BELOW it, the tensor was never quantized and the test proves nothing.
    assert rel < 0.10, f"median relative error {rel:.4f} — the scale is wrong"
    assert rel > 1e-4, f"median relative error {rel:.2e} is too good for fp8 — was the input quantized?"


def test_missing_scale_is_fatal():
    """An fp8 tensor with no ``weight_scale`` must raise. Emitting it unscaled is off by ~1/scale."""
    packed, _ = quantize(torch.randn(64, 32))
    with pytest.raises(ValueError, match="no matching per-tensor scale"):
        dequantize_state_dict({"model.layers.0.mlp.up_proj.weight": packed})


def test_blockwise_scale_is_rejected():
    """A multi-element scale means a BLOCKWISE checkpoint; broadcasting it here would be plausible
    and wrong, so it must raise and name the blockwise path instead."""
    packed, _ = quantize(torch.randn(64, 32))
    with pytest.raises(ValueError, match="BLOCKWISE"):
        dequantize_weight_tensor(packed, torch.ones(4, 2))


def test_scale_keys_never_reach_the_model():
    """Both scale kinds are metadata. ``input_scale`` is an ACTIVATION scale under the static scheme
    and must not be applied to a weight — the device computes activations in bf16."""
    torch.manual_seed(1)
    weight = torch.randn(64, 32) * 0.02
    packed, scale = quantize(weight)
    state = {
        "model.layers.0.self_attn.q_proj.weight": packed,
        "model.layers.0.self_attn.q_proj.weight_scale": scale,
        "model.layers.0.self_attn.q_proj.input_scale": torch.tensor(7.0),  # deliberately not 1.0
        "model.layers.0.input_layernorm.weight": torch.randn(32),
    }
    out = dequantize_state_dict(state)
    assert set(out) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.input_layernorm.weight",
    }, f"unexpected keys survived: {sorted(out)}"

    # If input_scale (7.0) had been applied, the result would be 7x too large.
    rel = (out["model.layers.0.self_attn.q_proj.weight"].float() - weight).abs() / weight.abs().clamp(min=1e-8)
    assert rel.median().item() < 0.10, "input_scale appears to have been applied to the weight"

    assert is_scale_key("x.weight_scale") and is_scale_key("x.input_scale") and is_scale_key("x.weight_scale_inv")
    assert not is_scale_key("x.weight")
    assert weight_scale_key("a.b.weight") == "a.b.weight_scale"


def test_modules_to_not_convert_pass_through():
    """``lm_head`` and the vision side are stored unquantized; they must arrive unchanged (bar the
    dtype cast) even though lm_head is a projection like any other."""
    for name in C.MODULES_TO_NOT_CONVERT:
        assert is_unquantized_module(f"{name}.weight"), f"{name} should be recognised as unquantized"
    assert not is_unquantized_module("model.layers.0.mlp.up_proj.weight")

    weight = torch.randn(16, 8)
    out = dequantize_state_dict({"lm_head.weight": weight})
    assert torch.equal(out["lm_head.weight"].float(), weight.to(torch.bfloat16).float())


def test_integer_tensors_are_copied_not_cast():
    """A non-floating tensor (e.g. an id table) must survive as an integer, not become bf16."""
    ids = torch.arange(8, dtype=torch.int32)
    out = dequantize_state_dict({"model.some_ids": ids})
    assert out["model.some_ids"].dtype == torch.int32
    assert torch.equal(out["model.some_ids"], ids)


# ---------------------------------------------------------------------------------------------
# End to end on a real on-disk checkpoint
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_checkpoint(tmp_path_factory):
    """A format-identical checkpoint written by the package's own generator.

    Real files: safetensors shards, an index, ``config.json`` with ``quantization_config``, the
    ``model.language_model.*`` wrapper prefix, fp8 weights with ``weight_scale`` / ``input_scale``
    siblings, and an unquantized ``lm_head``. Small dims so it builds in seconds.
    """
    out = tmp_path_factory.mktemp("synthetic_ckpt") / "ckpt"
    script = Path(__file__).resolve().parents[2] / "scripts" / "make_synthetic_checkpoint.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out",
            str(out),
            "--layers",
            "2",
            "--hidden",
            "512",
            "--intermediate",
            "1024",
            "--vocab",
            "256",
            "--seed",
            "5",
        ],
        check=True,
        capture_output=True,
    )
    return out


def test_loader_end_to_end_on_a_real_checkpoint(synthetic_checkpoint):
    """shards -> dequant -> prefix mapping -> Meta swizzle, with the swizzle verified directly."""
    from models.tt_transformers.tt.load_checkpoints import reverse_permute

    from models.demos.mistral_3_5_d_p.tt.model_config import ModelArgs

    ground_truth = torch.load(synthetic_checkpoint / "reference_bf16.pt", weights_only=True)
    head_dim = C.HEAD_DIM

    unswizzled = ModelArgs.load_state_dict(synthetic_checkpoint, convert_to_meta_format=False)
    swizzled = ModelArgs.load_state_dict(synthetic_checkpoint, convert_to_meta_format=True)

    assert set(unswizzled) == set(swizzled) == set(ground_truth), "the two load modes disagree on keys"

    # q/k are permuted; everything else is byte-identical between the two modes.
    for name in sorted(unswizzled):
        is_qk = ".q_proj.weight" in name or ".k_proj.weight" in name
        same = torch.equal(unswizzled[name], swizzled[name])
        assert same != is_qk, f"{name}: swizzle {'changed' if not same else 'did not change'} it unexpectedly"

    for name in sorted(n for n in unswizzled if ".q_proj.weight" in n or ".k_proj.weight" in n):
        tensor = unswizzled[name]
        n_heads = tensor.shape[0] // head_dim
        want = reverse_permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
        assert torch.equal(swizzled[name], want), f"{name}: Meta swizzle does not match reverse_permute"
    logger.info(f"loader end to end OK on {len(unswizzled)} tensors ({len(ground_truth)} expected)")


def test_reduced_depth_load_truncates_layers(synthetic_checkpoint):
    """``num_layers`` must drop the higher layers, so a reduced-depth device run can load a full
    checkpoint without materialising weights it will not build."""
    from models.demos.mistral_3_5_d_p.tt.model_config import ModelArgs

    one_layer = ModelArgs.load_state_dict(synthetic_checkpoint, convert_to_meta_format=False, num_layers=1)
    assert any("model.layers.0." in k for k in one_layer), "layer 0 was dropped"
    assert not any("model.layers.1." in k for k in one_layer), "layer 1 should have been truncated"
    # The top-level tensors are not layer-scoped and must survive.
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        assert name in one_layer, f"{name} was dropped by the layer truncation"

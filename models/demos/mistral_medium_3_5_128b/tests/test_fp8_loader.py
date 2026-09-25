# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1 loader, host only: the per-tensor fp8 dequantization of ``reference/checkpoint.py`` against a
reference dequantization of the packed fp8 weights + scales (the shared
``deepseek_v3_d_p`` per-tensor helper, and the explicit ``w_fp8.float() * scale_inv`` formula), plus the
checkpoint key walk: every language-model tensor is covered, every fp8 weight has its scale, and the
fail-loud paths fire."""

import os
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3_d_p.utils.test_utils import _dequantize_per_tensor_fp8_state_dict
from models.demos.mistral_medium_3_5_128b.config import MistralMediumConfig
from models.demos.mistral_medium_3_5_128b.reference.checkpoint import (
    LM_PREFIX,
    CheckpointReader,
    dequantize_fp8_per_tensor,
)

CKPT = os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL")
requires_checkpoint = pytest.mark.skipif(
    not (CKPT and (Path(CKPT) / "model.safetensors.index.json").is_file()),
    reason="needs the real checkpoint in PREFILL_HF_MODEL / HF_MODEL",
)
LINEARS = [f"self_attn.{p}_proj" for p in "qkvo"] + [f"mlp.{p}_proj" for p in ("gate", "up", "down")]


def test_dequant_semantics_and_fail_loud(expect_error):
    w = torch.randn(64, 32).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.0123, dtype=torch.bfloat16)
    raw = {
        "a.weight": w,
        "a.weight_scale_inv": scale,
        "a.activation_scale": torch.tensor(3.0),
        "n.weight": torch.ones(32, dtype=torch.bfloat16),
    }
    out = dequantize_fp8_per_tensor(raw)
    assert set(out) == {"a.weight", "n.weight"}, "scales and activation scales must not become weights"
    assert torch.equal(out["a.weight"], (w.float() * scale.float()).to(torch.bfloat16))
    assert torch.equal(out["n.weight"], raw["n.weight"])
    with expect_error(ValueError, "has no a.weight_scale_inv"):
        dequantize_fp8_per_tensor({"a.weight": w})
    with expect_error(ValueError, "per-tensor scalar"):
        dequantize_fp8_per_tensor({"a.weight": w, "a.weight_scale_inv": torch.ones(64, 1)})


@requires_checkpoint
def test_checkpoint_key_walk_covers_the_language_model():
    cfg = MistralMediumConfig.from_json(Path(CKPT) / "config.json")
    reader = CheckpointReader(CKPT)
    for i in (0, cfg.num_hidden_layers - 1):
        names = set(reader.names(f"{LM_PREFIX}layers.{i}."))
        want = {f"{LM_PREFIX}layers.{i}.{n}.weight" for n in ("input_layernorm", "post_attention_layernorm")}
        for lin in LINEARS:
            base = f"{LM_PREFIX}layers.{i}.{lin}"
            want |= {f"{base}.weight", f"{base}.weight_scale_inv", f"{base}.activation_scale"}
        assert names == want, f"layer {i}: unexpected checkpoint keys {sorted(names ^ want)}"
    lm = [k for k in reader.weight_map if k.startswith(LM_PREFIX) or k == "lm_head.weight"]
    assert len(lm) == 23 * cfg.num_hidden_layers + 3
    layer_ids = {int(k[len(f"{LM_PREFIX}layers.") :].split(".")[0]) for k in lm if ".layers." in k}
    assert layer_ids == set(range(cfg.num_hidden_layers))


@requires_checkpoint
@pytest.mark.timeout(900)
def test_dequantized_layer_matches_reference_dequant():
    reader = CheckpointReader(CKPT)
    prefix = f"{LM_PREFIX}layers.0."
    raw = reader.read(reader.names(prefix))
    for lin in LINEARS:
        assert raw[f"{prefix}{lin}.weight"].dtype == torch.float8_e4m3fn
        assert raw[f"{prefix}{lin}.weight_scale_inv"].ndim == 0
    ours = reader.layer_state_dict(0)
    oracle = {k[len(prefix) :]: v for k, v in _dequantize_per_tensor_fp8_state_dict(raw).items()}
    assert (
        set(ours)
        == set(oracle)
        == {f"{lin}.weight" for lin in LINEARS}
        | {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        }
    )
    for name in ours:
        assert ours[name].dtype == torch.bfloat16 and torch.equal(ours[name], oracle[name]), name
    q = raw[f"{prefix}self_attn.q_proj.weight"].float() * raw[f"{prefix}self_attn.q_proj.weight_scale_inv"].float()
    assert torch.equal(ours["self_attn.q_proj.weight"], q.to(torch.bfloat16))
    # modules_to_not_convert: the embedding and lm_head are stored bf16, not fp8.
    assert reader.read([f"{LM_PREFIX}embed_tokens.weight"])[f"{LM_PREFIX}embed_tokens.weight"].dtype == torch.bfloat16
    assert reader.read(["lm_head.weight"])["lm_head.weight"].dtype == torch.bfloat16

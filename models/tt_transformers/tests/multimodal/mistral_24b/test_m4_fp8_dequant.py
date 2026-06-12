# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for the Mistral-Small-4 fp8 dequant loader (no device).

Mistral-Small-4 ships vanilla fp8 (quant_method=fp8, weight_block_size=null): per-tensor
weight_scale_inv [] on MLA/shared-expert linears and per-expert *_scale_inv [E,1,1] on the
stacked experts. dequantize_fp8_state_dict turns W (F8_E4M3) into W.float()*scale_inv -> bf16
and drops the unused static activation scales. These tests prove that math without a device.
"""
import glob
import os

import pytest
import torch

from models.tt_transformers.tt.load_checkpoints import dequantize_fp8_state_dict

FP8 = getattr(torch, "float8_e4m3fn", None)
pytestmark = pytest.mark.skipif(FP8 is None, reason="torch lacks float8_e4m3fn")


def _snap():
    cands = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--mistralai--Mistral-Small-4-119B-2603/snapshots/*/")
    )
    return cands[0] if cands else None


def test_fp8_dequant_synthetic_per_tensor_and_per_expert():
    """Both scale shapes: scalar [] (linear) and [E,1,1] (stacked experts) — pure CPU, no ckpt."""
    torch.manual_seed(0)
    # per-tensor linear: "<name>.weight" (fp8) + "<name>.weight_scale_inv" (scalar)
    w = (torch.randn(8, 16) * 0.1).to(FP8)
    s = torch.tensor(0.0007, dtype=torch.bfloat16)
    # per-expert stacked: "<name>" (fp8 [E, ...]) + "<name>_scale_inv" ([E,1,1])
    E = 4
    ew = (torch.randn(E, 6, 5) * 0.1).to(FP8)
    es = (torch.rand(E, 1, 1) * 0.001).to(torch.bfloat16)

    sd = {
        "m.o_proj.weight": w,
        "m.o_proj.weight_scale_inv": s,
        "m.o_proj.activation_scale": torch.tensor(1.0, dtype=torch.bfloat16),  # must be dropped
        "m.experts.down_proj": ew,
        "m.experts.down_proj_scale_inv": es,
        "m.experts.down_proj_activation_scale": torch.ones(E, dtype=torch.bfloat16),  # dropped
        "m.norm.weight": torch.ones(8, dtype=torch.bfloat16),  # bf16 passthrough
    }
    out = dequantize_fp8_state_dict(sd)

    # scale / activation tensors are gone; weights + bf16 passthrough remain
    assert set(out) == {"m.o_proj.weight", "m.experts.down_proj", "m.norm.weight"}
    assert out["m.norm.weight"] is sd["m.norm.weight"]  # untouched passthrough

    # per-tensor correctness
    w_lin = out["m.o_proj.weight"]
    assert w_lin.dtype == torch.bfloat16 and w_lin.shape == (8, 16)
    exp_lin = (w.to(torch.float32) * s.to(torch.float32)).to(torch.bfloat16)
    assert torch.equal(w_lin, exp_lin)

    # per-expert broadcast correctness (each expert scaled by its own [1,1] scalar)
    w_exp = out["m.experts.down_proj"]
    assert w_exp.dtype == torch.bfloat16 and w_exp.shape == (E, 6, 5)
    exp_exp = (ew.to(torch.float32) * es.to(torch.float32)).to(torch.bfloat16)
    assert torch.equal(w_exp, exp_exp)
    for e in range(E):  # the broadcast really applied a per-expert scalar
        assert torch.equal(w_exp[e], (ew[e].to(torch.float32) * float(es[e])).to(torch.bfloat16))


def test_fp8_dequant_noop_on_bf16_only():
    sd = {"a.weight": torch.ones(4, 4, dtype=torch.bfloat16)}
    assert dequantize_fp8_state_dict(sd) is sd  # nothing fp8 -> same object, no copy


@pytest.mark.skipif(_snap() is None, reason="Mistral-Small-4 checkpoint not present")
def test_fp8_dequant_real_mla_linears():
    """Real per-tensor MLA weights from the checkpoint dequantize to finite bf16."""
    from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

    prefix = "language_model.model.layers.0.self_attn."
    sd = load_hf_state_dict_filtered(_snap(), [prefix])
    fp8_weights = {k for k, v in sd.items() if v.dtype == FP8}
    assert fp8_weights, "expected fp8 MLA weights in layer 0"

    out = dequantize_fp8_state_dict(sd)
    assert not any(k.endswith("_scale_inv") or k.endswith("activation_scale") for k in out)
    for k in fp8_weights:
        assert out[k].dtype == torch.bfloat16
        assert out[k].shape == sd[k].shape
        assert torch.isfinite(out[k]).all()
        # matches manual dequant with the per-tensor scalar scale
        exp = (sd[k].to(torch.float32) * sd[k[: -len(".weight")] + ".weight_scale_inv"].to(torch.float32)).to(
            torch.bfloat16
        )
        assert torch.equal(out[k], exp)

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scratch: DSV3 prefill MLA on a 1x4 (SP1 x TP4) mesh with per-stage output stats."""
import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.test_mla import run_model
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA


def _stat(name, t):
    if isinstance(t, (tuple, list)):
        for i, x in enumerate(t):
            _stat(f"{name}[{i}]", x)
        return
    if not isinstance(t, ttnn.Tensor):
        return
    ttnn.synchronize_device(t.device())
    rows = []
    for d, dt in enumerate(ttnn.get_device_tensors(t)):
        x = ttnn.to_torch(dt).float()
        rows.append(
            f"d{d}: mean|x|={x.abs().mean().item():.4g} nz={(x != 0).float().mean().item():.3f} nan={int(x.isnan().sum())}"
        )
    print(f"STAGE {name} shape={tuple(t.shape)} :: " + " | ".join(rows), flush=True)


_REF = {}


def _qa_ref_check(out):
    from tests.ttnn.utils_for_testing import comp_pcc

    config, weights = _REF["cw"]
    torch.manual_seed(42)
    h = torch.randn(1, 640, config.hidden_size).to(torch.bfloat16).float()
    q = h @ weights["q_a_proj.weight"].float().T
    ln = weights["q_a_layernorm.weight"].float()
    ref = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps) * ln
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(ref.shape)
    ok, msg = comp_pcc(ref, got, 0.99)
    print(
        f"QA_REF mean|ref|={ref.abs().mean().item():.4f} mean|got|={got.abs().mean().item():.4f} pcc={msg} "
        f"maxabs_err={(ref-got).abs().max().item():.4f} mean_abs_err={(ref-got).abs().mean().item():.5f}",
        flush=True,
    )


def _wrap(monkeypatch, name):
    orig = getattr(ttMLA, name)

    def f(self, *a, **k):
        out = orig(self, *a, **k)
        _stat(name, out)
        if (
            name == "_q_a_latent"
            and "cw" in _REF
            and out.shape[2] == 640
            and ttnn.get_device_tensors(out)[0].storage_type() is not None
        ):
            try:
                _qa_ref_check(out)
            except Exception as e:  # collect pass has stand-in tensors
                print("QA_REF skipped:", type(e).__name__, str(e)[:80], flush=True)
        return out

    monkeypatch.setattr(ttMLA, name, f)


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], ids=["fabric1d"], indirect=True
)
@pytest.mark.timeout(0)
def test_mla_agrn_1x4(request, mesh_device, variant, is_ci_env, is_ci_v2_env, device_params, monkeypatch):
    _REF["cw"] = request.getfixturevalue("random_weights")
    for n in ("_q_a_latent", "_q_stem", "_kv_stem", "_dense_single_attn"):
        if hasattr(ttMLA, n):
            _wrap(monkeypatch, n)
    run_model(variant, False, request, mesh_device, 640, False, True, False, True, is_ci_v2_env, device_params)

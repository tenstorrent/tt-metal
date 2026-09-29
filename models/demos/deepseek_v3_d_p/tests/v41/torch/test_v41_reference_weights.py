# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the device-weight helper ``tests/v41/reference_weights.py`` (host only, no device)."""

from types import SimpleNamespace

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as o
from models.demos.deepseek_v3_d_p.tests.v41 import reference_weights as rw


@pytest.fixture(autouse=True)
def _few_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prev)


def test_dequant_fp8_block_scales_hand_values():
    # 2 x 33 FP8 weight: 32x32 scale blocks, the second column block holds one column
    w = torch.tensor([[1.0] * 32 + [0.5], [-2.0] * 32 + [3.0]]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[2.0, 0.25]])
    got = rw.dequant(SimpleNamespace(weight=w, scale=scale))
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, torch.tensor([[2.0] * 32 + [0.125], [-4.0] * 32 + [0.75]]).bfloat16())


def test_device_weights_include_moe(tmp_path, monkeypatch):
    monkeypatch.setattr(o, "CACHE_DIR", tmp_path)
    spec = o.small_spec(8)
    model = o.build_reference(spec)
    full = rw.device_weights(model, 1)
    assert set(rw.MOE_KEYS) <= full.keys() and len(full["routed_expert_weights"]) == spec.args.n_routed_experts

    calls = []
    real = rw.dequant
    monkeypatch.setattr(rw, "dequant", lambda linear: calls.append(linear) or real(linear))
    dense = rw.device_weights(model, 1, include_moe=False)
    assert dense.keys() == full.keys() - set(rw.MOE_KEYS)
    experts = {id(m) for e in model.layers[1].ffn.experts for m in (e.w1, e.w2, e.w3)}
    assert not experts & {id(m) for m in calls}, "include_moe=False must not dequantize experts"
    for key in dense:
        a, b = dense[key], full[key]
        if isinstance(a, dict):
            assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), key
        elif isinstance(a, tuple):
            assert all(torch.equal(x, y) for x, y in zip(a, b)), key
        else:
            assert torch.equal(a, b), key

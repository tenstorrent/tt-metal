# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The fused forward kernels against the composite forward, which stays as their reference implementation.

ttml.ops.metal.chunk_gated_delta_rule_fw (prep + scan kernels) and
ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule_fw_composite (existing ttnn ops only) have the same
contract.  Both are checked against the float64 chunked reference elsewhere; this file pins that they agree
with each other on every output they return, the saved h / v_new / A included, because the backward consumes
exactly those.  It includes the Qwen3.8 TP=4 per-chip shape at sequence 8192.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn
import ttml

# conftest.py puts this directory on sys.path (pytest.ini uses --import-mode=importlib).
from chunk_gated_delta_rule_reference import WEAK_GATE, assert_close  # noqa: E402

pytestmark = pytest.mark.requires_device

PCC_MIN = 0.9999
RMS_MAX = 0.01
OUT_NAMES = ("o", "final_state", "h", "v_new", "A")

# (B, T, HK, HV, K, V), chunk_size
CASES = [
    ((1, 128, 2, 2, 64, 64), 64),
    ((1, 128, 2, 4, 64, 64), 32),
    ((2, 64, 1, 3, 64, 64), 32),
    ((1, 256, 2, 6, 128, 128), 64),
    ((1, 8192, 4, 12, 128, 128), 32),  # Qwen3.8 on 4-way tensor parallelism
    ((1, 8192, 4, 12, 128, 128), 64),
]
CASE_IDS = [f"B{s[0]}_T{s[1]}_HK{s[2]}_HV{s[3]}_K{s[4]}_V{s[5]}_c{c}" for s, c in CASES]


@pytest.fixture(scope="module")
def device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _inputs(shape, with_h0, gate_dtype, device, seed=7):
    B, T, HK, HV, K, V = shape
    gen = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=gen)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True)
    flat = lambda t: t.reshape(B, 1, T, -1)
    up = lambda t, dtype: ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return {
        "q": up(flat(l2(rn(B, T, HK, K))), ttnn.bfloat16),
        "k": up(flat(l2(rn(B, T, HK, K))), ttnn.bfloat16),
        "v": up(flat(rn(B, T, HV, V)), ttnn.bfloat16),
        "g": up(flat(F.logsigmoid(rn(B, T, HV)) * WEAK_GATE), gate_dtype),
        "beta": up(flat(torch.rand(B, T, HV, generator=gen)), gate_dtype),
        "initial_state": up(rn(B, HV, K, V) * 0.1, ttnn.float32) if with_h0 else None,
    }


@pytest.mark.parametrize("gate_dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16_gates", "fp32_gates"])
@pytest.mark.parametrize("with_h0", [False, True], ids=["no_h0", "h0"])
@pytest.mark.parametrize("shape,chunk_size", CASES, ids=CASE_IDS)
def test_fused_matches_composite(shape, chunk_size, with_h0, gate_dtype, device):
    if shape[1] > 1024 and (with_h0 or gate_dtype == ttnn.float32):
        pytest.skip("the long shape is run once, with bf16 gates and no initial state")
    t = _inputs(shape, with_h0, gate_dtype, device)
    kwargs = dict(key_head_dim=shape[4], chunk_size=chunk_size, output_head_major=True)
    fused = ttml.ops.metal.chunk_gated_delta_rule_fw(
        t["q"], t["k"], t["v"], t["g"], t["beta"], initial_state=t["initial_state"], **kwargs
    )
    composite = ttml.ops.chunk_gated_delta_rule.chunk_gated_delta_rule_fw_composite(
        t["q"], t["k"], t["v"], t["g"], t["beta"], initial_state=t["initial_state"], **kwargs
    )
    for name, f, c in zip(OUT_NAMES, fused, composite):
        got = ttnn.to_torch(f).double()
        ref = ttnn.to_torch(c).double().reshape(got.shape)
        assert_close(name, got, ref, pcc_min=PCC_MIN, rms_max=RMS_MAX)

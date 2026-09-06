# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TDD device test for MTP T=K speculative decode on the fused decode op.

Chains K fused T=1 ``ttnn.transformer.decode_gated_delta_rule`` steps with
K+1 staged state slots (MTP acceptance semantics: slot i = state after
accepting i draft tokens; slot 0 = committed state). Per step t the op runs
with ``initial_state=slots[t]`` and ``inplace_state=False`` so every slot is
a distinct snapshot; slot distinctness is proven by content — if the op
aliased ``initial_state``, slot t would hold H_{t+1} and fail its pcc gate.

Golden: K-step chain of ``recurrent_gated_delta_rule_decode_golden``
(imported from the proven T=1 test, identical by construction) with the
same chaining.

Shapes/seeds/distributions/dtype are exactly the proven T=1 set
(``i-gdn-decode.log`` / ``i-gdn-decode-verify.log`` PASS):

  B=1 H=24 K=128 V=128 (primary, qwen36 per-die) and  B=1 H=32 K=32 V=32
  bf16 TILE DRAM, K=4 draft steps, gates pcc >= 0.99 every step and slot.

Prints ``I_GDN_MTP ... VERDICT: PASS|FAIL``. Device claim requires a
mesh-handoff log of this node printing VERDICT: PASS.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.transformers.test_decode_gated_delta_rule import (
    _pcc,
    _to_torch_one_device,
    recurrent_gated_delta_rule_decode_golden,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

K_DRAFT = 4


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="p300_1x2")], indirect=True)
def test_decode_gated_delta_rule_mtp_k4_vs_golden(mesh_device):
    assert hasattr(
        ttnn.transformer, "decode_gated_delta_rule"
    ), "ttnn.transformer.decode_gated_delta_rule is not bound in this tree"

    cases = ((1, 24, 128, 128, 1), (1, 32, 32, 32, 0))
    worst_o = worst_h = 1.0
    shape_tags = []
    pairs = []
    for B, H, K, V, seed in cases:
        torch.manual_seed(seed)
        scale = K**-0.5
        # T=1 test distributions, K draws: q/k randn, v randn, beta rand, g -rand*0.5.
        q = [torch.randn(B, 1, H, K) for _ in range(K_DRAFT)]
        k = [torch.randn(B, 1, H, K) for _ in range(K_DRAFT)]
        v = [torch.randn(B, 1, H, V) for _ in range(K_DRAFT)]
        beta = [torch.rand(B, 1, H) for _ in range(K_DRAFT)]
        g = [-torch.rand(B, 1, H) * 0.5 for _ in range(K_DRAFT)]
        h0 = torch.randn(B, H, K, V)

        # Golden chain: H_0 = h0, H_{t+1} after accepting draft token t.
        gold_h = h0.float()
        gold_os = []
        gold_hs = []
        for t in range(K_DRAFT):
            gold_o, gold_h = recurrent_gated_delta_rule_decode_golden(
                q[t], k[t], v[t], beta[t], g[t], scale=scale, initial_state=gold_h
            )
            gold_os.append(gold_o)
            gold_hs.append(gold_h)

        kw = dict(
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # K+1 staged H slots; slot 0 = committed initial state.
        slots = [ttnn.from_torch(h0, **kw)]
        case_o_min = case_h_min = 1.0
        for t in range(K_DRAFT):
            o_t, h_next = ttnn.transformer.decode_gated_delta_rule(
                ttnn.from_torch(q[t], **kw),
                ttnn.from_torch(k[t], **kw),
                ttnn.from_torch(v[t], **kw),
                ttnn.from_torch(beta[t], **kw),
                ttnn.from_torch(g[t], **kw),
                scale=scale,
                initial_state=slots[t],
                inplace_state=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            slots.append(h_next)  # distinct buffer per non-inplace op contract
            pcc_o = _pcc(_to_torch_one_device(o_t), gold_os[t])
            pcc_h = _pcc(_to_torch_one_device(slots[t + 1]), gold_hs[t])
            print(f"I_GDN_MTP case B={B} H={H} K={K} V={V} step t={t} " f"pcc_o={pcc_o:.6f} pcc_h={pcc_h:.6f}")
            case_o_min, case_h_min = min(case_o_min, pcc_o), min(case_h_min, pcc_h)
            pairs.append((gold_os[t], o_t))
            pairs.append((gold_hs[t], slots[t + 1]))
        print(
            f"I_GDN_MTP case B={B} H={H} K={K} V={V} K={K_DRAFT} "
            f"pcc_o_min={case_o_min:.6f} pcc_h_min={case_h_min:.6f}"
        )
        worst_o, worst_h = min(worst_o, case_o_min), min(worst_h, case_h_min)
        shape_tags.append(f"H{H}K{K}V{V}")

    ok = worst_o >= 0.99 and worst_h >= 0.99
    print(
        f"I_GDN_MTP: K={K_DRAFT} chained fused T=1 decode, K+1 staged H slots, "
        f"shapes={','.join(shape_tags)} pcc_o_min={worst_o:.6f} "
        f"pcc_h_min={worst_h:.6f} VERDICT: {'PASS' if ok else 'FAIL'}"
    )
    for gold, got in pairs:
        assert_with_pcc(gold, _to_torch_one_device(got), pcc=0.99)

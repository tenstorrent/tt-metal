# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S0 CPU tests of the layer-moe task: placement, shapes, router set-match on real activations, bfp4 pack round-trip on real expert slices.
Run: python_env/bin/python -m pytest --noconftest -c /dev/null --rootdir=/tmp models/demos/blackhole/hy4_preview/tests/test_hy4_moe_cpu.py -q -s"""
import os

import pytest
import torch

from models.demos.blackhole.hy4_preview.tt import hy4_moe_place as P
from models.demos.blackhole.hy4_preview.tt.hy4_router import route_cpu, set_match

ACTS = "/mnt/tt-data/ssinghal/hy4-logs/program/layer-moe/acts/"


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def test_placement_256_over_32():
    a = P.validate()
    cnt = [a.count(c) for c in range(32)]
    assert set(cnt) == {8}
    for r in range(4):
        for c in range(8):
            es = P.experts_of_chip(r, c)
            assert all(P.chip_of(e) == r * 8 + c and P.row_col(e) == (r, c) for e in es)
            assert sorted(e for e in range(256) if a[e] == r * 8 + c) == sorted(es)


def test_placement_matches_ttmoe_sequential_mapping():
    """the decode placement (TTMoEDecodeConfig 'sequential', cluster_axis 0) is exactly hy4_moe_place.sequential(); the config of hy4_preview.yaml is consistent."""
    import yaml

    from models.common.moe.tt_moe_decode_config import (
        ExpertStateConfig as ExpertsConfig,  # the class that expands the keyword
    )
    from models.demos.blackhole.hy4_preview.tt.hy4_moe import CONFIG_PATH

    y = yaml.safe_load(CONFIG_PATH.read_text())
    assert (y["hidden_size"], y["num_routed_experts"], y["select_experts_k"], y["compute"]["intermediate_size"]) == (
        6144,
        256,
        8,
        2048,
    )
    assert y["score_func"] == "sigmoid" and y["routed_scaling_factor"] == 2.827 and y["compute"]["swiglu_limit"] == 10.0
    out = ExpertsConfig._expand_convenience_keywords(
        dict(
            mesh_shape=(4, 8), cluster_axis=0, num_routed_experts=256, num_shared_experts=0, expert_mapping="sequential"
        )
    )
    assert list(out["expert_mapping"]) == P.sequential()


def test_shapes_real_checkpoint():
    from models.demos.blackhole.hy4_preview.weights import Hy4Weights, layer_names

    W = Hy4Weights()
    for L in (0, 1, 40, 77, 78):
        assert W.check_headers(layer_names(W.cfg, L)) == []
    assert W.header("layers.1.moe.experts.gate_up")["shape"] == [256, 4096, 6144]
    assert W.header("layers.1.moe.experts.down")["shape"] == [256, 6144, 2048]
    assert W.header("layers.0.mlp.gate_proj")["shape"] == [18432, 6144]


@pytest.mark.skipif(not os.path.exists(ACTS + "acts_l0_l1.pt"), reason="real activations not dumped")
def test_router_setmatch_real_acts():
    acts = torch.load(ACTS + "acts_l0_l1.pt")
    g = torch.load(ACTS + "l1_gate_shared.pt")
    w, b = g["mlp.gate.weight"], g["mlp.gate.e_score_correction_bias"]
    X = torch.cat([h.reshape(-1, h.shape[-1]) for h, y, r in acts["moe"]])
    R = torch.cat([r.reshape(-1, 8) for h, y, r in acts["moe"]])
    wo, io, ro = route_cpu(X, w, b)
    assert set_match(io, R) == 1.0
    wd, idd, rd = route_cpu(X, w, b, emulate_device=True)
    sm = set_match(io, idd)
    bad = (io.sort(-1)[0] != idd.sort(-1)[0]).any(-1).nonzero().flatten()
    gaps = (
        torch.stack([(ro[r].topk(9)[0][7] - ro[r].topk(9)[0][8]).abs() for r in bad.tolist()])
        if len(bad)
        else torch.zeros(1)
    )
    print(
        f"router set-match bf16-input vs fp32 oracle: {sm:.5f} over {len(X)} tokens, max k/(k+1) gap of mismatches {gaps.max():.2e}"
    )
    assert sm >= 0.98 and gaps.max() < 3e-4  # every flip is a near tie (bf16 activation rounding)


@pytest.mark.parametrize("layer,e", [(1, 250), (1, 5), (40, 17), (77, 255)])
def test_bfp4_pack_roundtrip_real_expert(layer, e):
    import ttnn
    from models.demos.blackhole.hy4_preview.weights import Hy4Weights

    W = Hy4Weights()
    g, u, d = W.load_expert(layer, e)
    out = {}
    for name, m in (("gate", g), ("up", u), ("down", d)):
        t = m.T.contiguous().to(torch.bfloat16)[None, None]  # [in, out] as uploaded
        q = ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
        out[name] = (pcc(q, t.float()), float((q - t.float()).norm() / t.float().norm()))
        assert out[name][0] > 0.98
    print(
        f"bfp4 round trip layer {layer} expert {e}: "
        + " ".join(f"{k} pcc {v[0]:.4f} relL2 {v[1]:.3f}" for k, v in out.items())
    )

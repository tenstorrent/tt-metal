# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device router expert choice vs the HF fp32 router on real hidden states.

The router picks ``top_k`` of ``num_experts`` experts per token from ``sigmoid(logits) + bias``. HF does
this in fp32; ``ttnn.topk`` accepts only bf16, so near-equal experts can swap. This test runs the
production ``OptimizedDecoder._route`` (shared by every decoder class) on router inputs captured from a
real prompt by ``gen_streamed_reference.py --router-dump`` and measures, per MoE layer:

  * slot agreement: mean over tokens of |device picks ∩ HF picks| / top_k;
  * the routing-weight error on the experts both picked.

``precise`` (the default router) must agree at least as well as the original ``bf16`` router and stay
above ``MIN_SLOT_AGREEMENT``. Set ``LAGUNA_ROUTER_DUMP`` to the dump directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.laguna.tests import laguna_reference as R
from models.demos.laguna.tests.laguna_test_utils import close_mesh, open_mesh, resolve_profile
from models.demos.laguna.tt.optimized_decoder import LayerConfig, OptimizedDecoder

DUMP_DIR = os.environ.get("LAGUNA_ROUTER_DUMP")
pytestmark = pytest.mark.skipif(not DUMP_DIR, reason="set LAGUNA_ROUTER_DUMP to a gen_streamed_reference router dump")

PROFILE = resolve_profile()
# Floor for the default (precise) router. The device router input is the bf16 activation stream, so a
# small residue of disagreement on genuine near-ties is expected even with an exact top-k. Measured on S
# (2026-10-01, all 47 MoE layers): precise min 0.9931 / mean 0.9985; bf16 min 0.7824 / mean 0.9755.
MIN_SLOT_AGREEMENT = 0.99


class _RouterOnly(OptimizedDecoder):
    """Just the router state ``_route`` reads: cfg, router weights, router compute configs, precision."""

    def __init__(self, cfg, weights, mesh_device, precision):
        self.cfg = cfg
        self.w = weights
        self.device = mesh_device
        self.router_precision = precision
        self._ck_router = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,  # production policy fid_router
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._ck_router_precise = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,  # OptimizedDecoder._ck_router_precise
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )


@pytest.fixture(scope="module")
def device():
    dev = open_mesh(ttnn, PROFILE)
    yield dev
    close_mesh(ttnn, dev)


def _dumps():
    if not DUMP_DIR:
        return []
    return sorted(Path(DUMP_DIR).glob("router_L*.pt"))


def _rep(device, t, dtype, layout=ttnn.TILE_LAYOUT):
    mapper = ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, mesh_mapper=mapper)


def _to_host(device, t):
    if device.get_num_devices() == 1:
        return ttnn.to_torch(t)
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0:1]


def _route(device, cfg, dump, precision):
    E = cfg.num_experts
    weights = {
        "gate_w": _rep(device, dump["gate_weight"].t().contiguous(), ttnn.bfloat16),
        "e_bias": _rep(device, dump["e_score_correction_bias"].reshape(1, 1, 1, E), ttnn.bfloat16),
        "e_bias_f32": _rep(device, dump["e_score_correction_bias"].reshape(1, 1, 1, E), ttnn.float32),
    }
    router = _RouterOnly(cfg, weights, device, precision)
    x = dump["x"]
    T = x.shape[0]
    _, idx, wsel = router._route(_rep(device, x.reshape(1, 1, T, -1), ttnn.bfloat16))
    idx_h = _to_host(device, idx).reshape(-1, cfg.top_k)[:T].to(torch.int64)
    w_h = _to_host(device, wsel).reshape(-1, cfg.top_k)[:T].float()
    return idx_h, w_h


def _agreement(cfg, dump, idx, w):
    hf_idx = dump["hf_experts"].to(torch.int64)
    hf_w = dump["hf_weights"].float() * cfg.routed_scaling
    slots, weight_err = [], []
    for t in range(hf_idx.shape[0]):
        dev_set, hf_set = set(idx[t].tolist()), set(hf_idx[t].tolist())
        slots.append(len(dev_set & hf_set) / cfg.top_k)
        hf_map = dict(zip(hf_idx[t].tolist(), hf_w[t].tolist()))
        for e, wv in zip(idx[t].tolist(), w[t].tolist()):
            if e in hf_map:
                weight_err.append(abs(wv - hf_map[e]))
    return sum(slots) / len(slots), max(weight_err) if weight_err else 0.0


@pytest.mark.parametrize("dump_path", _dumps(), ids=lambda p: p.stem)
def test_router_matches_hf_on_real_hidden_states(device, dump_path):
    dump = torch.load(dump_path)
    cfg = LayerConfig.from_hf(R.build_config(), int(dump["layer"]))
    assert cfg.is_moe
    results = {}
    for precision in ("bf16", "precise"):
        idx, w = _route(device, cfg, dump, precision)
        assert ((idx >= 0) & (idx < cfg.num_experts)).all()
        assert all(len(set(row.tolist())) == cfg.top_k for row in idx), "duplicate expert in a token's picks"
        results[precision] = _agreement(cfg, dump, idx, w)
    print(
        f"ROUTER layer={dump['layer']} tokens={dump['x'].shape[0]} "
        f"bf16 slot_agree={results['bf16'][0]:.4f} max_w_err={results['bf16'][1]:.4f} | "
        f"precise slot_agree={results['precise'][0]:.4f} max_w_err={results['precise'][1]:.4f}"
    )
    assert results["precise"][0] >= results["bf16"][0], "precise router agrees less with HF than the bf16 router"
    assert results["precise"][0] >= MIN_SLOT_AGREEMENT, (
        f"layer {dump['layer']} precise router slot agreement {results['precise'][0]:.4f} < {MIN_SLOT_AGREEMENT}"
    )

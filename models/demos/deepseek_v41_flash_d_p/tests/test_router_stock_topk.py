# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The router's fused gate (ROUTER_IMPL=fused: tt/router.py:fused_gate, the stock
ttnn.experimental.deepseek_prefill.moe_grouped_topk) against its torch semantics in float64. Device test, 2x2:

    scripts/run_safe_pytest.sh --run-all --no-precompile \
        models/demos/deepseek_v41_flash_d_p/tests/test_router_stock_topk.py -s

Ported from the retired fork's model case (ttnn/ttnn/bringup/moe_grouped_topk/tests/test_moe_grouped_topk.py, case
deepseek_v41_flash_d_p-2x2-s1024-e96-top6-sqrtsoftplus-fp32, O.1, sig 0fad887387), same inputs and bars; the call is
the model's own (fused_gate), so its arguments cannot drift from router.py. Every chip gets its own rows (sharded over
the mesh on dim 0) and is checked on them.

- Selection (data selection on the fp32 keys): on every CLEAR row the set of selected experts equals the float64 one
  (exact match). A row is clear when its k-th / (k+1)-th key gap exceeds the keys' noise floor (fork_cases.noise_limit:
  calib x the fp32 keys' worst error vs float64 on this case's inputs); the rest are near ties either order may take,
  and the clear rows must be the bulk of the case (>= 99 %). The ids must be valid (k distinct experts < E).
- Weights (math): the dense [rows, E] weights of the clear rows vs float64 (the float64 selection), rel L2 within
  noise_limit at the declared formats (fp32 logits, bf16 out); PCC printed (a diagnostic).
- Determinism: 5 runs on A with one on B between, bit-identical.

Semantics (single group: n_groups 1, topk_groups 1): scores = sqrt(softplus(logits)); keys = scores + bias (the
selection only); the top-k experts by key, descending (ties to the lower expert id); weights = the UNBIASED scores of
the selected experts / (their sum + epsilon) x route_scale."""

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism
from models.demos.common.bringup.testing.fork_cases import noise_limit
from models.demos.deepseek_v41_flash_d_p.tt.router import fused_gate

CLEAR_SHARE = 0.99

CASE = {
    # The router after the gate matmul on the chip's 1024 rows (scaled 2x2 run): sqrtsoftplus scores of the 96 (scaled)
    # experts, a per-expert fp32 bias for the selection only (passed full-shape, [rows, E]), top 6, the unbiased scores
    # normalized x 1.5 / (sum + 1e-20). Logits ~ 2 N(0, 1) - 2 (most negative, a top tail), bias 0.05 N(0, 1) per
    # expert.
    "id": "deepseek_v41_flash_d_p-2x2-s1024-e96-top6-sqrtsoftplus",
    "mesh": [2, 2],
    "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
    "shape": [1, 1, 1024, 96],
    "top_k": 6,
    "route_scale": 1.5,
    "epsilon": 1e-20,
    "logit_scale": 2.0,
    "logit_shift": -2.0,
    "bias_scale": 0.05,
    "seed": 0,
    "noise": {"keys": {"inputs": ["fp32", "fp32"], "out": "fp32"}, "weights": {"inputs": ["fp32"], "out": "bf16"}},
}


# ---- the torch reference (single group, sqrtsoftplus)
def ref_scores(logits):
    return torch.nn.functional.softplus(logits).sqrt()


def ref_keys(logits, bias):
    """The selection keys in the inputs' dtype."""
    return ref_scores(logits) + bias


def ref_select(k, top_k):
    """[rows, top_k] expert ids by key, descending, ties to the lower id (a stable sort of -key)."""
    return torch.sort(-k, dim=-1, stable=True).indices[..., :top_k]


def ref_weights(logits, idx, route_scale, epsilon):
    """The normalized weights of the experts ``idx`` (a fixed selection), in the logits' dtype."""
    w = ref_scores(logits).gather(-1, idx)
    return w / (w.sum(-1, keepdim=True) + epsilon) * route_scale


def ref_gap(k, top_k):
    """Per row: the key of the k-th selected expert minus the best unselected one (a row whose gap is within the
    keys' error can legitimately select another expert)."""
    s = torch.sort(k, dim=-1, descending=True).values
    return s[..., top_k - 1] - s[..., top_k]


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    p["require_exact_physical_num_devices"] = True
    return p


def _inputs(mesh_device, c, seed):
    """Host logits / bias [n_dev, 1, R, E] (device d: block d) and their device tensors."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    g = torch.Generator().manual_seed(seed)
    _, _, R, E = c["shape"]
    logits = torch.randn(n_dev, 1, R, E, generator=g) * c["logit_scale"] + c["logit_shift"]
    bias = (torch.randn(E, generator=g) * c["bias_scale"]).expand(n_dev, 1, R, E).contiguous()
    put = lambda t: ttnn.from_torch(  # noqa: E731
        t,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    return (logits, bias), (put(logits), put(bias))


def _call(c, dev):
    return fused_gate(*dev, c["top_k"], c["route_scale"], memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [(tuple(CASE["mesh"]), _device_params(CASE))],
    ids=[CASE["id"]],
    indirect=["mesh_device", "device_params"],
)
def test_router_stock_topk(mesh_device, device_params):
    c = CASE
    k = c["top_k"]
    (logits, bias), dev = _inputs(mesh_device, c, c["seed"])
    out = _call(c, dev)
    w_d, i_d = out
    assert w_d.dtype == ttnn.bfloat16, w_d.dtype
    _, _, R, E = c["shape"]
    for d, (wt, it) in enumerate(zip(ttnn.get_device_tensors(w_d), ttnn.get_device_tensors(i_d))):
        lg, bs = logits[d, 0], bias[d, 0]
        w = ttnn.to_torch(wt).double().reshape(R, -1)[:, :k]
        idx = ttnn.to_torch(it).to(torch.int64).reshape(R, -1)[:, :k]
        assert int(idx.min()) >= 0 and int(idx.max()) < E, "expert ids out of range"
        assert all(len(set(r.tolist())) == k for r in idx), "a row selects an expert twice"
        key64 = ref_keys(lg.double(), bs.double())
        idx64 = ref_select(key64, k)
        # the keys' noise floor: calib x the fp32 keys' worst error vs float64 (abs, the gap's unit)
        key32 = ref_keys(lg, bs).double()
        lim_k = noise_limit(ref_keys, [lg, bs], *c["noise"]["keys"].values())
        tol = lim_k["rel"] / max(lim_k["noise_rel"], 1e-300) * float((key32 - key64).abs().max())
        clear = ref_gap(key64, k) > tol
        n_clear = int(clear.sum())
        got_sets = [frozenset(r.tolist()) for r in idx]
        want_sets = [frozenset(r.tolist()) for r in idx64]
        bad = [r for r in range(R) if clear[r] and got_sets[r] != want_sets[r]]
        near = [r for r in range(R) if not clear[r] and got_sets[r] != want_sets[r]]
        print(
            f"{c['id']} dev {d}: {n_clear}/{R} clear rows (gap > {tol:.2e}); {len(bad)} clear rows differ, "
            f"{len(near)} near-tie rows differ"
        )
        assert n_clear >= CLEAR_SHARE * R, f"dev {d}: only {n_clear} of {R} rows clear of the keys' noise"
        assert not bad, f"dev {d}: {len(bad)} clear rows select other experts than float64 (first {bad[:5]})"
        m = clear.nonzero().flatten()
        dense = torch.zeros(R, E, dtype=torch.float64).scatter_(1, idx, w)[m]
        fw = lambda lgt: torch.zeros(len(m), E, dtype=lgt.dtype).scatter_(  # noqa: E731
            1, idx64[m], ref_weights(lgt, idx64[m], c["route_scale"], c["epsilon"])
        )
        want = fw(lg[m].double())
        lim = noise_limit(fw, [lg[m]], *c["noise"]["weights"].values())
        rel = float((dense - want).norm() / want.norm())
        a, b = (dense - dense.mean()).flatten(), (want - want.mean()).flatten()
        pcc = float(a @ b / (a.norm() * b.norm()))
        print(
            f"{c['id']} dev {d}: weights rel {rel:.3e} (limit {lim['rel']:.3e}, noise {lim['noise_rel']:.3e}) "
            f"pcc {pcc:.8f} (a diagnostic)"
        )
        assert rel <= lim["rel"], f"dev {d}: weights rel L2 {rel:.3e} > limit {lim['rel']:.3e}"
    dev_b = _inputs(mesh_device, c, c["seed"] + 1)[1]
    determinism.assert_deterministic(lambda: _call(c, dev), lambda: _call(c, dev_b), first=out, label=c["id"])

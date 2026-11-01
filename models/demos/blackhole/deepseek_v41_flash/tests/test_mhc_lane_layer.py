"""One decode layer (real chain activations), baseline vs DSV41_MHC_LANEPAR (and other mHC arms) in ONE process on all 32 chips.

For every arm in DSV41_LAYER_ARMS (comma list of arm names, see ARMS): eager forward -> outputs to host (bit / PCC / max-abs comparison against the first arm,
every chip), trace capture of one layer forward; then DSV41_LAYER_ROUNDS rounds in which every arm's trace is replayed DSV41_LAYER_REPLAYS times (blocking sync at
the end): us per layer. Env of the run = the production levers (see the caller) + MOE_COMPUTE_PIN_EAST_X=7; only the mHC switches differ per arm.
Env: DSV41_OPT_LAYER (3), DSV41_OPT_UPR (T per mesh row), DSV41_OPT_CHAIN, DSV41_LAYER_OUT (json).
"""

import gc
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.loader import layer_meta
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain

LAYER = int(os.environ.get("DSV41_OPT_LAYER", "3"))
UPR = int(os.environ.get("DSV41_OPT_UPR", "4"))
CHAIN = os.environ.get("DSV41_OPT_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-m")
OUT = os.environ.get("DSV41_LAYER_OUT", f"/tmp/mhc_lane_layer_T{UPR}.json")


def fit_b(t, B):
    """chain refs hold 16 users; take the first B (B<=16) or tile with small seeded noise so replicas route to different experts."""
    if not hasattr(t, "shape") or t.dim() == 0 or t.shape[0] != 16 or B == 16:
        return t
    if B < 16:
        return t[:B].contiguous()
    g = torch.Generator().manual_seed(1234)
    reps = [t] + [
        (t.float() + 0.1 * t.float().std() * torch.randn(t.shape, generator=g)).to(t.dtype) for _ in range(B // 16 - 1)
    ]
    return torch.cat(reps, 0)


def fit_ref(ref, B):
    for k in ("dec_in", "pre_in"):
        ref[k] = fit_b(ref[k], B)
    for k, v in ref["state"].items():
        if v is not None:
            ref["state"][k] = fit_b(v, B) if B <= 16 else torch.cat([v] * (B // 16), 0)
    return ref


ARMS = {
    "base": {},
    "ff": {"DSV41_MHC_FRONT_FAST": "1"},
    "lane": {"DSV41_MHC_LANEPAR": "1"},
    "lanefuse": {"DSV41_MHC_LANEPAR": "1", "DSV41_MHC_LANEFUSE": "1"},
}
for _k, _v in os.environ.items():
    if _k.startswith("DSV41_LAYER_ARM_"):
        ARMS[_k[len("DSV41_LAYER_ARM_") :]] = dict(kv.split("=") for kv in _v.split(",") if kv)
ALLK = {k for a in ARMS.values() for k in a}


def set_arm(name):
    for k in ALLK:
        os.environ.pop(k, None)
    os.environ.update(ARMS[name])


def host_all(t):
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if torch.equal(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(3000)
@torch.no_grad()
def test_mhc_lane_layer(mesh_device):
    md = mesh_device
    arms = os.environ.get("DSV41_LAYER_ARMS", "base,lane").split(",")
    rounds = int(os.environ.get("DSV41_LAYER_ROUNDS", "3"))
    replays = int(os.environ.get("DSV41_LAYER_REPLAYS", "50"))
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S = toks["prefill_tokens"].shape[1]
    set_arm(arms[0])
    chain = DSV41DecodeChain(md, users_per_row=UPR, log=lambda m: print(m, flush=True))
    meta = layer_meta(LAYER)
    if meta["ratio"] and not meta["is_kv_source"]:
        src = meta["kv_source"]
        ref_s = torch.load(os.path.join(CHAIN, f"layer_{src}.pt"))
        ref_s["S"] = S
        fit_ref(ref_s, 4 * UPR)
        chain.build_layer(src, ref_s)
    ref = torch.load(os.path.join(CHAIN, f"layer_{LAYER}.pt"))
    ref["S"] = S
    fit_ref(ref, 4 * UPR)
    layer, attn = chain.build_layer(LAYER, ref)
    B = chain.B
    st = attn.step_inputs(torch.full((B,), S))
    x, pre = chain.to_dev(ref["dec_in"].reshape(B, -1), 4 * 5120), chain.to_dev(ref["pre_in"].reshape(B, -1), 4)

    outs, tids = {}, {}
    for arm in arms:
        set_arm(arm)
        for _ in range(3):
            o, n = layer.forward(x, pre, st)
            ttnn.synchronize_device(md)
            outs[arm] = (host_all(o), host_all(n))
            # the layer output may be the persistent pooled stream buffer of the valid-row expand (mhc_stream_pool): never deallocate it
            gc.collect()
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        layer.forward(x, pre, st)
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        tids[arm] = tid
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)

    res = dict(layer=LAYER, T=UPR, B=B, arms={}, compare={})
    print(
        f"LANEUSED fused expand+front calls so far: {getattr(layer.mhc_ffn, 'lane_used', 0)} skipped: {getattr(layer.mhc_ffn, 'lane_skipped', None)}",
        flush=True,
    )
    us = {a: [] for a in arms}
    for r in range(rounds):
        for arm in arms:
            t0 = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(md, tids[arm], cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            us[arm].append((time.perf_counter() - t0) / replays * 1e6)
    for arm in arms:
        res["arms"][arm] = dict(us=us[arm], mean=sum(us[arm]) / len(us[arm]), min=min(us[arm]))
        print(
            f"LAYERARM {arm} T={UPR} mean={res['arms'][arm]['mean']:.1f} min={min(us[arm]):.1f} us all={['%.1f' % v for v in us[arm]]}",
            flush=True,
        )
    b0 = arms[0]
    for arm in arms[1:]:
        worst_eq, worst_pcc, worst_diff, ndiff = True, 1.0, 0.0, 0
        for which, name in ((0, "x"), (1, "pre")):
            for c, (a, b) in enumerate(zip(outs[arm][which], outs[b0][which])):
                eq = torch.equal(a, b)
                ndiff += 0 if eq else 1
                worst_eq &= eq
                worst_pcc = min(worst_pcc, pcc(a, b))
                worst_diff = max(worst_diff, float((a - b).abs().max()))
        res["compare"][arm] = dict(
            bit_identical=worst_eq, chips_tensors_differing=ndiff, min_pcc=worst_pcc, max_abs_diff=worst_diff
        )
        print(f"LAYERCMP {arm} vs {b0} T={UPR}: {res['compare'][arm]}", flush=True)
        d = res["arms"][arm]["mean"] - res["arms"][b0]["mean"]
        print(
            f"LAYERDELTA {arm} vs {b0} T={UPR}: {d:+.1f} us per layer, {d * 40 / 1000:+.2f} ms per 40-layer round",
            flush=True,
        )
    for tid in tids.values():
        ttnn.release_trace(md, tid)
    json.dump(res, open(OUT, "w"), indent=1)

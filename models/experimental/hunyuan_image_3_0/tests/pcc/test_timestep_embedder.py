# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — HunyuanTtTimestepEmbedder vs PyTorch reference, on the REAL
# HunyuanImage-3.0 weights, for each instantiated embedder (timestep_emb,
# time_embed, time_embed_2).
#
# Run (pytest):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_timestep_embedder.py -v -s
# Run (script):
#   python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_timestep_embedder.py

import sys, json, glob
import torch
from safetensors import safe_open

ROOT = "/home/iguser/ign-tt/tt-metal"
HUNYUAN = "/home/iguser/ign-tt/hunyan_instruct"
WEIGHTS = "/home/iguser/ign-tt/base"
for p in (ROOT, HUNYUAN):
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder

# The embedder outputs are NOT normalized — timestep_emb reaches |x|~27 — so an
# absolute tolerance is meaningless here; bf16 carries ~0.4% relative precision,
# so we gate on PCC plus a relative (scaled by output absmax) max-error bound.
PCC_THR = 0.999
RTOL = 0.01
EMBEDDERS = ["timestep_emb", "time_embed", "time_embed_2"]


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


# --- sharded weight loading -------------------------------------------------
_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
_OPEN = {}


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(f"{WEIGHTS}/{shard}", framework="pt"))
    return f.get_tensor(key)


def _load_prefix(prefix):
    return {k: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


def _run_one(device, prefix, hidden_size):
    sd = _load_prefix(prefix)

    ref = TimestepEmbedder(hidden_size)
    ref.load_state_dict({k[len(prefix) + 1 :]: v.float() for k, v in sd.items()}, strict=True)
    ref.eval()

    torch.manual_seed(0)
    t = torch.rand(8)  # 8 fractional timesteps, mirrors the scheduler range [0,1)
    with torch.no_grad():
        ref_out = ref(t)  # [8, hidden]

    tt = HunyuanTtTimestepEmbedder(device, hidden_size, sd, prefix)
    out_tt = tt(t)
    tt_out = ttnn.to_torch(out_tt)[..., :hidden_size]
    ttnn.deallocate(out_tt)
    tt.deallocate()

    p = _pcc(ref_out, tt_out)
    d = (ref_out.float() - tt_out.float()).abs().max().item()
    rel = d / (ref_out.float().abs().max().item() + 1e-9)
    return p, d, rel


def _run(device):
    cfg = json.load(open(f"{WEIGHTS}/config.json"))
    H = cfg["hidden_size"]
    results = {}
    for prefix in EMBEDDERS:
        p, d, rel = _run_one(device, prefix, H)
        ok = (p >= PCC_THR) and (rel <= RTOL)
        print(f"  [{'PASS' if ok else 'FAIL'}] {prefix:14s} PCC={p:.6f}  max|diff|={d:.6f}  rel={rel:.4%}")
        results[prefix] = (p, d, rel, ok)
    return results


def test_timestep_embedder_pcc(device):
    results = _run(device)
    for prefix, (p, d, rel, ok) in results.items():
        assert ok, f"{prefix}: PCC={p:.6f} (>= {PCC_THR}), rel={rel:.4%} (<= {RTOL:.2%})"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        results = _run(dev)
    finally:
        ttnn.close_device(dev)
    n = sum(1 for *_, ok in results.values() if ok)
    print("\n" + "=" * 60)
    print(f"TimestepEmbedder: {n}/{len(results)} PASSED")
    print("ALL PASS ✓" if n == len(results) else "SOME FAILED ✗")
    print("=" * 60)
    sys.exit(0 if n == len(results) else 1)

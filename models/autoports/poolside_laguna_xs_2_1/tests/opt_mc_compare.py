"""Validate OptimizedMultichipDecoder vs MultichipDecoder (baseline) vs HF, all on 1x4 mesh.
Prefill + decode PCC for a given layer. Usage: opt_mc_compare.py <layer> <seq> <decode_pos>"""
import json
import os
import sys

import numpy as np
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_multichip_decoder import OptimizedMultichipDecoder

HIDDEN = 2048
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 513
DPOS = int(sys.argv[3]) if len(sys.argv) > 3 else SEQ
# Mesh is env-parameterized (TT_LAGUNA_MESH="1,2" for a 2-chip host); default 1x4. Shard factors
# derive from get_num_devices(), so any factor of the head/expert dims works.
_MR, _MC = (int(x) for x in os.environ.get("TT_LAGUNA_MESH", "1,4").split(","))
_MESH_TAG = "" if (_MR, _MC) == (1, 4) else f"_{_MR}x{_MC}"


def pcc(a, b):
    a = a.flatten().float().numpy()
    b = b.flatten().float().numpy()
    return float(np.corrcoef(a, b)[0, 1])


def make_inputs():
    torch.manual_seed(SEQ)
    return torch.randn(1, SEQ, HIDDEN) * 0.5, torch.randn(1, 1, HIDDEN) * 0.5


def run(cls):
    cfg = R.build_config()
    raw = W.load_layer_tensors(LAYER)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(_MR, _MC), trace_region_size=90_000_000)
    try:
        dec = cls.from_state_dict(raw, hf_config=cfg, layer_idx=LAYER, mesh_device=dev, max_seq_len=SEQ + 64)
        mm = ttnn.ReplicateTensorToMesh(dev)
        to = lambda t, l=ttnn.TILE_LAYOUT, d=ttnn.bfloat16: ttnn.from_torch(
            t, dtype=d, layout=l, device=dev, mesh_mapper=mm
        )
        comp = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[0:1]
        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=SEQ + 64, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        x, xd = make_inputs()
        pout = dec.prefill_forward(to(x), kv, pt, user_id=0, start_pos=0)
        pref = comp(pout).float().reshape(1, SEQ, HIDDEN)
        cur = ttnn.from_torch(
            torch.tensor([DPOS], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        ridx = ttnn.from_torch(
            torch.tensor([[DPOS]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        dout = dec.decode_forward(to(xd.reshape(1, 1, 1, HIDDEN)), cur, ridx, pt, kv)
        dref = comp(dout).float().reshape(1, 1, HIDDEN)
        return pref, dref
    finally:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    cfg = R.build_config()
    raw = W.load_layer_tensors(LAYER)
    ctx = R.make_context(cfg, LAYER, state_dict=W.to_hf_layer_state_dict(raw, cfg, LAYER), dtype=torch.float32)
    x, xd = make_inputs()
    hf_p, pkv = R.reference_forward(ctx, x)
    hf_d, _ = R.reference_forward(ctx, xd, past_key_values=pkv)
    print(f"=== layer {LAYER} seq {SEQ} decode_pos {DPOS} ===")
    bp, bd = run(MultichipDecoder)
    print(f"multichip(base): prefill vs HF {pcc(bp, hf_p):.5f}  decode vs HF {pcc(bd, hf_d):.5f}")
    op, od = run(OptimizedMultichipDecoder)
    print(f"optimized-mc:    prefill vs HF {pcc(op, hf_p):.5f}  decode vs HF {pcc(od, hf_d):.5f}")
    print(f"opt vs base:     prefill {pcc(op, bp):.6f}  decode {pcc(od, bd):.6f}")
    rec = {
        "layer": LAYER,
        "seq": SEQ,
        "decode_pos": DPOS,
        "base_vs_hf": {"prefill": round(pcc(bp, hf_p), 5), "decode": round(pcc(bd, hf_d), 5)},
        "opt_vs_hf": {"prefill": round(pcc(op, hf_p), 5), "decode": round(pcc(od, hf_d), 5)},
        "opt_vs_base": {"prefill": round(pcc(op, bp), 6), "decode": round(pcc(od, bd), 6)},
    }
    art = "/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/optimized_multichip_decoder"
    with open(f"{art}/opt_vs_base_layer{LAYER}{_MESH_TAG}.json", "w") as f:
        json.dump(rec, f, indent=2)
    print("PCC_JSON", json.dumps(rec))


if __name__ == "__main__":
    main()

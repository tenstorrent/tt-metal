"""Compare MultichipDecoder (1x4) vs single-chip OptimizedDecoder (1x1) and HF reference.

Runs both TTNN paths sequentially in one process with identical real weights + seeded
inputs, so the multichip output is validated against the single-chip TTNN optimized baseline
(the multichip skill's recommended comparison) as well as HF.

Usage:
    cd /tmp && TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
      PYTHONPATH=/home/ttuser/dev/tt-metal python .../tests/mc_compare.py <layer> <seq> <decode_pos>
"""
import json
import sys

import numpy as np
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_decoder import OptimizedDecoder

HIDDEN = 2048
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 32
DPOS = int(sys.argv[3]) if len(sys.argv) > 3 else SEQ


def pcc(a, b):
    a = a.flatten().float().numpy()
    b = b.flatten().float().numpy()
    return float(np.corrcoef(a, b)[0, 1])


def make_inputs():
    torch.manual_seed(SEQ)
    x = torch.randn(1, SEQ, HIDDEN) * 0.5
    xd = torch.randn(1, 1, HIDDEN) * 0.5
    return x, xd


def run(decoder_cls, mesh_shape, fabric):
    cfg = R.build_config()
    raw = W.load_layer_tensors(LAYER)
    ttnn.set_fabric_config(fabric)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape), trace_region_size=90_000_000)
    try:
        dec = decoder_cls.from_state_dict(raw, hf_config=cfg, layer_idx=LAYER, mesh_device=dev, max_seq_len=SEQ + 64)
        multichip = mesh_shape != (1, 1)
        mm = ttnn.ReplicateTensorToMesh(dev) if multichip else None

        def to_dev(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev, mesh_mapper=mm)

        def compose(t):
            if multichip:
                return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[0:1]
            return ttnn.to_torch(t)

        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=SEQ + 64, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        x, xd = make_inputs()
        pout = dec.prefill_forward(to_dev(x), kv, pt, user_id=0, start_pos=0)
        pref = compose(pout).float().reshape(1, SEQ, HIDDEN)
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
        dout = dec.decode_forward(to_dev(xd.reshape(1, 1, 1, HIDDEN)), cur, ridx, pt, kv)
        dref = compose(dout).float().reshape(1, 1, HIDDEN)
        return pref, dref
    finally:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    # HF reference
    cfg = R.build_config()
    raw = W.load_layer_tensors(LAYER)
    ctx = R.make_context(cfg, LAYER, state_dict=W.to_hf_layer_state_dict(raw, cfg, LAYER), dtype=torch.float32)
    x, xd = make_inputs()
    hf_pref, pkv = R.reference_forward(ctx, x)
    hf_dref, _ = R.reference_forward(ctx, xd, past_key_values=pkv)

    print(f"=== layer {LAYER} seq {SEQ} decode_pos {DPOS} ===")
    sc_p, sc_d = run(OptimizedDecoder, (1, 1), ttnn.FabricConfig.DISABLED)
    print(f"single-chip: prefill vs HF {pcc(sc_p, hf_pref):.5f}  decode vs HF {pcc(sc_d, hf_dref):.5f}")
    mc_p, mc_d = run(MultichipDecoder, (1, 4), ttnn.FabricConfig.FABRIC_1D_RING)
    print(f"multichip:   prefill vs HF {pcc(mc_p, hf_pref):.5f}  decode vs HF {pcc(mc_d, hf_dref):.5f}")
    print(f"multichip vs SINGLE-CHIP:  prefill {pcc(mc_p, sc_p):.6f}  decode {pcc(mc_d, sc_d):.6f}")
    rec = {
        "layer": LAYER,
        "seq": SEQ,
        "decode_pos": DPOS,
        "single_chip_vs_hf": {"prefill": round(pcc(sc_p, hf_pref), 5), "decode": round(pcc(sc_d, hf_dref), 5)},
        "multichip_vs_hf": {"prefill": round(pcc(mc_p, hf_pref), 5), "decode": round(pcc(mc_d, hf_dref), 5)},
        "multichip_vs_single_chip": {"prefill": round(pcc(mc_p, sc_p), 6), "decode": round(pcc(mc_d, sc_d), 6)},
    }
    art = "/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/multichip_decoder"
    with open(f"{art}/mc_vs_singlechip_layer{LAYER}.json", "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()

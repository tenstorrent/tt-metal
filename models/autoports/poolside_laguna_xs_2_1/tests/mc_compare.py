"""Compare a production MultichipDecoder on P150 D=1/2/4 with HF.

The default ``both`` mode retains the direct optimized-vs-multichip comparison only on D1,
where both decoders use the same topology. Cross-topology comparison is deliberately rejected:
run the D1 baseline and ``--mode mc`` target in fresh processes with a device reset between them.

Usage:
    python -m models.autoports.poolside_laguna_xs_2_1.tests.mc_compare \
      <layer> <seq> <decode_pos> --profile p150x2 --mode mc
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import (
    DOC_DIR,
    add_profile_args,
    close_mesh,
    open_mesh,
    profile_from_args,
    profile_summary,
    resolve_profile,
)
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_decoder import OptimizedDecoder

HIDDEN = 2048


def pcc(a, b):
    a = a.flatten().float().numpy()
    b = b.flatten().float().numpy()
    return float(np.corrcoef(a, b)[0, 1])


def make_inputs(seq):
    torch.manual_seed(seq)
    x = torch.randn(1, seq, HIDDEN) * 0.5
    xd = torch.randn(1, 1, HIDDEN) * 0.5
    return x, xd


def run(decoder_cls, profile, layer, seq, decode_pos):
    cfg = R.build_config()
    raw = W.load_layer_tensors(layer)
    dev = open_mesh(ttnn, profile)
    try:
        dec = decoder_cls.from_state_dict(raw, hf_config=cfg, layer_idx=layer, mesh_device=dev, max_seq_len=seq + 64)
        production = issubclass(decoder_cls, MultichipDecoder)
        if production and not dec.PACK_GATE_UP:
            raise AssertionError("comparison must exercise production PACK_GATE_UP=True")
        mm = ttnn.ReplicateTensorToMesh(dev) if production else None

        def to_dev(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev, mesh_mapper=mm)

        def compose(t):
            if production:
                return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[0:1]
            return ttnn.to_torch(t)

        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=seq + 64, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        x, xd = make_inputs(seq)
        pout = dec.prefill_forward(to_dev(x), kv, pt, user_id=0, start_pos=0)
        pref = compose(pout).float().reshape(1, seq, HIDDEN)
        cur = ttnn.from_torch(
            torch.tensor([decode_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        ridx = ttnn.from_torch(
            torch.tensor([[decode_pos]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        dout = dec.decode_forward(to_dev(xd.reshape(1, 1, 1, HIDDEN)), cur, ridx, pt, kv)
        dref = compose(dout).float().reshape(1, 1, HIDDEN)
        return pref, dref
    finally:
        close_mesh(ttnn, dev)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layer", type=int, nargs="?", default=0)
    parser.add_argument("seq", type=int, nargs="?", default=32)
    parser.add_argument("decode_pos", type=int, nargs="?", default=None)
    parser.add_argument("--mode", choices=("mc", "both"), default="both")
    parser.add_argument("--out", type=str, default=None)
    add_profile_args(parser, default_trace_region_size=90_000_000)
    args = parser.parse_args()
    profile = profile_from_args(args)
    layer, seq = args.layer, args.seq
    decode_pos = args.decode_pos if args.decode_pos is not None else seq
    if args.mode == "both" and profile.num_devices != 1:
        parser.error(
            "--mode both would open D1 and a different mesh in one process, which is not a valid "
            "qualification run. Run the D1 baseline and this target as separate fresh Python "
            "processes with 'tt-smi -r all' between them; use --mode mc for the target run."
        )

    # HF reference
    cfg = R.build_config()
    raw = W.load_layer_tensors(layer)
    ctx = R.make_context(cfg, layer, state_dict=W.to_hf_layer_state_dict(raw, cfg, layer), dtype=torch.float32)
    x, xd = make_inputs(seq)
    hf_pref, pkv = R.reference_forward(ctx, x)
    hf_dref, _ = R.reference_forward(ctx, xd, past_key_values=pkv)

    print(f"=== {profile.name} layer {layer} seq {seq} decode_pos {decode_pos} ===")
    sc_p = sc_d = None
    if args.mode == "both":
        single = resolve_profile("p150", trace_region_size=90_000_000, validate_visible_devices=False)
        sc_p, sc_d = run(OptimizedDecoder, single, layer, seq, decode_pos)
        print(f"single-chip: prefill vs HF {pcc(sc_p, hf_pref):.5f}  decode vs HF {pcc(sc_d, hf_dref):.5f}")
    mc_p, mc_d = run(MultichipDecoder, profile, layer, seq, decode_pos)
    print(f"production:  prefill vs HF {pcc(mc_p, hf_pref):.5f}  decode vs HF {pcc(mc_d, hf_dref):.5f}")
    rec = {
        **profile_summary(profile),
        "layer": layer,
        "seq": seq,
        "decode_pos": decode_pos,
        "pack_gate_up": True,
        "multichip_vs_hf": {"prefill": round(pcc(mc_p, hf_pref), 5), "decode": round(pcc(mc_d, hf_dref), 5)},
    }
    if args.mode == "both":
        print(f"production vs SINGLE-CHIP: prefill {pcc(mc_p, sc_p):.6f}  decode {pcc(mc_d, sc_d):.6f}")
        rec["single_chip_vs_hf"] = {
            "prefill": round(pcc(sc_p, hf_pref), 5),
            "decode": round(pcc(sc_d, hf_dref), 5),
        }
        rec["multichip_vs_single_chip"] = {
            "prefill": round(pcc(mc_p, sc_p), 6),
            "decode": round(pcc(mc_d, sc_d), 6),
        }
    suffix = "vs_singlechip" if args.mode == "both" else "vs_hf"
    out = Path(args.out) if args.out else DOC_DIR / "multichip_decoder" / f"{profile.name}_{suffix}_layer{layer}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()

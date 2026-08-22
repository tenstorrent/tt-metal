"""Warmed prefill + traced decode perf for one production Laguna layer on P150 D=1/2/4.

Mirrors tests/perf_trace_opt.py but drives packed ``MultichipDecoder`` on the selected profile.
The historical cross-topology ``both`` mode is rejected because D1 and D2/D4 must run in fresh
processes with a device reset between them. Measure D1 with ``perf_trace_opt.py`` and combine the
saved results offline.

Run under Tracy for the ops CSV (multichip device rows incl. CCL/AllReduce):
    python -m tracy -r -p -v -o <outdir> -m \
      models.autoports.poolside_laguna_xs_2_1.tests.perf_trace_mc \
      <layer> <prefill_len> <decode_iters> [mode] --profile p150x2
mode: "mc" (default, multichip only); "both" fails closed with reset instructions.
Signposts: PERF_PREFILL/_END, PERF_DECODE/_END. Wall-clock -> perf_walltime_mc_layer<L>.json.
"""
import argparse
import json
from pathlib import Path
import time

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import (
    DOC_DIR,
    add_profile_args,
    close_mesh,
    open_mesh,
    print_memory_snapshot,
    profile_from_args,
    profile_summary,
)
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder

try:
    from tracy import signpost
except Exception:

    def signpost(_):
        pass


HIDDEN = 2048
ART = DOC_DIR / "multichip_decoder"


def measure(decoder_cls, profile, layer, prefill_len, decode_iters, do_signpost):
    dev = open_mesh(ttnn, profile)
    production = issubclass(decoder_cls, MultichipDecoder)
    mm = ttnn.ReplicateTensorToMesh(dev) if production else None
    memory = []
    try:
        cfg = R.build_config()
        raw = W.load_layer_tensors(layer)
        max_seq = prefill_len + 64
        dec = decoder_cls.from_state_dict(raw, hf_config=cfg, layer_idx=layer, mesh_device=dev, max_seq_len=max_seq)
        if production and not dec.PACK_GATE_UP:
            raise AssertionError("performance qualification must use PACK_GATE_UP=True")
        memory.append(print_memory_snapshot(ttnn, dev, "layer_weights"))
        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=max_seq, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        memory.append(print_memory_snapshot(ttnn, dev, "layer_kv"))
        torch.manual_seed(0)
        x = torch.randn(1, prefill_len, HIDDEN) * 0.5
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=mm)

        dec.prefill_forward(xt, kv, pt, user_id=0, start_pos=0)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        if do_signpost:
            signpost("PERF_PREFILL")
        dec.prefill_forward(xt, kv, pt, user_id=0, start_pos=0)
        ttnn.synchronize_device(dev)
        if do_signpost:
            signpost("PERF_PREFILL_END")
        prefill_ms = (time.perf_counter() - t0) * 1e3

        xd = torch.randn(1, 1, 1, HIDDEN) * 0.5
        x_dev = ttnn.from_torch(xd, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=mm)
        cur = ttnn.from_torch(
            torch.tensor([prefill_len], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        ridx = ttnn.from_torch(
            torch.tensor([[prefill_len]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        dec.decode_forward(x_dev, cur, ridx, pt, kv)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        dec.decode_forward(x_dev, cur, ridx, pt, kv)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        memory.append(print_memory_snapshot(ttnn, dev, "layer_trace_capture", synchronize=False))
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        if do_signpost:
            signpost("PERF_DECODE")
        for _ in range(decode_iters):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        if do_signpost:
            signpost("PERF_DECODE_END")
        decode_ms = (time.perf_counter() - t0) * 1e3 / decode_iters
        ttnn.release_trace(dev, tid)
        return prefill_ms, decode_ms, memory
    finally:
        close_mesh(ttnn, dev)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layer", type=int, nargs="?", default=4)
    parser.add_argument("prefill_len", type=int, nargs="?", default=512)
    parser.add_argument("decode_iters", type=int, nargs="?", default=30)
    parser.add_argument("mode", choices=("mc", "both"), nargs="?", default="mc")
    parser.add_argument("--out", type=str, default=None)
    add_profile_args(parser, default_trace_region_size=200_000_000)
    args = parser.parse_args()
    profile = profile_from_args(args)
    layer, prefill_len, decode_iters = args.layer, args.prefill_len, args.decode_iters
    if prefill_len <= 0 or decode_iters <= 0:
        parser.error("prefill_len and decode_iters must be positive")
    if args.mode == "both":
        parser.error(
            "mode 'both' would open D1 and the selected mesh in one process, which is not a valid "
            "qualification run. Run perf_trace_opt.py for D1 and this command in mode 'mc' as "
            "separate fresh Python processes with 'tt-smi -r all' between them, then compare the "
            "saved JSON results offline."
        )

    cfg = R.build_config()
    res = {
        **profile_summary(profile),
        "layer": layer,
        "attention_type": cfg.layer_types[layer],
        "is_moe": layer not in cfg.mlp_only_layers,
        "pack_gate_up": True,
        "prefill_len": prefill_len,
        "decode_iters": decode_iters,
    }
    mc_p, mc_d, memory = measure(
        MultichipDecoder, profile, layer, prefill_len, decode_iters, do_signpost=True
    )
    res["multichip_prefill_ms"] = round(mc_p, 3)
    res["multichip_decode_ms_per_token"] = round(mc_d, 4)
    res["memory"] = memory
    out = Path(args.out) if args.out else ART / f"perf_walltime_{profile.name}_layer{layer}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("PERF_RESULT", json.dumps(res))


if __name__ == "__main__":
    main()

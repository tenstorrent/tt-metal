"""Warmed prefill + traced warmed decode perf for one Laguna layer, MULTICHIP (1x4 mesh).

Mirrors tests/perf_trace_opt.py (single-chip optimized) but drives MultichipDecoder on the 1x4
Blackhole mesh so before/after numbers use the same harness. Also (optionally) measures the
single-chip optimized baseline on a 1x1 mesh in the same process so speedup/efficiency are
computed against a co-measured baseline.

Run under Tracy for the ops CSV (multichip device rows incl. CCL/AllReduce):
    cd /tmp && TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
      PYTHONPATH=/home/ttuser/dev/tt-metal python -m tracy -r -p -v -o <outdir> \
      .../tests/perf_trace_mc.py <layer> <prefill_len> <decode_iters> [mode]
mode: "mc" (default, multichip only), "both" (single-chip + multichip + speedup).
Signposts: PERF_PREFILL/_END, PERF_DECODE/_END. Wall-clock -> perf_walltime_mc_layer<L>.json.
"""
import json
import sys
import time

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_decoder import OptimizedDecoder

try:
    from tracy import signpost
except Exception:

    def signpost(_):
        pass


LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 4
PREFILL_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 512
DECODE_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
MODE = sys.argv[4] if len(sys.argv) > 4 else "mc"
HIDDEN = 2048
ART = "/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/multichip_decoder"


def measure(decoder_cls, mesh_shape, fabric, do_signpost):
    ttnn.set_fabric_config(fabric)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape), trace_region_size=200_000_000)
    multichip = mesh_shape != (1, 1)
    mm = ttnn.ReplicateTensorToMesh(dev) if multichip else None
    try:
        cfg = R.build_config()
        raw = W.load_layer_tensors(LAYER)
        max_seq = PREFILL_LEN + 64
        dec = decoder_cls.from_state_dict(raw, hf_config=cfg, layer_idx=LAYER, mesh_device=dev, max_seq_len=max_seq)
        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=max_seq, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        torch.manual_seed(0)
        x = torch.randn(1, PREFILL_LEN, HIDDEN) * 0.5
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
            torch.tensor([PREFILL_LEN], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        ridx = ttnn.from_torch(
            torch.tensor([[PREFILL_LEN]], dtype=torch.int32),
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
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        if do_signpost:
            signpost("PERF_DECODE")
        for _ in range(DECODE_ITERS):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        if do_signpost:
            signpost("PERF_DECODE_END")
        decode_ms = (time.perf_counter() - t0) * 1e3 / DECODE_ITERS
        ttnn.release_trace(dev, tid)
        return prefill_ms, decode_ms
    finally:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    cfg = R.build_config()
    res = {
        "layer": LAYER,
        "attention_type": cfg.layer_types[LAYER],
        "is_moe": LAYER not in cfg.mlp_only_layers,
        "prefill_len": PREFILL_LEN,
        "decode_iters": DECODE_ITERS,
    }
    if MODE == "both":
        sc_p, sc_d = measure(OptimizedDecoder, (1, 1), ttnn.FabricConfig.DISABLED, do_signpost=False)
        res["single_chip_prefill_ms"] = round(sc_p, 3)
        res["single_chip_decode_ms_per_token"] = round(sc_d, 4)
    mc_p, mc_d = measure(MultichipDecoder, (1, 4), ttnn.FabricConfig.FABRIC_1D_RING, do_signpost=True)
    res["multichip_prefill_ms"] = round(mc_p, 3)
    res["multichip_decode_ms_per_token"] = round(mc_d, 4)
    if MODE == "both":
        res["decode_speedup"] = round(res["single_chip_decode_ms_per_token"] / res["multichip_decode_ms_per_token"], 3)
        res["decode_efficiency"] = round(res["decode_speedup"] / 4.0, 3)
        res["prefill_speedup"] = round(res["single_chip_prefill_ms"] / res["multichip_prefill_ms"], 3)
    with open(f"{ART}/perf_walltime_mc_layer{LAYER}.json", "w") as f:
        json.dump(res, f, indent=2)
    print("PERF_RESULT", json.dumps(res))


if __name__ == "__main__":
    main()

"""Before/after warmed prefill + traced decode: MultichipDecoder (before) vs
OptimizedMultichipDecoder (after), both on the 1x4 mesh, co-measured in one process.
Usage: perf_trace_omc.py <layer> <prefill_len> <decode_iters>"""
import json
import sys
import time

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_multichip_decoder import OptimizedMultichipDecoder

try:
    from tracy import signpost
except Exception:

    def signpost(_):
        pass


LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 4
PREFILL = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
SIGN = sys.argv[4] == "sign" if len(sys.argv) > 4 else False
HIDDEN = 2048
ART = "/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/optimized_multichip_decoder"


def measure(cls, do_sign):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    mm = ttnn.ReplicateTensorToMesh(dev)
    try:
        cfg = R.build_config()
        raw = W.load_layer_tensors(LAYER)
        maxs = PREFILL + 64
        dec = cls.from_state_dict(raw, hf_config=cfg, layer_idx=LAYER, mesh_device=dev, max_seq_len=maxs)
        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=maxs, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        torch.manual_seed(0)
        xt = ttnn.from_torch(
            torch.randn(1, PREFILL, HIDDEN) * 0.5,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        dec.prefill_forward(xt, kv, pt, user_id=0, start_pos=0)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        if do_sign:
            signpost("PERF_PREFILL")
        dec.prefill_forward(xt, kv, pt, user_id=0, start_pos=0)
        ttnn.synchronize_device(dev)
        if do_sign:
            signpost("PERF_PREFILL_END")
        prefill_ms = (time.perf_counter() - t0) * 1e3
        x_dev = ttnn.from_torch(
            torch.randn(1, 1, 1, HIDDEN) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=mm
        )
        cur = ttnn.from_torch(
            torch.tensor([PREFILL], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        ridx = ttnn.from_torch(
            torch.tensor([[PREFILL]], dtype=torch.int32),
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
        if do_sign:
            signpost("PERF_DECODE")
        for _ in range(ITERS):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        if do_sign:
            signpost("PERF_DECODE_END")
        decode_ms = (time.perf_counter() - t0) * 1e3 / ITERS
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
        "prefill_len": PREFILL,
        "decode_iters": ITERS,
    }
    afteronly = len(sys.argv) > 4 and sys.argv[4] == "afteronly"
    if afteronly:
        op, od = measure(OptimizedMultichipDecoder, True)
        res["after_optimized"] = {"prefill_ms": round(op, 3), "decode_ms_per_token": round(od, 4)}
        print("PERF_RESULT", json.dumps(res))
        return
    bp, bd = measure(MultichipDecoder, False)
    op, od = measure(OptimizedMultichipDecoder, SIGN)
    res["before_multichip"] = {"prefill_ms": round(bp, 3), "decode_ms_per_token": round(bd, 4)}
    res["after_optimized"] = {"prefill_ms": round(op, 3), "decode_ms_per_token": round(od, 4)}
    res["decode_speedup"] = round(bd / od, 4)
    res["prefill_speedup"] = round(bp / op, 4)
    with open(f"{ART}/perf_before_after_layer{LAYER}.json", "w") as f:
        json.dump(res, f, indent=2)
    print("PERF_RESULT", json.dumps(res))


if __name__ == "__main__":
    main()

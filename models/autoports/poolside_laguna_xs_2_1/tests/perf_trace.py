"""Warmed prefill + traced warmed decode performance harness for one Laguna layer.

Run under Tracy to emit an ops CSV with signposted measured windows:
    python -m tracy -r -p -v -o <outdir> -m \
      models.autoports.poolside_laguna_xs_2_1.tests.perf_trace \
      <layer> <prefill_len> <decode_iters>

Signposts: PERF_PREFILL/PERF_PREFILL_END around the warmed prefill; PERF_DECODE/
PERF_DECODE_END around the measured traced decode replays. Wall-clock latencies are
written to perf_walltime_layer<L>.json in the artifact dir.
"""
import json
import sys
import time

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import DOC_DIR
from models.autoports.poolside_laguna_xs_2_1.tt.functional_decoder import FunctionalDecoder

try:
    from tracy import signpost
except Exception:

    def signpost(_):
        pass


LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PREFILL_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 512
DECODE_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
HIDDEN = 2048
ART = DOC_DIR / "functional_decoder"


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        cfg = R.build_config()
        raw = W.load_layer_tensors(LAYER)
        max_seq = PREFILL_LEN + 64
        dec = FunctionalDecoder.from_state_dict(
            raw, hf_config=cfg, layer_idx=LAYER, mesh_device=dev, max_seq_len=max_seq
        )
        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=max_seq, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        torch.manual_seed(0)
        x = torch.randn(1, PREFILL_LEN, HIDDEN) * 0.5
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        # ---- warmed prefill ----
        dec.prefill_forward(xt, kv, pt, user_id=0, start_pos=0)  # compile
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        signpost("PERF_PREFILL")
        outp = dec.prefill_forward(xt, kv, pt, user_id=0, start_pos=0)
        ttnn.synchronize_device(dev)
        signpost("PERF_PREFILL_END")
        prefill_ms = (time.perf_counter() - t0) * 1e3

        # ---- traced warmed decode ----
        xd = torch.randn(1, 1, 1, HIDDEN) * 0.5
        x_dev = ttnn.from_torch(xd, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cur = ttnn.from_torch(
            torch.tensor([PREFILL_LEN], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
        )
        ridx = ttnn.from_torch(
            torch.tensor([[PREFILL_LEN]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
        )
        # compile
        dec.decode_forward(x_dev, cur, ridx, pt, kv)
        ttnn.synchronize_device(dev)
        # capture trace
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out_trace = dec.decode_forward(x_dev, cur, ridx, pt, kv)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        # warm replay
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        # measured
        t0 = time.perf_counter()
        signpost("PERF_DECODE")
        for _ in range(DECODE_ITERS):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        signpost("PERF_DECODE_END")
        decode_ms = (time.perf_counter() - t0) * 1e3 / DECODE_ITERS

        ttnn.release_trace(dev, tid)
        res = {
            "layer": LAYER,
            "attention_type": cfg.layer_types[LAYER],
            "is_moe": LAYER not in cfg.mlp_only_layers,
            "prefill_len": PREFILL_LEN,
            "prefill_ms_warmed": round(prefill_ms, 3),
            "decode_ms_per_token_traced": round(decode_ms, 4),
            "decode_iters": DECODE_ITERS,
        }
        with open(ART / f"perf_walltime_layer{LAYER}.json", "w") as f:
            json.dump(res, f, indent=2)
        print("PERF_RESULT", json.dumps(res))
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
